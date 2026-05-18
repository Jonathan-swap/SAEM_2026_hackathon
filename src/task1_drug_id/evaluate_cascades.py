"""Evaluate the cascade-variants comparison.

Reads the summary CSV written by ``compare_cascades.py`` and produces:

  - Per-(split, metric, model) winning architecture rank table
  - Win-count table across the 18 (3 models × 3 metrics × 2 splits)
    head-to-head comparisons
  - Per-architecture mean / std / best macro AUC and PR-AUC
  - Per-class winning architecture (for the 4 OvR AUCs)
  - Stability score: variance of an architecture's macro AUC across
    model families (lower = more model-agnostic)
  - Consistency check: does each cascade win on *both* CV and
    holdout, or only one?
  - Plain-text deployment recommendation backed by the data

Output:
  derived/task1_cascade_evaluation_report.md
  derived/task1_cascade_evaluation_rankings.csv

No model fitting — pure analysis of pre-computed metrics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
SOURCE = DERIVED / "task1_cascade_combinations_summary.csv"

ARCHS = ["direct", "casc_A_tier12", "casc_B_K_prev", "casc_C_K_TC"]
ARCH_LABEL = {
    "direct":         "Direct 4-class",
    "casc_A_tier12":  "Cascade A (tier-1 + tier-2-multi)",
    "casc_B_K_prev":  "Cascade B (tier-1 + K-vs-rest + prev)",
    "casc_C_K_TC":    "Cascade C (tier-1 + K-vs-rest + T-vs-C)",
}
MODELS = ["logreg", "rforest", "hgb"]
SPLITS = ["cv", "temporal"]
HEADLINE_METRICS = ["macro_auc", "macro_prauc", "accuracy"]
PER_CLASS_AUC = ["auc_none", "auc_kraken", "auc_triton", "auc_coral"]


# ---------- Loading ---------------------------------------------------

def load_summary() -> pd.DataFrame:
    if not SOURCE.exists():
        raise SystemExit(
            f"FAIL: {SOURCE} not found. Run "
            f"src/task1_drug_id/compare_cascades.py first to "
            f"produce the comparison data.")
    df = pd.read_csv(SOURCE)
    needed = {"split", "model", "arch", "macro_auc", "macro_prauc",
              "accuracy", "logloss"} | set(PER_CLASS_AUC)
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(
            f"FAIL: summary CSV is missing columns: {missing}")
    print(f"Loaded {len(df)} rows from {SOURCE}")
    return df


# ---------- Analyses --------------------------------------------------

def rank_table(df: pd.DataFrame, metric: str,
                 higher_is_better: bool = True) -> pd.DataFrame:
    """Per (split, model) the rank of each architecture on ``metric``.

    Returns a long-form DataFrame with columns
    split, model, arch, value, rank (1 = best).
    """
    rows = []
    for split in SPLITS:
        for model in MODELS:
            sub = df[(df["split"] == split) & (df["model"] == model)]
            ranks = sub[metric].rank(
                ascending=not higher_is_better, method="min")
            for _, r in sub.iterrows():
                rows.append({
                    "split": split, "model": model,
                    "arch": r["arch"],
                    "metric": metric,
                    "value": float(r[metric]),
                    "rank": int(ranks.loc[r.name]),
                })
    return pd.DataFrame(rows)


def win_counts(df: pd.DataFrame) -> pd.DataFrame:
    """How many (split, model, metric) cells does each architecture
    rank #1 on? Max possible = len(SPLITS) * len(MODELS) * len(metrics)
    = 2 * 3 * 3 = 18 for the headline metrics."""
    all_ranks = pd.concat(
        [rank_table(df, m, higher_is_better=(m != "logloss"))
         for m in HEADLINE_METRICS + ["logloss"]])
    wins = (all_ranks[all_ranks["rank"] == 1]
              .groupby("arch").size()
              .reindex(ARCHS, fill_value=0).rename("wins"))
    total = (all_ranks.groupby("arch").size()
               .reindex(ARCHS, fill_value=0).rename("total"))
    out = pd.concat([wins, total], axis=1)
    out["win_rate"] = out["wins"] / out["total"]
    return out


def per_arch_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCHS:
        for split in SPLITS:
            sub = df[(df["arch"] == arch) & (df["split"] == split)]
            for m in HEADLINE_METRICS:
                vals = sub[m].to_numpy()
                rows.append({
                    "arch": arch, "split": split, "metric": m,
                    "mean":   float(np.mean(vals)),
                    "std":    float(np.std(vals)),
                    "best":   float(np.max(vals)),
                    "worst":  float(np.min(vals)),
                    "n_models": len(vals),
                })
    return pd.DataFrame(rows)


def per_class_winners(df: pd.DataFrame) -> pd.DataFrame:
    """For each (split, model) and each class's OvR AUC, the winner."""
    rows = []
    for split in SPLITS:
        for model in MODELS:
            sub = df[(df["split"] == split) & (df["model"] == model)]
            for col in PER_CLASS_AUC:
                best_arch = sub.loc[sub[col].idxmax(), "arch"]
                best_val = float(sub[col].max())
                rows.append({
                    "split": split, "model": model,
                    "class_auc": col,
                    "winning_arch": best_arch,
                    "winning_value": best_val,
                })
    return pd.DataFrame(rows)


def stability_score(df: pd.DataFrame) -> pd.DataFrame:
    """Variance of an architecture's macro AUC across the 3 model
    families, per split. Lower = more model-agnostic."""
    rows = []
    for split in SPLITS:
        sub = df[df["split"] == split]
        for arch in ARCHS:
            vals = sub[sub["arch"] == arch]["macro_auc"].to_numpy()
            rows.append({
                "arch": arch, "split": split,
                "macro_auc_mean": float(np.mean(vals)),
                "macro_auc_std":  float(np.std(vals)),
                "macro_auc_spread": float(np.max(vals) - np.min(vals)),
            })
    return pd.DataFrame(rows)


def consistency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Does each architecture beat direct on both CV and holdout?"""
    rows = []
    for model in MODELS:
        d_cv  = df[(df["model"] == model) & (df["split"] == "cv")
                    & (df["arch"] == "direct")].iloc[0]
        d_tm  = df[(df["model"] == model) & (df["split"] == "temporal")
                    & (df["arch"] == "direct")].iloc[0]
        for arch in ARCHS:
            if arch == "direct":
                continue
            c_cv = df[(df["model"] == model) & (df["split"] == "cv")
                        & (df["arch"] == arch)].iloc[0]
            c_tm = df[(df["model"] == model) & (df["split"] == "temporal")
                        & (df["arch"] == arch)].iloc[0]
            rows.append({
                "model": model, "arch": arch,
                "cv_delta_auc":       float(c_cv["macro_auc"] - d_cv["macro_auc"]),
                "temporal_delta_auc": float(c_tm["macro_auc"] - d_tm["macro_auc"]),
                "cv_beats_direct":       bool(c_cv["macro_auc"] > d_cv["macro_auc"]),
                "temporal_beats_direct": bool(c_tm["macro_auc"] > d_tm["macro_auc"]),
                "both_beat_direct":   bool(c_cv["macro_auc"] > d_cv["macro_auc"]
                                            and c_tm["macro_auc"] > d_tm["macro_auc"]),
            })
    return pd.DataFrame(rows)


# ---------- Report ----------------------------------------------------

def write_report(df: pd.DataFrame) -> None:
    wins = win_counts(df)
    summary = per_arch_summary(df)
    class_winners = per_class_winners(df)
    stability = stability_score(df)
    consistency = consistency_table(df)

    # Persist rankings for downstream / spreadsheet use
    all_ranks = pd.concat(
        [rank_table(df, m, higher_is_better=(m != "logloss"))
         for m in HEADLINE_METRICS + ["logloss"]])
    all_ranks.to_csv(
        DERIVED / "task1_cascade_evaluation_rankings.csv", index=False)

    lines = []
    lines.append("# Task 1 cascade-variants evaluation\n")
    lines.append("Pure analysis layer over the metrics emitted by "
                  "`src/task1_drug_id/compare_cascades.py`. No model "
                  "fitting here — read `task1_cascade_combinations_"
                  "summary.csv` and compute rankings / win counts / "
                  "consistency.\n")

    # 1. Win counts ----
    lines.append("## 1. Headline metric wins per architecture\n")
    lines.append("Counts the number of (split × model × metric) cells "
                  "where each architecture ranks #1 across the four "
                  "headline metrics (macro ROC-AUC, macro PR-AUC, "
                  "accuracy, log-loss). Total cells = 2 splits × 3 "
                  "models × 4 metrics = 24.\n")
    lines.append("| Architecture | Wins | Win rate |")
    lines.append("|---|---:|---:|")
    for arch in sorted(wins.index, key=lambda a: -wins.loc[a, "wins"]):
        lines.append(f"| {ARCH_LABEL[arch]} | "
                      f"{int(wins.loc[arch, 'wins'])} | "
                      f"{wins.loc[arch, 'win_rate']*100:.0f}% |")
    lines.append("")

    # 2. Per-arch summary ----
    lines.append("## 2. Per-architecture summary (across 3 models)\n")
    for split in SPLITS:
        lines.append(f"### {split.upper()} split\n")
        lines.append("| Architecture | Macro AUC mean | best | std | "
                      "Macro PR-AUC mean | best | std | Accuracy mean | best |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arch in ARCHS:
            r_auc = summary[(summary["arch"] == arch)
                              & (summary["split"] == split)
                              & (summary["metric"] == "macro_auc")].iloc[0]
            r_pra = summary[(summary["arch"] == arch)
                              & (summary["split"] == split)
                              & (summary["metric"] == "macro_prauc")].iloc[0]
            r_acc = summary[(summary["arch"] == arch)
                              & (summary["split"] == split)
                              & (summary["metric"] == "accuracy")].iloc[0]
            lines.append(
                f"| {ARCH_LABEL[arch]} | "
                f"{r_auc['mean']:.4f} | {r_auc['best']:.4f} | {r_auc['std']:.4f} | "
                f"{r_pra['mean']:.4f} | {r_pra['best']:.4f} | {r_pra['std']:.4f} | "
                f"{r_acc['mean']:.4f} | {r_acc['best']:.4f} |")
        lines.append("")

    # 3. Stability ----
    lines.append("## 3. Stability across model families\n")
    lines.append("How much does macro AUC vary when you swap "
                  "logreg ↔ rforest ↔ hgb under the same "
                  "architecture? Lower spread = more model-agnostic.\n")
    for split in SPLITS:
        lines.append(f"### {split.upper()} split\n")
        lines.append("| Architecture | mean | std | spread |")
        lines.append("|---|---:|---:|---:|")
        sub = stability[stability["split"] == split].sort_values(
            "macro_auc_spread")
        for _, r in sub.iterrows():
            lines.append(f"| {ARCH_LABEL[r['arch']]} | "
                          f"{r['macro_auc_mean']:.4f} | "
                          f"{r['macro_auc_std']:.4f} | "
                          f"{r['macro_auc_spread']:.4f} |")
        lines.append("")

    # 4. Per-class winners ----
    lines.append("## 4. Per-class OvR AUC winners (holdout only)\n")
    lines.append("Which architecture has the highest one-vs-rest "
                  "ROC-AUC for each class?\n")
    lines.append("| Model | Class | Winning architecture | AUC |")
    lines.append("|---|---|---|---:|")
    for _, r in class_winners[class_winners["split"] == "temporal"].iterrows():
        cls = r["class_auc"].replace("auc_", "").capitalize()
        lines.append(f"| {r['model']} | {cls} | "
                      f"{ARCH_LABEL[r['winning_arch']]} | "
                      f"{r['winning_value']:.3f} |")
    lines.append("")

    # 5. Consistency ----
    lines.append("## 5. Consistency — does each cascade beat direct on "
                  "**both** CV and holdout?\n")
    lines.append("| Model | Cascade | Δ AUC (CV) | Δ AUC (holdout) | "
                  "Beats direct on CV? | Holdout? | Both? |")
    lines.append("|---|---|---:|---:|---|---|---|")
    for _, r in consistency.iterrows():
        lines.append(f"| {r['model']} | {ARCH_LABEL[r['arch']]} | "
                      f"{r['cv_delta_auc']:+.4f} | "
                      f"{r['temporal_delta_auc']:+.4f} | "
                      f"{'YES' if r['cv_beats_direct'] else 'no'} | "
                      f"{'YES' if r['temporal_beats_direct'] else 'no'} | "
                      f"{'YES' if r['both_beat_direct'] else 'no'} |")
    lines.append("")

    # 6. Recommendation ----
    both_winners = consistency[consistency["both_beat_direct"]]
    lines.append("## 6. Deployment recommendation\n")
    if both_winners.empty:
        lines.append("No cascade beats direct on *both* CV and holdout "
                      "for any single model. Stay with direct 4-class.\n")
    else:
        # Best mean-of-(cv, holdout) Δ AUC among consistent winners
        both_winners = both_winners.copy()
        both_winners["mean_delta"] = (both_winners["cv_delta_auc"]
                                        + both_winners["temporal_delta_auc"]) / 2
        best = both_winners.sort_values("mean_delta",
                                          ascending=False).iloc[0]
        lines.append(
            f"**Recommended: `{best['model']}` × **{ARCH_LABEL[best['arch']]}**.**\n")
        lines.append(f"This (model × architecture) pair beats direct "
                      f"4-class on *both* the 5-fold CV "
                      f"(Δ = {best['cv_delta_auc']:+.4f}) and the temporal "
                      f"holdout (Δ = {best['temporal_delta_auc']:+.4f}). "
                      f"Among all consistent winners it has the highest "
                      f"mean Δ macro AUC across the two splits "
                      f"({best['mean_delta']:+.4f}).\n")
        lines.append("Other architectures that also beat direct on both "
                      "splits (for at least one model):")
        for _, r in both_winners.iterrows():
            if r["arch"] == best["arch"] and r["model"] == best["model"]:
                continue
            lines.append(
                f"- `{r['model']}` × {ARCH_LABEL[r['arch']]}: "
                f"CV Δ {r['cv_delta_auc']:+.4f}, "
                f"holdout Δ {r['temporal_delta_auc']:+.4f}")
        lines.append("")

    out = DERIVED / "task1_cascade_evaluation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out}")
    print(f"Wrote: {DERIVED / 'task1_cascade_evaluation_rankings.csv'}")


# ---------- Main ------------------------------------------------------

def main() -> None:
    df = load_summary()
    write_report(df)

    # Print a one-screen summary too
    print("\n" + "=" * 78)
    print("Win counts (rank-#1 across split × model × metric, 24 cells)")
    print("=" * 78)
    wins = win_counts(df).sort_values("wins", ascending=False)
    for arch in wins.index:
        print(f"  {ARCH_LABEL[arch]:46s}  "
              f"{int(wins.loc[arch, 'wins']):>2d} / {int(wins.loc[arch, 'total'])}")

    print("\n" + "=" * 78)
    print("Consistency — cascades that beat direct on BOTH CV and holdout")
    print("=" * 78)
    cons = consistency_table(df)
    both = cons[cons["both_beat_direct"]]
    if both.empty:
        print("  None. Stay with direct 4-class.")
    else:
        for _, r in both.iterrows():
            print(f"  {r['model']:8s} x {ARCH_LABEL[r['arch']]:46s}  "
                  f"CV {r['cv_delta_auc']:+.4f}  "
                  f"holdout {r['temporal_delta_auc']:+.4f}")


if __name__ == "__main__":
    main()
