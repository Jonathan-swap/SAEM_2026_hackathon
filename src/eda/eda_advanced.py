"""Advanced EDA — data integrity + missingness + feature importance.

Complements `eda_initial.py` (which inspects raw xlsx + codebook).
This script operates on the engineered feature tables and labels:
  derived/features_triage.csv
  derived/features_fourh.csv
  derived/probs_avg.csv
  derived/derived_labels.csv (optional)

Produces:
  derived/eda_advanced_report.md    — markdown summary
  derived/eda_plots/                — PNGs

Sections:
  A. Missingness  (per-column + by-class)
  B. Data integrity  (duplicates / constants / out-of-range / correlation)
  C. Univariate feature importance  (mutual information vs each task target)
  D. Multivariate (tree-based) feature importance
  E. Target distribution + cross-tabs
  F. Note-feature coverage diagnostics
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
PLOTS = DERIVED / "eda_plots"
PLOTS.mkdir(parents=True, exist_ok=True)
REPORT = DERIVED / "eda_advanced_report.md"

CRITICAL = {
    "triage_heart_rate":            (40, 200),
    "triage_respiratory_rate":      (8, 50),
    "triage_snapshot.systolic_bp":  (60, 240),
    "triage_snapshot.diastolic_bp": (40, 130),
    "triage_snapshot.oxygen_saturation": (70, 101),
    "triage_temperature_c":         (34.0, 41.0),
    "triage_gcs":                   (3, 15),
    "triage_age":                   (0, 110),
}


# ---------- helpers ---------------------------------------------------

def numerify(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encode object/categorical columns to numeric for analysis."""
    df = df.copy()
    drop = [c for c in df.columns
            if c in {"encounter_id", "encounter_arrival_date"}]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    text_col = "triage_brief_note" if "triage_brief_note" in obj_cols else None
    if text_col:
        df = df.drop(columns=[text_col])
        obj_cols = [c for c in obj_cols if c != text_col]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, dummy_na=True)
    bools = df.select_dtypes(include="bool").columns.tolist()
    for c in bools:
        df[c] = df[c].astype(float)
    return df, obj_cols


def section_a_missingness(rep: list[str], triage: pd.DataFrame,
                            fourh: pd.DataFrame,
                            argmax: pd.Series, dispo: pd.Series) -> None:
    rep.append("## A. Missingness\n")

    for name, df in [("features_triage", triage), ("features_fourh", fourh)]:
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        n = len(df)
        rep.append(f"### {name}.csv — {df.shape[0]} rows × {df.shape[1]} cols")
        if miss.empty:
            rep.append("- No missing values in any column.")
        else:
            rep.append(f"- Columns with missing values: **{len(miss)}** "
                       f"(out of {df.shape[1]})")
            rep.append("- Top-20 most-missing columns:\n")
            rep.append("| Column | Missing | % |")
            rep.append("|---|---:|---:|")
            for col, c in miss.head(20).items():
                rep.append(f"| {col} | {c} | {c/n*100:.1f}% |")
        rep.append("")

    # By-class missingness for selected interesting columns
    rep.append("### By-class missingness (4h features, key analytes)")
    key_cols = [c for c in fourh.columns if
                c in {"ed_course_reassessment_4h.lactate_4h",
                       "ed_course_reassessment_4h.cpk_4h",
                       "ed_course_reassessment_4h.vbg_ph_4h",
                       "ed_course_reassessment_4h.troponin_4h",
                       "lts_lactate_was_drawn",
                       "lts_troponin_was_drawn"}]
    if key_cols and argmax is not None:
        rep.append("\n| Column | Kraken | Triton | Coral | None |")
        rep.append("|---|---:|---:|---:|---:|")
        for col in key_cols:
            row = [col]
            for cls in ["Kraken Candy", "Triton Tabs", "Coral Dust", "None"]:
                mask = (argmax == cls)
                if mask.sum() == 0:
                    row.append("—")
                else:
                    pct = fourh.loc[mask, col].isna().mean() * 100 \
                        if col in fourh.columns else float("nan")
                    row.append(f"{pct:.0f}%")
            rep.append("| " + " | ".join(str(x) for x in row) + " |")
    rep.append("")


def section_b_integrity(rep: list[str], triage: pd.DataFrame,
                          fourh: pd.DataFrame) -> None:
    rep.append("## B. Data integrity\n")

    # Duplicates
    dup_triage = triage["encounter_id"].duplicated().sum()
    dup_fourh = fourh["encounter_id"].duplicated().sum()
    rep.append(f"- Duplicate `encounter_id` rows: "
               f"triage={dup_triage}, fourh={dup_fourh}")

    # Triage vs fourh encounter-set parity
    set_t = set(triage["encounter_id"])
    set_f = set(fourh["encounter_id"])
    rep.append(f"- encounter_id parity: triage∖fourh = {len(set_t - set_f)}, "
               f"fourh∖triage = {len(set_f - set_t)}")

    # Constant / near-constant columns
    for name, df in [("features_triage", triage), ("features_fourh", fourh)]:
        num = df.select_dtypes(include="number")
        const = []
        near_const = []
        for col in num.columns:
            s = num[col].dropna()
            if s.empty:
                continue
            if s.nunique() == 1:
                const.append(col)
            else:
                top_share = s.value_counts(normalize=True).iloc[0]
                if top_share > 0.95:
                    near_const.append((col, float(top_share)))
        rep.append(f"\n### {name}: constants and near-constants")
        rep.append(f"- Constant numeric columns ({len(const)}): "
                   f"{const[:10]}{' ...' if len(const) > 10 else ''}")
        rep.append(f"- Near-constant (>95% same value): {len(near_const)}")
        for c, share in near_const[:10]:
            rep.append(f"  - `{c}` ({share*100:.1f}%)")

    # Out-of-range vital checks
    rep.append("\n### Out-of-range vitals (triage)")
    for col, (lo, hi) in CRITICAL.items():
        if col not in triage.columns:
            continue
        s = pd.to_numeric(triage[col], errors="coerce")
        n_low = int((s < lo).sum())
        n_hi = int((s > hi).sum())
        if n_low or n_hi:
            rep.append(f"- `{col}` in [{lo}, {hi}]: {n_low} below, {n_hi} above")
    rep.append("")

    # Highly correlated pairs (|r| > 0.9) on features_fourh numeric subset
    rep.append("### Highly correlated pairs (features_fourh, |r| > 0.9)")
    num = fourh.select_dtypes(include="number")
    # Cap to a manageable subset for speed
    if num.shape[1] > 200:
        num = num.sample(n=200, axis=1, random_state=42)
        rep.append("- (sampled 200 numeric columns for the correlation check)")
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    high = upper.stack().sort_values(ascending=False)
    high = high[high > 0.9].head(20)
    if high.empty:
        rep.append("- None found in sample.")
    else:
        rep.append("\n| Feature A | Feature B | \\|r\\| |")
        rep.append("|---|---|---:|")
        for (a, b), r in high.items():
            rep.append(f"| {a} | {b} | {r:.3f} |")
    rep.append("")


def section_c_univariate(rep: list[str], X_t: pd.DataFrame,
                          X_f: pd.DataFrame, y_drug: np.ndarray,
                          y_dispo: np.ndarray) -> None:
    rep.append("## C. Univariate feature importance (mutual information)\n")

    for name, X_df, y, target in [
        ("Task 1 — Drug class (argmax of probs_avg)",
         X_t, y_drug, "drug"),
        ("Task 2 — Disposition (Discharge/Floor/ICU, drug-positive cohort)",
         X_f, y_dispo, "dispo"),
    ]:
        rep.append(f"### {name}\n")
        if y is None or len(y) == 0:
            rep.append("(no target)\n")
            continue
        X_num, _ = numerify(X_df)
        # align rows: y has the same index as the source df
        X_num = X_num.reset_index(drop=True)
        # Drop columns that are all-NaN after numerify
        keep = [c for c in X_num.columns if X_num[c].notna().any()]
        X_num = X_num[keep]
        imp = SimpleImputer(strategy="median")
        X_arr = imp.fit_transform(X_num.values)
        mi = mutual_info_classif(X_arr, y, random_state=42)
        ranked = pd.Series(mi, index=X_num.columns).sort_values(ascending=False)
        rep.append("Top 20 features by MI:\n")
        rep.append("| Rank | Feature | MI |")
        rep.append("|---:|---|---:|")
        for i, (col, v) in enumerate(ranked.head(20).items(), 1):
            rep.append(f"| {i} | {col} | {v:.4f} |")
        rep.append("")
        # Plot
        fig, ax = plt.subplots(figsize=(7, 6))
        ranked.head(20)[::-1].plot.barh(ax=ax)
        ax.set_title(f"MI ranking — {target}")
        ax.set_xlabel("Mutual information (nats)")
        fig.tight_layout()
        path = PLOTS / f"feature_importance_mi_{target}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        rep.append(f"![{target} MI]({path.relative_to(DERIVED).as_posix()})\n")


def section_d_tree_importance(rep: list[str], X_t: pd.DataFrame,
                                X_f: pd.DataFrame, y_drug: np.ndarray,
                                y_dispo: np.ndarray) -> None:
    rep.append("## D. Tree-based (multivariate) feature importance\n")

    for name, X_df, y, target in [
        ("Task 1 — Drug class", X_t, y_drug, "drug"),
        ("Task 2 — Disposition", X_f, y_dispo, "dispo"),
    ]:
        rep.append(f"### {name}\n")
        X_num, _ = numerify(X_df)
        keep = [c for c in X_num.columns if X_num[c].notna().any()]
        X_num = X_num[keep]
        imp = SimpleImputer(strategy="median")
        X_arr = imp.fit_transform(X_num.values)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                      random_state=42, n_jobs=-1)
        rf.fit(X_arr, y)
        ranked = pd.Series(rf.feature_importances_,
                             index=X_num.columns).sort_values(ascending=False)
        rep.append("Top 20 features by RF importance (Gini):\n")
        rep.append("| Rank | Feature | Importance |")
        rep.append("|---:|---|---:|")
        for i, (col, v) in enumerate(ranked.head(20).items(), 1):
            rep.append(f"| {i} | {col} | {v:.4f} |")
        rep.append("")
        fig, ax = plt.subplots(figsize=(7, 6))
        ranked.head(20)[::-1].plot.barh(ax=ax)
        ax.set_title(f"RF importance — {target}")
        ax.set_xlabel("Gini importance")
        fig.tight_layout()
        path = PLOTS / f"feature_importance_rf_{target}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        rep.append(f"![{target} RF]({path.relative_to(DERIVED).as_posix()})\n")


def section_e_targets(rep: list[str], argmax: pd.Series,
                        dispo: pd.Series) -> None:
    rep.append("## E. Target distributions\n")
    if argmax is not None:
        rep.append("### Drug class (argmax of probs_avg)")
        vc = argmax.value_counts().to_string()
        rep.append("```\n" + vc + "\n```")
    if dispo is not None:
        rep.append("\n### Disposition")
        rep.append("```\n" + dispo.value_counts().to_string() + "\n```")
    if argmax is not None and dispo is not None:
        rep.append("\n### Drug × Disposition cross-tab")
        ct = pd.crosstab(argmax, dispo)
        rep.append("```\n" + ct.to_string() + "\n```")
    rep.append("")


def section_f_notes(rep: list[str], triage: pd.DataFrame,
                      argmax: pd.Series) -> None:
    rep.append("## F. Note-feature coverage\n")
    if "note_has_onset_phrase" not in triage.columns:
        rep.append("(note features not present)\n")
        return
    n = len(triage)
    n_onset = int(triage["note_has_onset_phrase"].sum())
    n_fest = int(triage["note_is_festival_template"].sum())
    rep.append(f"- Onset phrase parsed: **{n_onset}/{n} ({n_onset/n*100:.1f}%)**")
    rep.append(f"- Festival template:   **{n_fest}/{n} ({n_fest/n*100:.1f}%)**")
    onset = triage["note_onset_minutes"].dropna()
    if not onset.empty:
        rep.append(f"- onset_minutes range: [{onset.min():.0f}, "
                   f"{onset.max():.0f}], median {onset.median():.0f}")
    rep.append("")
    if argmax is not None:
        rep.append("### Mean onset_minutes by drug class\n")
        rep.append("| Class | n with onset | mean | median |")
        rep.append("|---|---:|---:|---:|")
        for cls in ["Kraken Candy", "Triton Tabs", "Coral Dust", "None"]:
            mask = (argmax == cls)
            o = triage.loc[mask, "note_onset_minutes"].dropna()
            if o.empty:
                rep.append(f"| {cls} | 0 | — | — |")
            else:
                rep.append(f"| {cls} | {len(o)} | {o.mean():.0f} | "
                           f"{o.median():.0f} |")
        # Plot histogram by class
        fig, ax = plt.subplots(figsize=(8, 5))
        for cls, col in [("Kraken Candy", "tab:red"),
                          ("Triton Tabs", "tab:blue"),
                          ("Coral Dust", "tab:green"),
                          ("None", "tab:gray")]:
            mask = (argmax == cls)
            o = triage.loc[mask, "note_onset_minutes"].dropna()
            if not o.empty:
                ax.hist(o, bins=15, alpha=0.5, label=cls, color=col)
        ax.set_xlabel("onset minutes")
        ax.set_ylabel("count")
        ax.set_title("Onset-minutes distribution by inferred drug class")
        ax.legend()
        fig.tight_layout()
        path = PLOTS / "onset_minutes_by_drug_class.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        rep.append("")
        rep.append(f"![onset by class]({path.relative_to(DERIVED).as_posix()})")
    rep.append("")


# ---------- main ------------------------------------------------------

def main() -> None:
    rep: list[str] = []
    rep.append("# Advanced EDA — Data Integrity, Missingness, Feature Importance\n")
    rep.append("Auto-generated. Re-run: "
               "`./.venv/Scripts/python.exe src/eda/eda_advanced.py`\n")

    triage = pd.read_csv(DERIVED / "features_triage.csv")
    fourh = pd.read_csv(DERIVED / "features_fourh.csv")
    probs = pd.read_csv(DERIVED / "probs_avg.csv",
                         keep_default_na=False, na_values=[""])
    print(f"Loaded triage={triage.shape}, fourh={fourh.shape}, "
          f"probs={probs.shape}")

    # Outcomes (canonical label source — feature tables no longer
    # carry encounter_disposition_label).
    outcomes = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "encounter_disposition_label"]]

    # Align targets to feature rows
    triage_ix = triage["encounter_id"].tolist()
    fourh_ix = fourh["encounter_id"].tolist()
    argmax_full = probs.set_index("encounter_id")["argmax_class"]
    argmax_t = argmax_full.reindex(triage_ix).reset_index(drop=True)

    dispo_full = outcomes.set_index("encounter_id")[
        "encounter_disposition_label"].reindex(triage_ix)
    # Task-2 cohort: drug-positive
    cohort_ids = probs[probs["argmax_class"] != "None"]["encounter_id"].tolist()
    fourh_cohort = (fourh[fourh["encounter_id"].isin(cohort_ids)]
                       .merge(outcomes, on="encounter_id", how="left")
                       .reset_index(drop=True))
    y_dispo_label = fourh_cohort["encounter_disposition_label"]
    dispo_map = {"Discharge": 0, "Floor": 1, "ICU": 2}
    y_dispo = y_dispo_label.map(dispo_map).to_numpy()
    drug_map = {"Kraken Candy": 0, "Triton Tabs": 1, "Coral Dust": 2,
                 "None": 3}
    y_drug = argmax_t.map(drug_map).to_numpy()

    # X for tasks (drop the label col before passing to model code)
    X_t_for_drug = triage.copy()  # outcomes already absent
    X_f_for_dispo = fourh_cohort.drop(columns=["encounter_disposition_label"])

    section_a_missingness(rep, triage, fourh, argmax_t, dispo_full)
    section_b_integrity(rep, triage, fourh)
    section_c_univariate(rep, X_t_for_drug, X_f_for_dispo, y_drug, y_dispo)
    section_d_tree_importance(rep, X_t_for_drug, X_f_for_dispo, y_drug, y_dispo)
    section_e_targets(rep, argmax_t, dispo_full)
    section_f_notes(rep, triage, argmax_t)

    REPORT.write_text("\n".join(rep), encoding="utf-8")
    print(f"\nWrote: {REPORT}")
    print(f"Plots: {PLOTS}/")


if __name__ == "__main__":
    main()
