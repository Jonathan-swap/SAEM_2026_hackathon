"""Threshold-tradeoff exploration focused on boosting cascade-level
Kraken specificity.

Re-uses the OOF probabilities already saved by `threshold_cascade_b.py`
(no model fitting here, just relabeling). Sweeps tau_drug and tau_kraken
on a fine grid, computes the assembled 4-class label, and reports the
per-class one-vs-rest sensitivity / specificity plus overall accuracy
and macro F1.

Three views are produced:

  1. Kraken-spec tradeoff curve — for tau_drug fixed at the current
     deployment value (0.57), step tau_kraken from 0.45 up, showing
     how cascade-level Kraken Sens and Spec move together.
  2. Joint sweep — every (tau_drug, tau_kraken) cell with per-class
     Sens/Spec. Filtered to "interesting" cells: cascade Kraken
     Spec >= 0.95.
  3. Recommendation — highest Kraken Sens subject to Kraken Spec
     >= TARGET_SPEC (default 0.97).

Outputs:
  derived/task1_kraken_spec_tradeoff.csv     full grid metrics
  derived/task1_kraken_spec_curve.csv        tau_drug fixed, sweep tau_kraken
  derived/task1_kraken_spec_report.md        markdown summary

Run:
  .venv/Scripts/python.exe src/task1_drug_id/explore_kraken_specificity.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

CLASS_NAMES = ["None", "Kraken", "Triton", "Coral"]
NONE_IDX, KRAKEN_IDX, TRITON_IDX, CORAL_IDX = 0, 1, 2, 3

GRID_STEP = 0.02
GRID_LO, GRID_HI = 0.05, 0.95

# Frozen deployment thresholds (current default).
CURRENT_TAU_DRUG = 0.57
CURRENT_TAU_KRAKEN = 0.45

# Recommendation target.
TARGET_KRAKEN_SPEC = 0.97


def hard_labels(p_drug, p_kraken, tc_u, triton_prev, td, tk):
    labels = np.full(len(p_drug), NONE_IDX, dtype=int)
    is_drug = p_drug >= td
    labels[is_drug & (p_kraken >= tk)] = KRAKEN_IDX
    is_non_k = is_drug & (p_kraken < tk)
    labels[is_non_k & (tc_u < triton_prev)] = TRITON_IDX
    labels[is_non_k & (tc_u >= triton_prev)] = CORAL_IDX
    return labels


def per_class_sens_spec(y_true, y_pred):
    out = {}
    for k, name in enumerate(CLASS_NAMES):
        y_b = (y_true == k).astype(int)
        p_b = (y_pred == k).astype(int)
        tp = int(((p_b == 1) & (y_b == 1)).sum())
        fp = int(((p_b == 1) & (y_b == 0)).sum())
        fn = int(((p_b == 0) & (y_b == 1)).sum())
        tn = int(((p_b == 0) & (y_b == 0)).sum())
        out[name] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sens": tp / (tp + fn) if (tp + fn) else float("nan"),
            "spec": tn / (tn + fp) if (tn + fp) else float("nan"),
        }
    return out


def main() -> None:
    print("=" * 78)
    print("Cascade-B Kraken-specificity trade-off (using OOF probabilities)")
    print("=" * 78)

    labels = pd.read_csv(DERIVED / "task1_cascade_b_threshold_labels.csv")
    labels = labels[labels["split"] == "oof"].reset_index(drop=True)
    truth = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "ground_truth_drug"]]
    df = labels.merge(truth, on="encounter_id", how="inner")

    y = df["ground_truth_drug"].astype(int).to_numpy()
    p_drug = df["p_drug"].to_numpy()
    p_kraken = df["p_kraken_given_drug"].to_numpy()
    tc_u = df["tc_uniform_draw"].to_numpy()
    triton_prev = float(df["triton_prev"].iloc[0])

    print(f"n={len(y)}  triton_prev={triton_prev:.4f}  "
          f"class counts={np.bincount(y, minlength=4).tolist()}")

    # ------ View 1: tau_drug = current, sweep tau_kraken ------
    print()
    print(f"--- Kraken-spec curve at tau_drug={CURRENT_TAU_DRUG:.2f} ---")
    rows = []
    for tk in np.arange(0.30, 0.99, 0.02):
        preds = hard_labels(p_drug, p_kraken, tc_u, triton_prev,
                              CURRENT_TAU_DRUG, tk)
        m = per_class_sens_spec(y, preds)
        acc = float((preds == y).mean())
        f1m = float(f1_score(y, preds, average="macro",
                               labels=[0, 1, 2, 3], zero_division=0))
        rows.append({
            "tau_drug": CURRENT_TAU_DRUG,
            "tau_kraken": round(float(tk), 2),
            "accuracy": acc,
            "macro_f1": f1m,
            "kraken_sens": m["Kraken"]["sens"],
            "kraken_spec": m["Kraken"]["spec"],
            "kraken_tp": m["Kraken"]["tp"],
            "kraken_fp": m["Kraken"]["fp"],
            "none_sens": m["None"]["sens"],
            "none_spec": m["None"]["spec"],
        })
    curve = pd.DataFrame(rows)
    curve.to_csv(DERIVED / "task1_kraken_spec_curve.csv", index=False)

    print(f"{'tau_K':>5} {'K_sens':>7} {'K_spec':>7} {'K_TP':>5} {'K_FP':>5} "
          f"{'acc':>6} {'macroF1':>8}")
    for _, r in curve.iterrows():
        print(f"{r['tau_kraken']:>5.2f} {r['kraken_sens']:>7.3f} "
              f"{r['kraken_spec']:>7.3f} {int(r['kraken_tp']):>5} "
              f"{int(r['kraken_fp']):>5} {r['accuracy']:>6.3f} "
              f"{r['macro_f1']:>8.3f}")

    # ------ View 2: joint (tau_drug, tau_kraken) grid ------
    print()
    print("--- Joint grid sweep (tau_drug, tau_kraken) ---")
    grid_rows = []
    tau_grid = np.arange(GRID_LO, GRID_HI + 1e-9, GRID_STEP)
    for td in tau_grid:
        for tk in tau_grid:
            preds = hard_labels(p_drug, p_kraken, tc_u, triton_prev,
                                  td, tk)
            m = per_class_sens_spec(y, preds)
            acc = float((preds == y).mean())
            f1m = float(f1_score(y, preds, average="macro",
                                   labels=[0, 1, 2, 3], zero_division=0))
            grid_rows.append({
                "tau_drug": round(float(td), 2),
                "tau_kraken": round(float(tk), 2),
                "accuracy": acc,
                "macro_f1": f1m,
                "kraken_sens": m["Kraken"]["sens"],
                "kraken_spec": m["Kraken"]["spec"],
                "none_sens": m["None"]["sens"],
                "none_spec": m["None"]["spec"],
                "triton_sens": m["Triton"]["sens"],
                "triton_spec": m["Triton"]["spec"],
                "coral_sens": m["Coral"]["sens"],
                "coral_spec": m["Coral"]["spec"],
            })
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(DERIVED / "task1_kraken_spec_tradeoff.csv", index=False)
    print(f"  {len(grid)} grid cells written to "
          f"derived/task1_kraken_spec_tradeoff.csv")

    # ------ View 3: recommendation ------
    # Maximise Kraken Sens subject to Kraken Spec >= TARGET_KRAKEN_SPEC,
    # then break ties by macro F1 (prevents Kraken-only optimisation
    # from torching the other classes).
    feasible = grid[grid["kraken_spec"] >= TARGET_KRAKEN_SPEC].copy()
    if feasible.empty:
        target = float(grid["kraken_spec"].max())
        print(f"\nNo cell meets Kraken Spec >= "
              f"{TARGET_KRAKEN_SPEC:.2f}; max achievable spec is "
              f"{target:.3f}. Relaxing target.")
        feasible = grid[grid["kraken_spec"] >= target - 1e-9].copy()
    feasible = feasible.sort_values(
        ["kraken_sens", "macro_f1"], ascending=[False, False],
    )
    rec = feasible.iloc[0]
    print()
    print(f"--- Recommendation (Kraken Spec >= "
          f"{TARGET_KRAKEN_SPEC:.2f}) ---")
    print(f"  tau_drug   = {rec['tau_drug']:.2f}")
    print(f"  tau_kraken = {rec['tau_kraken']:.2f}")
    print(f"  Kraken: Sens={rec['kraken_sens']:.3f}  "
          f"Spec={rec['kraken_spec']:.3f}")
    print(f"  None:   Sens={rec['none_sens']:.3f}  "
          f"Spec={rec['none_spec']:.3f}")
    print(f"  Triton: Sens={rec['triton_sens']:.3f}  "
          f"Spec={rec['triton_spec']:.3f}")
    print(f"  Coral:  Sens={rec['coral_sens']:.3f}  "
          f"Spec={rec['coral_spec']:.3f}")
    print(f"  accuracy={rec['accuracy']:.3f}  macroF1={rec['macro_f1']:.3f}")

    # ------ Comparison vs current default ------
    cur_mask = (
        (np.abs(grid["tau_drug"] - CURRENT_TAU_DRUG) < 1e-9)
        & (np.abs(grid["tau_kraken"] - CURRENT_TAU_KRAKEN) < 1e-9)
    )
    if cur_mask.any():
        cur = grid[cur_mask].iloc[0]
        print()
        print("--- Delta vs current default "
              f"(tau_drug={CURRENT_TAU_DRUG:.2f}, "
              f"tau_kraken={CURRENT_TAU_KRAKEN:.2f}) ---")
        for col in ("accuracy", "macro_f1",
                    "kraken_sens", "kraken_spec",
                    "none_sens", "none_spec",
                    "triton_sens", "triton_spec",
                    "coral_sens", "coral_spec"):
            d = float(rec[col] - cur[col])
            sign = "+" if d >= 0 else ""
            print(f"  {col:14s}  current={float(cur[col]):.3f}  "
                  f"recommended={float(rec[col]):.3f}  "
                  f"delta={sign}{d:.3f}")

    # ------ Markdown report ------
    md = [
        "# Cascade-B threshold tradeoff — boosting Kraken specificity",
        "",
        f"OOF probabilities (n={len(y)}, "
        f"triton_prev={triton_prev:.4f}). No model refitting — only "
        f"the cascade decision rule's threshold pair is changed.",
        "",
        "## Recommendation",
        "",
        f"To boost cascade-level Kraken specificity to "
        f">= **{TARGET_KRAKEN_SPEC:.2f}**, use:",
        "",
        f"- **tau_drug = {rec['tau_drug']:.2f}**",
        f"- **tau_kraken = {rec['tau_kraken']:.2f}**",
        "",
        "Effects on the cascade-level confusion (one-vs-rest):",
        "",
        "| Class | Sensitivity | Specificity |",
        "|---|---:|---:|",
        f"| None | {rec['none_sens']:.3f} | {rec['none_spec']:.3f} |",
        f"| Kraken | {rec['kraken_sens']:.3f} | "
        f"{rec['kraken_spec']:.3f} |",
        f"| Triton | {rec['triton_sens']:.3f} | "
        f"{rec['triton_spec']:.3f} |",
        f"| Coral | {rec['coral_sens']:.3f} | "
        f"{rec['coral_spec']:.3f} |",
        f"| **overall acc** | {rec['accuracy']:.3f} | — |",
        f"| **macro F1** | {rec['macro_f1']:.3f} | — |",
        "",
        "## Kraken-specificity curve (tau_drug fixed at "
        f"{CURRENT_TAU_DRUG:.2f})",
        "",
        "| tau_kraken | Kraken Sens | Kraken Spec | TP | FP | "
        "Accuracy | Macro F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in curve.iterrows():
        md.append(
            f"| {r['tau_kraken']:.2f} | {r['kraken_sens']:.3f} | "
            f"{r['kraken_spec']:.3f} | {int(r['kraken_tp'])} | "
            f"{int(r['kraken_fp'])} | {r['accuracy']:.3f} | "
            f"{r['macro_f1']:.3f} |"
        )
    md += [
        "",
        "## Files",
        "",
        "- `derived/task1_kraken_spec_curve.csv` — sweep at fixed tau_drug",
        "- `derived/task1_kraken_spec_tradeoff.csv` — full joint grid",
        "",
    ]
    (DERIVED / "task1_kraken_spec_report.md").write_text(
        "\n".join(md), encoding="utf-8",
    )
    print(f"\nWrote: {DERIVED / 'task1_kraken_spec_report.md'}")


if __name__ == "__main__":
    main()
