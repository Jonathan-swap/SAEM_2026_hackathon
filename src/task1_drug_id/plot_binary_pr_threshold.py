"""Precision + recall vs decision-threshold plots for each binary
model in the best Task-1 cascade stack.

Best cascade stack = rforest × Cascade-B (§7h of RUNBOOK), composed of:
  1. tier-1 binary: drug vs no-drug                (all 261)
  2. Kraken-vs-rest binary: K vs (T+C)             (157 drug-positive)

Cascade-B does NOT use a Triton-vs-Coral classifier — it splits the
non-Kraken drug-positive probability mass by training prevalence
(~52% Triton / 48% Coral), so there are only TWO fitted binary models
in the stack.

For each binary, plots precision (blue) and recall (red) curves vs the
decision threshold, with two panels per figure:
  - left:  5-fold CV out-of-fold probabilities
  - right: temporal holdout (last day)

Reads the consolidated probability CSVs in derived/ (produced by
src/task1_drug_id/export_binary_probabilities.py).

Outputs:
  derived/plots/binary_pr_threshold_tier1.png
  derived/plots/binary_pr_threshold_kraken_vs_rest.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
OUT = DERIVED / "plots"
OUT.mkdir(exist_ok=True)

MODEL = "rforest"  # the deployment-champion model per §7h
THRESHOLDS = np.linspace(0.0, 1.0, 201)

# (probability-csv, positive_label, display name, output stem)
BINARY_TASKS = [
    (
        "task1_tier1_probabilities.csv",
        "Drug-positive",
        "Tier-1: drug vs no-drug",
        "binary_pr_threshold_tier1",
    ),
    (
        "task1_kraken_vs_rest_probabilities.csv",
        "Kraken Candy",
        "Kraken vs rest (drug-positive cohort)",
        "binary_pr_threshold_kraken_vs_rest",
    ),
]


def precision_recall_at_thresholds(probs: np.ndarray, y_bin: np.ndarray
                                    ) -> tuple[np.ndarray, np.ndarray]:
    prec = np.full_like(THRESHOLDS, np.nan, dtype=float)
    rec = np.full_like(THRESHOLDS, np.nan, dtype=float)
    P = int(y_bin.sum())
    for i, t in enumerate(THRESHOLDS):
        pred_pos = probs >= t
        tp = int(np.sum(pred_pos & (y_bin == 1)))
        fp = int(np.sum(pred_pos & (y_bin == 0)))
        rec[i] = tp / P if P > 0 else np.nan
        prec[i] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    return prec, rec


def plot_panel(ax: plt.Axes, probs: np.ndarray, y_bin: np.ndarray,
               title: str) -> None:
    prec, rec = precision_recall_at_thresholds(probs, y_bin)
    prev = float(y_bin.mean())
    ax.plot(THRESHOLDS, prec, color="#1f77b4", linewidth=1.8, label="precision")
    ax.plot(THRESHOLDS, rec, color="#d62728", linewidth=1.8, label="recall")
    ax.axhline(prev, color="grey", linestyle="--", linewidth=1,
               label=f"prevalence = {prev:.2f}")
    # Mark the precision = recall crossover (operating point of interest).
    diff = prec - rec
    sign_change = np.where(np.diff(np.signbit(diff)))[0]
    if len(sign_change):
        idx = sign_change[0]
        t_cross = THRESHOLDS[idx]
        val_cross = (prec[idx] + rec[idx]) / 2
        ax.axvline(t_cross, color="black", linestyle=":", linewidth=1,
                   alpha=0.6)
        ax.annotate(f"P=R @ t={t_cross:.2f}\n({val_cross:.2f})",
                    xy=(t_cross, val_cross),
                    xytext=(t_cross + 0.04, val_cross - 0.10),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="black", alpha=0.5))
    ax.set_title(f"{title}  (positives = {int(y_bin.sum())} / {len(y_bin)})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Decision threshold on rforest probability")
    ax.set_ylabel("Precision / Recall")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)


def main() -> None:
    for csv_name, pos_label, display_name, stem in BINARY_TASKS:
        df = pd.read_csv(DERIVED / csv_name)
        y_full = (df["true_label"] == pos_label).astype(int).to_numpy()

        # CV-OOF panel uses all encounters in the cohort
        probs_cv = df[f"prob_{MODEL}_cv_oof"].to_numpy()

        # Temporal panel uses only last-day encounters
        last = df["is_last_day"] == 1
        probs_temp = df.loc[last, f"prob_{MODEL}_temporal"].to_numpy()
        y_temp = (df.loc[last, "true_label"] == pos_label).astype(int).to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        plot_panel(axes[0], probs_cv, y_full,
                   f"5-fold CV OOF (n={len(y_full)})")
        plot_panel(axes[1], probs_temp, y_temp,
                   f"Temporal holdout — last day (n={len(y_temp)})")
        fig.suptitle(
            f"{display_name}  —  rforest precision + recall vs threshold",
            fontsize=12, y=1.02)
        fig.tight_layout()
        out_path = OUT / f"{stem}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
