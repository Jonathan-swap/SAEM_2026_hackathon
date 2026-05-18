"""Precision-Recall curves for the three Task-1 binary classifiers.

Generates one PNG per binary task with two panels (5-fold CV OOF on the
left, temporal holdout on the right) and three curves per panel (one
per model: logreg, rforest, hgb). Each curve is labeled with its PR-AUC
(average precision) and prevalence is drawn as a dashed baseline.

Reads existing OOF + temporal prediction CSVs in derived/ (produced by
train_binary.py, train_kraken_binary.py, train_triton_coral.py) — no
retraining.

Outputs:
  derived/plots/pr_tier1_drug_vs_nodrug.png
  derived/plots/pr_kraken_vs_rest.png
  derived/plots/pr_triton_vs_coral.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
OUT = DERIVED / "plots"
OUT.mkdir(exist_ok=True)

MODELS = ["logreg", "rforest", "hgb"]
COLORS = {"logreg": "#1f77b4", "rforest": "#2ca02c", "hgb": "#d62728"}

# (display name, positive_label, prob_prefix, oof_csv, temporal_csv, panel_suptitle)
TASKS = [
    (
        "Tier-1 (drug vs no-drug)",
        "Drug-positive",
        "p_drug",
        "task1_binary_oof_predictions.csv",
        "task1_binary_temporal_predictions.csv",
        "pr_tier1_drug_vs_nodrug",
    ),
    (
        "Kraken vs rest (drug-positive cohort)",
        "Kraken Candy",
        "p_kraken",
        "task1_kraken_binary_oof_predictions.csv",
        "task1_kraken_binary_temporal_predictions.csv",
        "pr_kraken_vs_rest",
    ),
    (
        "Triton vs Coral (non-Kraken drug-positive)",
        "Triton Tabs",
        "p_triton",
        "task1_triton_coral_oof_predictions.csv",
        "task1_triton_coral_temporal_predictions.csv",
        "pr_triton_vs_coral",
    ),
]


def plot_panel(ax: plt.Axes, y_true: pd.Series, model_probs: dict[str, pd.Series],
               title: str) -> None:
    """Draw PR curves for each model on a single Axes."""
    prev = y_true.mean()
    ax.axhline(prev, color="grey", linestyle="--", linewidth=1,
               label=f"prevalence = {prev:.2f}")
    for model in MODELS:
        probs = model_probs[model]
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax.plot(rec, prec, color=COLORS[model], linewidth=1.5,
                label=f"{model}: PR-AUC = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)


def main() -> None:
    for display_name, pos_label, prob_prefix, oof_name, temp_name, stem in TASKS:
        oof = pd.read_csv(DERIVED / oof_name)
        temp = pd.read_csv(DERIVED / temp_name)

        # CV OOF: y_true + p_<prefix>_<model> columns
        y_cv = (oof["true_label"] == pos_label).astype(int)
        probs_cv = {m: oof[f"{prob_prefix}_{m}"] for m in MODELS}

        # Temporal: long-format with `model` column and `p_<prefix>` column
        y_temp_full = (temp["true_label"] == pos_label).astype(int)
        prob_col = prob_prefix
        probs_temp: dict[str, pd.Series] = {}
        y_temp: pd.Series | None = None
        for m in MODELS:
            block = temp[temp["model"] == m]
            probs_temp[m] = block[prob_col].reset_index(drop=True)
            y_this = (block["true_label"] == pos_label).astype(int).reset_index(drop=True)
            if y_temp is None:
                y_temp = y_this
            else:
                assert y_temp.equals(y_this), \
                    f"true_label differs across model rows in {temp_name}"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_panel(axes[0], y_cv, probs_cv,
                   f"5-fold CV (n={len(y_cv)})")
        plot_panel(axes[1], y_temp, probs_temp,
                   f"Temporal holdout (last day, n={len(y_temp)})")
        fig.suptitle(f"{display_name} — PR curves", fontsize=13, y=1.02)
        fig.tight_layout()

        out_path = OUT / f"{stem}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
