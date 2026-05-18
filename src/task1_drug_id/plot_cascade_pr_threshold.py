"""Precision + recall vs decision-threshold plots for Cascade-B (rforest).

Cascade-B = tier-1 binary (drug vs no-drug) × Kraken-vs-rest binary,
with Triton/Coral split by training prevalence. This is the
deployment-recommended Task-1 architecture (§7h of RUNBOOK).

For each of the 4 classes (None / Kraken / Triton / Coral), sweeps
the one-vs-rest decision threshold from 0 to 1 and plots:
  - precision(t) = TP / (TP + FP) at threshold t  (y-axis)
  - recall(t)    = TP / (TP + FN) at threshold t  (y-axis)
on the same axes vs threshold (x-axis). Prevalence drawn as a dashed
baseline (the precision a random classifier achieves at any threshold
where it predicts the class).

Two figures saved:
  derived/plots/cascade_B_pr_threshold_cv.png        (5-fold OOF, n=261)
  derived/plots/cascade_B_pr_threshold_temporal.png  (last-day holdout, n=74)

Reuses fit_cascade_B + preprocessor from compare_cascades.py so the
splits + seeds match the existing RUNBOOK numbers exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import compare_cascades as cc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
OUT = DERIVED / "plots"
OUT.mkdir(exist_ok=True)

CLASSES = cc.CLASSES4  # ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]
MODEL = "rforest"  # the deployment champion per §7h
THRESHOLDS = np.linspace(0.0, 1.0, 201)


def cv_cascade_b_probs(X_df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """5-fold OOF Cascade-B probabilities for rforest. (n, 4)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(y), 4))
    for fold, (tr, te) in enumerate(skf.split(X_df, y)):
        pre = cc.make_preprocessor(X_df)
        pre.fit(X_df.iloc[tr])
        X_tr = np.asarray(pre.transform(X_df.iloc[tr]), dtype=float)
        X_te = np.asarray(pre.transform(X_df.iloc[te]), dtype=float)
        oof[te] = cc.fit_cascade_B(MODEL, X_tr, y[tr], X_te)
    return oof


def temporal_cascade_b_probs(X_df: pd.DataFrame, y: np.ndarray,
                              arrival: pd.Series
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Temporal holdout Cascade-B probabilities for rforest.
    Returns (p_test, y_test) where the train/test split mirrors
    compare_cascades.run_temporal."""
    dates = pd.to_datetime(arrival)
    last_day = dates.dt.date.max()
    is_test = (dates.dt.date == last_day).to_numpy()
    is_train = ~is_test

    pre = cc.make_preprocessor(X_df)
    pre.fit(X_df.iloc[is_train])
    X_tr = np.asarray(pre.transform(X_df.iloc[is_train]), dtype=float)
    X_te = np.asarray(pre.transform(X_df.iloc[is_test]), dtype=float)
    p_te = cc.fit_cascade_B(MODEL, X_tr, y[is_train], X_te)
    return p_te, y[is_test]


def precision_recall_curves(p_class: np.ndarray, y_bin: np.ndarray,
                             thresholds: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """At each threshold t, compute precision and recall for the
    one-vs-rest decision `p_class >= t`."""
    prec = np.full_like(thresholds, np.nan, dtype=float)
    rec = np.full_like(thresholds, np.nan, dtype=float)
    P = int(y_bin.sum())
    for i, t in enumerate(thresholds):
        pred_pos = p_class >= t
        tp = int(np.sum(pred_pos & (y_bin == 1)))
        fp = int(np.sum(pred_pos & (y_bin == 0)))
        rec[i] = tp / P if P > 0 else np.nan
        prec[i] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    return prec, rec


def plot_panels(p_full: np.ndarray, y_true: np.ndarray, n_label: str,
                out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for k, (ax, cls) in enumerate(zip(axes.flat, CLASSES)):
        y_bin = (y_true == k).astype(int)
        prev = float(y_bin.mean())
        prec, rec = precision_recall_curves(p_full[:, k], y_bin, THRESHOLDS)
        ax.plot(THRESHOLDS, prec, color="#1f77b4", linewidth=1.8,
                label="precision")
        ax.plot(THRESHOLDS, rec, color="#d62728", linewidth=1.8,
                label="recall")
        ax.axhline(prev, color="grey", linestyle="--", linewidth=1,
                   label=f"prevalence = {prev:.2f}")
        ax.set_title(f"{cls}  (positives = {int(y_bin.sum())} / "
                     f"{len(y_bin)})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        if ax in axes[-1, :]:
            ax.set_xlabel("Threshold on Cascade-B P(class)")
        if ax in axes[:, 0]:
            ax.set_ylabel("Precision / Recall")
        ax.legend(loc="lower left", fontsize=9)
    fig.suptitle(
        f"Cascade-B (rforest × tier-1 × K-vs-rest × prevalence)  —  "
        f"precision + recall vs threshold  ({n_label})",
        fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    X_df, y, arrival, _ids = cc.load_features_and_y()

    print("=== 5-fold OOF Cascade-B probabilities (rforest) ===")
    p_cv = cv_cascade_b_probs(X_df, y)
    plot_panels(p_cv, y, n_label=f"5-fold OOF, n={len(y)}",
                out_path=OUT / "cascade_B_pr_threshold_cv.png")

    print("\n=== Temporal-holdout Cascade-B probabilities (rforest) ===")
    p_te, y_te = temporal_cascade_b_probs(X_df, y, arrival)
    plot_panels(p_te, y_te,
                n_label=f"temporal holdout (last day), n={len(y_te)}",
                out_path=OUT / "cascade_B_pr_threshold_temporal.png")


if __name__ == "__main__":
    main()
