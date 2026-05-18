"""Shared Siren Spark (5th-class) scoring utility.

Siren Spark is a brand-new synthetic drug introduced in the Phase-2
dataset. There is **no ground truth** for which patients are on Siren
Spark, and **no published clinical profile** in the challenge release —
so the 5th class has to be inferred from the data itself.

Each of the 10 narrative-scoring agents (agents 1–10) produces a 4-class
probability vector P(None, Kraken, Triton, Coral) using its own lens.
After that 4-class scoring, this utility extends the row to 5 classes by
allocating probability mass to Siren Spark based on:

  1. **Confidence signal** (always available, per-row):
        U = 1 - max(p_none, p_kraken, p_triton, p_coral)
     If no known class scores high, the encounter is hard to fit to any
     known pattern — possible novel drug.

  2. **Feature-anomaly signal** (optional, per-row):
        F = clipped z-score outlierness vs Phase-1 feature distribution
     Encounters whose features sit far from any Phase-1 archetype are
     more likely to be on the new drug.

The two components combine into a per-encounter siren score:
        s_raw = alpha * U + (1 - alpha) * F        (alpha ∈ [0, 1])
The final p_siren is `min(s_raw * sharpness, cap)` where `cap` keeps
Siren Spark from dominating in cases where the 4-class scoring is
already weakly informative. The remaining 4 classes are scaled by
(1 - p_siren) and the full 5-vector is re-normalised.

`alpha` and `sharpness` are per-agent tuning knobs that give the
ensemble its diversity — different agents weight the two signals
differently.

Public API:
    add_siren_spark(probs_4class_df, features_df=None, *,
                     alpha=0.5, sharpness=1.0, cap=0.7,
                     phase1_features_path=None)
        Returns a (n, 5) DataFrame with columns
        [encounter_id, p_none, p_kraken, p_triton, p_coral, p_siren_spark]
        all summing to 1.0 per row.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DERIVED = ROOT / "derived"

CLASS_COLS_4 = ["p_none", "p_kraken", "p_triton", "p_coral"]
ALL_CLASS_COLS_5 = CLASS_COLS_4 + ["p_siren_spark"]

# Numeric features used for the anomaly signal. We pick a small set of
# vital/lab/composite columns that exist in both Phase-1 and Phase-2
# feature tables and that have meaningful variance.
ANOMALY_FEATURES = [
    "triage_heart_rate",
    "triage_respiratory_rate",
    "triage_snapshot.systolic_bp",
    "triage_snapshot.diastolic_bp",
    "triage_snapshot.oxygen_saturation",
    "triage_temperature_c",
    "triage_gcs",
    "triage_pain_scale",
    "triage_lab_glucose",
    "triage_lab_ph",
    "triage_lab_sodium",
    "triage_lab_potassium",
    "triage_lab_hemoglobin",
    "triage_lab_anion_gap",
]


def _confidence_signal(p4: np.ndarray) -> np.ndarray:
    """U = 1 - max(p4) per row. Higher = more uncertain → more likely novel."""
    return 1.0 - p4.max(axis=1)


def _phase1_stats(phase1_features_path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load Phase-1 means + stds for the anomaly features. Falls back to
    None if Phase-1 reference data isn't available."""
    if phase1_features_path is None:
        phase1_features_path = DERIVED / "features_triage.csv"
    if not phase1_features_path.exists():
        return None
    df1 = pd.read_csv(phase1_features_path)
    keep = [c for c in ANOMALY_FEATURES if c in df1.columns]
    if not keep:
        return None
    stats = pd.DataFrame({
        "mean": df1[keep].mean(),
        "std": df1[keep].std(ddof=0).replace(0, np.nan),
    })
    return stats


def _feature_anomaly_signal(features_df: pd.DataFrame,
                              stats: pd.DataFrame) -> np.ndarray:
    """Per-row outlierness vs Phase-1 reference stats. We compute the
    mean |z| across the available anomaly features (skipping NaNs) and
    squash it to [0, 1] with a soft cap at |z|=3."""
    keep = [c for c in stats.index if c in features_df.columns]
    if not keep:
        return np.zeros(len(features_df))
    sub = features_df[keep].to_numpy(dtype=float)
    mean = stats.loc[keep, "mean"].to_numpy()
    std = stats.loc[keep, "std"].to_numpy()
    z = (sub - mean) / std  # shape (n, k)
    abs_z = np.abs(z)
    abs_z = np.where(np.isnan(abs_z), 0.0, abs_z)
    mean_abs_z = abs_z.mean(axis=1)
    return np.clip(mean_abs_z / 3.0, 0.0, 1.0)


def add_siren_spark(
    probs_4class: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    *,
    alpha: float = 0.5,
    sharpness: float = 1.0,
    cap: float = 0.7,
    phase1_features_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Extend a 4-class probability frame to 5 classes by adding
    `p_siren_spark`. Returns a fresh DataFrame with columns
    [encounter_id, p_none, p_kraken, p_triton, p_coral, p_siren_spark].

    probs_4class: must contain columns [encounter_id, p_none, p_kraken,
        p_triton, p_coral]. Rows are assumed to sum to 1 (re-normalised
        defensively if not).

    features_df: optional, must contain encounter_id + at least one of
        ANOMALY_FEATURES. Used to compute the feature-anomaly signal F.
        If None or no matching columns, only the confidence signal U is
        used.

    alpha: weight on the confidence signal U (and 1-alpha on F).
        Per-agent variation knob.

    sharpness: multiplier on the combined raw score before clipping.
        Higher → bigger Siren Spark mass for the same uncertainty.

    cap: maximum allowed p_siren per row. Prevents the 5th column from
        dominating when the 4-class scoring is itself weak."""
    needed = {"encounter_id", *CLASS_COLS_4}
    missing = needed - set(probs_4class.columns)
    if missing:
        raise ValueError(f"probs_4class missing columns: {sorted(missing)}")

    out = probs_4class[["encounter_id", *CLASS_COLS_4]].copy()
    p4 = out[CLASS_COLS_4].to_numpy(dtype=float)
    # Defensive re-normalise (no-op for well-formed input)
    row_sum = p4.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    p4 = p4 / row_sum

    U = _confidence_signal(p4)

    F = np.zeros(len(out))
    if features_df is not None:
        stats = _phase1_stats(phase1_features_path)
        if stats is not None and "encounter_id" in features_df.columns:
            ordered = (out[["encounter_id"]]
                       .merge(features_df, on="encounter_id", how="left"))
            F = _feature_anomaly_signal(ordered, stats)

    s_raw = alpha * U + (1.0 - alpha) * F
    p_siren = np.clip(s_raw * sharpness, 0.0, cap)

    # Distribute remaining 1 - p_siren across the 4 known classes,
    # preserving relative shares.
    out_p4 = p4 * (1.0 - p_siren)[:, None]
    out[CLASS_COLS_4] = out_p4
    out["p_siren_spark"] = p_siren

    # Final defensive renormalisation to handle floating-point drift.
    five_sum = out[ALL_CLASS_COLS_5].sum(axis=1).to_numpy().copy()
    five_sum[five_sum == 0] = 1.0
    out[ALL_CLASS_COLS_5] = out[ALL_CLASS_COLS_5].to_numpy() / five_sum[:, None]

    return out
