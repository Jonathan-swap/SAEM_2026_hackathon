"""Stand-alone Task-1 cascade inference (5-class with Siren Spark).

Loads the artifacts in this directory and produces a 5-class drug
prediction for one or more encounters in the same feature shape used
at training time.

Classes (0..4):
  0  None
  1  Kraken Candy
  2  Triton Tabs
  3  Coral Dust
  4  Siren Spark  (novel-class, Phase-2 only)

For Phase-1 encounters (where the 4-class cascade is the ground truth
target) p_siren_spark should stay near zero. For Phase-2 encounters
that look unlike anything in Phase-1, p_siren_spark rises and the
4-class mass is renormalised down by (1 - p_siren).

Usage (Python):
    from production.task1.predict import load_model, predict
    model = load_model()
    df = pd.read_csv("derived/phase2/features_triage.csv")
    preds = predict(model, df)
        # columns: encounter_id, drug_class (0-4),
        #          p_drug, p_kraken_given_drug,
        #          p_none, p_kraken, p_triton, p_coral, p_siren_spark,
        #          siren_U, siren_F   (diagnostics)

CLI:
    python production/task1/predict.py path/to/features_triage.csv \
        path/to/out_predictions.csv
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

NONE_IDX, KRAKEN_IDX, TRITON_IDX, CORAL_IDX, SIREN_IDX = 0, 1, 2, 3, 4
HERE = Path(__file__).resolve().parent


def _stable_uniform(encounter_id: str) -> float:
    """md5(encounter_id) -> uniform[0,1). Deterministic per encounter."""
    h = hashlib.md5(str(encounter_id).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def load_model(production_dir: Path = HERE) -> dict:
    meta = json.loads((production_dir / "metadata.json").read_text("utf-8"))
    return {
        "preprocessor": joblib.load(production_dir / "preprocessor.joblib"),
        "tier1":        joblib.load(production_dir / "tier1_model.joblib"),
        "kraken":       joblib.load(production_dir / "kraken_vs_rest_model.joblib"),
        "tau_drug":     float(meta["tau_drug"]),
        "tau_kraken":   float(meta["tau_kraken"]),
        "triton_prev":  float(meta["triton_prev"]),
        "feature_cols": list(meta["feature_columns"]),
        "siren_alpha":     float(meta.get("siren_alpha", 0.2)),
        "siren_sharpness": float(meta.get("siren_sharpness", 1.0)),
        "siren_cap":       float(meta.get("siren_cap", 0.50)),
        "siren_f_baseline":float(meta.get("siren_f_baseline", 0.798)),
        "siren_f_gate":    float(meta.get("siren_f_gate", 0.05)),
        "siren_clinical_features":
            list(meta.get("siren_clinical_features", [])),
        "siren_ref_mean":  dict(meta.get("siren_ref_mean", {})),
        "siren_ref_std":   dict(meta.get("siren_ref_std", {})),
        "metadata":     meta,
    }


def _feature_anomaly_F(X_full: pd.DataFrame, model: dict) -> np.ndarray:
    """Mean |z-score| of clinical vitals + labs vs the Phase-1
    reference distribution, baseline-subtracted and clipped to [0,1].

    F ≈ 0 for typical Phase-1 patients; F → 1 for encounters whose
    clinical signature is far from any Phase-1 archetype.
    """
    feats = model["siren_clinical_features"]
    if not feats:
        return np.zeros(len(X_full), dtype=float)
    zs = np.zeros((len(X_full), len(feats)), dtype=float)
    for j, c in enumerate(feats):
        if c not in X_full.columns:
            continue
        s = pd.to_numeric(X_full[c], errors="coerce")
        mu = model["siren_ref_mean"].get(c, float(s.mean()))
        sd = model["siren_ref_std"].get(c, float(s.std(ddof=1)) or 1.0)
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        zs[:, j] = np.where(
            s.notna().to_numpy(),
            (s.to_numpy() - mu) / sd,
            0.0,
        )
    mean_abs_z = np.abs(zs).mean(axis=1)
    F = mean_abs_z - model["siren_f_baseline"]
    return np.clip(F, 0.0, 1.0)


def predict(model: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with encounter_id + drug_class (0..4) + soft
    probabilities + diagnostic Siren signals."""
    if "encounter_id" not in X.columns:
        raise ValueError("X must contain an encounter_id column")
    encounter_ids = X["encounter_id"].astype(str).to_numpy()

    # Build the feature matrix for the cascade.
    drop = [c for c in ("encounter_id", "encounter_arrival_date",
                          "ground_truth_drug", "ground_truth_drug_name",
                          "encounter_disposition_label")
             if c in X.columns]
    Xf = X.drop(columns=drop)
    missing = [c for c in model["feature_cols"] if c not in Xf.columns]
    if missing:
        print(f"WARNING: {len(missing)} feature(s) missing from input "
              f"(filled with 0): {missing[:5]}"
              f"{' ...' if len(missing) > 5 else ''}")
        for c in missing:
            Xf[c] = 0
    Xf = Xf[model["feature_cols"]]
    Xp = np.asarray(model["preprocessor"].transform(Xf), dtype=float)

    p_drug = model["tier1"].predict_proba(Xp)[:, 1]
    p_kraken = model["kraken"].predict_proba(Xp)[:, 1]
    prev = model["triton_prev"]

    # Soft 4-class probabilities (chain product, sums to 1).
    p_none = 1.0 - p_drug
    p_K = p_drug * p_kraken
    p_T = p_drug * (1.0 - p_kraken) * prev
    p_C = p_drug * (1.0 - p_kraken) * (1.0 - prev)
    p4 = np.stack([p_none, p_K, p_T, p_C], axis=1)

    # Siren signals: U = cascade uncertainty, F = feature anomaly.
    U = 1.0 - p4.max(axis=1)
    F = _feature_anomaly_F(X, model)
    a = model["siren_alpha"]
    p_siren_raw = a * U + (1.0 - a) * F
    p_siren = np.clip(model["siren_sharpness"] * p_siren_raw,
                       0.0, model["siren_cap"])

    # Renormalise the 4-class mass.
    scale = (1.0 - p_siren)
    p_none, p_K, p_T, p_C = (p_none * scale, p_K * scale,
                              p_T * scale, p_C * scale)

    # Hard label = argmax over 5 classes, but keep the existing
    # threshold-based 4-class decision when p_siren is below cap/2 so
    # the 4-class cascade behaviour on Phase-1 patients is preserved.
    drug_class = np.full(len(p_drug), NONE_IDX, dtype=int)
    is_drug = p_drug >= model["tau_drug"]
    drug_class[is_drug & (p_kraken >= model["tau_kraken"])] = KRAKEN_IDX
    is_non_k = is_drug & (p_kraken < model["tau_kraken"])
    tc_u = np.array([_stable_uniform(e) for e in encounter_ids])
    drug_class[is_non_k & (tc_u < prev)] = TRITON_IDX
    drug_class[is_non_k & (tc_u >= prev)] = CORAL_IDX
    # Hard-label override to Siren Spark requires BOTH (a) p_siren
    # beats the renormalised cascade class probability AND (b) F is
    # above the siren_f_gate. Gate (b) prevents Phase-1 patients
    # (where F ≈ 0 by construction) from getting Siren labels just
    # because the 4-class cascade was uncertain about them.
    soft5 = np.stack([p_none, p_K, p_T, p_C, p_siren], axis=1)
    cascade_class_prob = soft5[np.arange(len(soft5)), drug_class]
    f_gate = model["siren_f_gate"]
    drug_class = np.where(
        (p_siren > cascade_class_prob) & (F > f_gate),
        SIREN_IDX, drug_class,
    )

    return pd.DataFrame({
        "encounter_id": encounter_ids,
        "drug_class": drug_class,
        "p_drug": p_drug,
        "p_kraken_given_drug": p_kraken,
        "p_none": p_none,
        "p_kraken": p_K,
        "p_triton": p_T,
        "p_coral": p_C,
        "p_siren_spark": p_siren,
        "siren_U": U,
        "siren_F": F,
    })


def main() -> None:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python predict.py <features.csv> "
                          "<out_predictions.csv>\n")
        raise SystemExit(2)
    X = pd.read_csv(sys.argv[1])
    model = load_model()
    out = predict(model, X)
    out.to_csv(sys.argv[2], index=False)
    print(f"Wrote {len(out)} predictions to {sys.argv[2]}")


if __name__ == "__main__":
    main()
