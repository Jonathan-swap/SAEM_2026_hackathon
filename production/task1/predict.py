"""Stand-alone Task-1 cascade inference.

Loads the artifacts in this directory and produces a 4-class drug
prediction (0=None / 1=Kraken / 2=Triton / 3=Coral) for one or more
encounters in the same feature shape used at training time.

Usage (Python):
    from production.predict import load_model, predict
    model = load_model()
    df = pd.read_csv("derived/features_triage.csv")   # same schema
    preds = predict(model, df)   # DataFrame with encounter_id + drug_class

CLI:
    python production/predict.py path/to/features_triage.csv \
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

NONE_IDX, KRAKEN_IDX, TRITON_IDX, CORAL_IDX = 0, 1, 2, 3
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
        "metadata":     meta,
    }


def predict(model: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with encounter_id + drug_class + all 4 probs."""
    if "encounter_id" not in X.columns:
        raise ValueError("X must contain an encounter_id column")
    encounter_ids = X["encounter_id"].astype(str).to_numpy()
    drop = [c for c in ("encounter_id", "encounter_arrival_date",
                          "ground_truth_drug", "ground_truth_drug_name",
                          "encounter_disposition_label")
             if c in X.columns]
    Xf = X.drop(columns=drop)
    # Re-order columns to match what the preprocessor saw at fit time.
    missing = [c for c in model["feature_cols"] if c not in Xf.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:5]}"
                         f"{' ... ' if len(missing)>5 else ''}")
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

    # Hard cascade label using the frozen thresholds.
    drug_class = np.full(len(p_drug), NONE_IDX, dtype=int)
    is_drug = p_drug >= model["tau_drug"]
    drug_class[is_drug & (p_kraken >= model["tau_kraken"])] = KRAKEN_IDX
    is_non_k = is_drug & (p_kraken < model["tau_kraken"])
    tc_u = np.array([_stable_uniform(e) for e in encounter_ids])
    drug_class[is_non_k & (tc_u < prev)] = TRITON_IDX
    drug_class[is_non_k & (tc_u >= prev)] = CORAL_IDX

    return pd.DataFrame({
        "encounter_id": encounter_ids,
        "drug_class": drug_class,
        "p_drug": p_drug,
        "p_kraken_given_drug": p_kraken,
        "p_none": p_none,
        "p_kraken": p_K,
        "p_triton": p_T,
        "p_coral": p_C,
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
