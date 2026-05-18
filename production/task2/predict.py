"""Stand-alone Task-2 disposition inference.

Loads the artifacts in this directory and produces a 3-class
disposition prediction (0=Discharge, 1=Floor, 2=ICU) per encounter.

Required inputs for new data:
  features_fourh.csv  — same schema used at training time
  probs_avg.csv       — Task-1 LLM-agent consensus probabilities
                         (columns p_kraken, p_triton, p_coral, p_none).
                         If the production cohort was drug-positive,
                         these are also used to filter the cohort
                         (argmax != "p_none" via ground_truth_drug -
                         in real deployment, replace this with the
                         predicted drug class from Task-1 cascade).

Usage (Python):
    from production.task2.predict import load_model, predict
    model = load_model()
    X = pd.read_csv("derived/features_fourh.csv")
    probs = pd.read_csv("derived/probs_avg.csv")
    preds = predict(model, X, probs)   # encounter_id + disposition_class

CLI:
    python production/task2/predict.py path/to/features_fourh.csv \
        path/to/probs_avg.csv path/to/out_predictions.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROB_COLS = ["p_kraken", "p_triton", "p_coral", "p_none"]
CLASS_NAMES = ["Discharge", "Floor", "ICU"]


def load_model(production_dir: Path = HERE) -> dict:
    meta = json.loads((production_dir / "metadata.json").read_text("utf-8"))
    return {
        "preprocessor": joblib.load(production_dir / "preprocessor.joblib"),
        "model":        joblib.load(production_dir / "model.joblib"),
        "feature_cols": list(meta["feature_columns"]),
        "metadata":     meta,
    }


def predict(model: dict, X: pd.DataFrame,
             probs: pd.DataFrame) -> pd.DataFrame:
    """Predict disposition class per encounter.

    X      — features_fourh.csv frame (must contain encounter_id)
    probs  — probs_avg.csv frame (must contain encounter_id + the
              four p_* columns).
    """
    if "encounter_id" not in X.columns:
        raise ValueError("X must contain an encounter_id column")
    if "encounter_id" not in probs.columns:
        raise ValueError("probs must contain an encounter_id column")
    missing_probs = [c for c in PROB_COLS if c not in probs.columns]
    if missing_probs:
        raise ValueError(f"probs missing columns: {missing_probs}")

    # Drop stale outcome columns if present
    for c in ("encounter_disposition_label", "ground_truth_drug",
               "ground_truth_drug_name"):
        if c in X.columns:
            X = X.drop(columns=[c])

    df = X.merge(probs[["encounter_id", *PROB_COLS]],
                  on="encounter_id", how="inner")
    if len(df) == 0:
        raise ValueError("No rows after merging features with probs")

    encounter_ids = df["encounter_id"].astype(str).to_numpy()
    drop = [c for c in ("encounter_id", "encounter_arrival_date")
             if c in df.columns]
    Xf = df.drop(columns=drop)

    missing = [c for c in model["feature_cols"] if c not in Xf.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:5]}"
                         f"{' ... ' if len(missing)>5 else ''}")
    Xf = Xf[model["feature_cols"]]
    Xp = np.asarray(model["preprocessor"].transform(Xf), dtype=float)

    proba = model["model"].predict_proba(Xp)
    col_order = list(model["model"].classes_)
    if col_order != list(range(len(CLASS_NAMES))):
        order = [col_order.index(i) for i in range(len(CLASS_NAMES))]
        proba = proba[:, order]

    pred = proba.argmax(axis=1)
    return pd.DataFrame({
        "encounter_id": encounter_ids,
        "disposition_class": pred,
        "disposition_label": [CLASS_NAMES[i] for i in pred],
        "p_discharge": proba[:, 0],
        "p_floor": proba[:, 1],
        "p_icu": proba[:, 2],
    })


def main() -> None:
    if len(sys.argv) != 4:
        sys.stderr.write(
            "usage: python predict.py <features_fourh.csv> "
            "<probs_avg.csv> <out_predictions.csv>\n"
        )
        raise SystemExit(2)
    X = pd.read_csv(sys.argv[1])
    probs = pd.read_csv(sys.argv[2], keep_default_na=False, na_values=[""])
    model = load_model()
    out = predict(model, X, probs)
    out.to_csv(sys.argv[3], index=False)
    print(f"Wrote {len(out)} predictions to {sys.argv[3]}")


if __name__ == "__main__":
    main()
