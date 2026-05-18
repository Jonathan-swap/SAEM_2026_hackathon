"""Train the deployment Task-1 model on 100% of the dataset and save
the artifacts to `production/`.

This is the final retrain step: the picked thresholds and prevalence
were chosen on 5-fold OOF (see threshold_cascade_b.py), but for
deployment we want the underlying tier-1 and K-vs-rest classifiers
trained on every available encounter — no held-out fold left behind.

Cascade-B (RUNBOOK §7h deployment champion):
  Stage 1 (tier-1):    rforest, P(drug-positive | X)
  Stage 2 (K-vs-rest): rforest, P(Kraken | drug-positive, X)
  Stage 3 (T-vs-C):    deterministic per-encounter Bernoulli against
                       the training-set prevalence (no model)

Picked decision thresholds (frozen from the macro-F1 optimum on
5-fold OOF — these are NOT re-picked on the 100% fit):
  tau_drug   = 0.57
  tau_kraken = 0.45

Artifacts written to production/task1/ (sibling of production/task2/):
  preprocessor.joblib            ColumnTransformer fitted on all 261
  tier1_model.joblib             rforest, P(drug+) on all 261
  kraken_vs_rest_model.joblib    rforest, P(K | drug+) on all 157
  metadata.json                  thresholds, prevalence, feature schema,
                                  class mapping, sklearn/numpy versions,
                                  training counts, fitted date
  predict.py                     minimal stand-alone inference script
  README.md                      how to load + use the artifacts

Run:
    .venv/Scripts/python.exe src/task1_drug_id/train_production.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
PRODUCTION = ROOT / "production" / "task1"

CLASS_NAMES = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]
NONE_IDX, KRAKEN_IDX, TRITON_IDX, CORAL_IDX = 0, 1, 2, 3

TEXT_COL = "triage_brief_note"

# Frozen thresholds. tau_drug came from the macro-F1 5-fold-OOF
# optimum; tau_kraken was bumped from 0.45 to 0.50 to improve cascade
# Kraken specificity from 0.931 to 0.961 (+3.0 pp) at the cost of
# Kraken sensitivity 0.190 -> 0.172 (-1.8 pp) and macro F1 0.430 ->
# 0.423 (-0.7 pp). See derived/task1_kraken_spec_report.md and
# src/task1_drug_id/explore_kraken_specificity.py.
# DO NOT re-pick these on the 100% fit — that would optimistically bias
# the deployment thresholds. Re-pick only if the underlying feature set
# or training distribution materially changes.
TAU_DRUG = 0.57
TAU_KRAKEN = 0.50


def load_features_and_y() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X = pd.read_csv(DERIVED / "features_triage.csv")
    outcomes = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "ground_truth_drug"]]
    for c in ("encounter_disposition_label", "ground_truth_drug",
              "ground_truth_drug_name"):
        if c in X.columns:
            X = X.drop(columns=[c])
    df = X.merge(outcomes, on="encounter_id", how="inner")
    drop = ["encounter_id", "encounter_arrival_date", "ground_truth_drug"]
    drop = [c for c in drop if c in df.columns]
    y = df["ground_truth_drug"].astype(int).to_numpy()
    ids = df["encounter_id"].to_numpy()
    return df.drop(columns=drop), y, ids


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    text_col = TEXT_COL if TEXT_COL in X.columns else None
    obj_cols = [c for c in X.select_dtypes(include=["object", "string"]).columns
                if c != text_col]
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    num_cols = list(set(num_cols + bool_cols))
    transformers = []
    if text_col:
        transformers.append(("text", TfidfVectorizer(
            max_features=50, stop_words="english", ngram_range=(1, 2),
            min_df=3), text_col))
    if obj_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant",
                                     fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False)),
        ]), obj_cols))
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols))
    return ColumnTransformer(transformers, remainder="drop")


def make_rforest() -> RandomForestClassifier:
    """Same hyperparameters as the OOF threshold-picking run."""
    return RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


def write_predict_script(production_dir: Path) -> None:
    """Tiny stand-alone inference script. No external imports beyond
    joblib + pandas + numpy + sklearn (matched by metadata.json)."""
    code = '''"""Stand-alone Task-1 cascade inference.

Loads the artifacts in this directory and produces a 4-class drug
prediction (0=None / 1=Kraken / 2=Triton / 3=Coral) for one or more
encounters in the same feature shape used at training time.

Usage (Python):
    from production.predict import load_model, predict
    model = load_model()
    df = pd.read_csv("derived/features_triage.csv")   # same schema
    preds = predict(model, df)   # DataFrame with encounter_id + drug_class

CLI:
    python production/predict.py path/to/features_triage.csv \\
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
                          "<out_predictions.csv>\\n")
        raise SystemExit(2)
    X = pd.read_csv(sys.argv[1])
    model = load_model()
    out = predict(model, X)
    out.to_csv(sys.argv[2], index=False)
    print(f"Wrote {len(out)} predictions to {sys.argv[2]}")


if __name__ == "__main__":
    main()
'''
    (production_dir / "predict.py").write_text(code, encoding="utf-8")


def write_readme(production_dir: Path, meta: dict) -> None:
    md = f"""# Task-1 production model (Cascade-B, rforest)

Trained on **100% of the dataset** (n={meta['n_train_total']} encounters,
class counts {meta['class_counts']}). Picked decision thresholds frozen
from the macro-F1 optimum on 5-fold OOF — they are NOT re-picked at the
100% retrain, because doing so would optimistically bias the deployment.

## Contents

| File | Purpose |
|---|---|
| `preprocessor.joblib` | sklearn ColumnTransformer (TF-IDF on `triage_brief_note`, OHE on categoricals, median-impute + StandardScaler on numerics). Fitted on all {meta['n_train_total']} encounters. |
| `tier1_model.joblib` | rforest, predicts `P(drug-positive)`. Trained on all {meta['n_train_total']} encounters with binary label `ground_truth_drug != 0`. |
| `kraken_vs_rest_model.joblib` | rforest, predicts `P(Kraken \\| drug-positive)`. Trained on the {meta['n_train_drug_pos']} drug-positive encounters with binary label `ground_truth_drug == 1`. |
| `metadata.json` | Frozen thresholds, prevalence, feature schema, version pins. |
| `predict.py` | Stand-alone inference: load the model and score new encounters. |

## Frozen decision rule

```text
if  P(drug)     <  tau_drug       -> None       (drug_class = 0)
elif P(K|drug)  >= tau_kraken     -> Kraken     (drug_class = 1)
elif md5(eid)/2^64 <  triton_prev -> Triton     (drug_class = 2)
else                              -> Coral      (drug_class = 3)

tau_drug    = {meta['tau_drug']}
tau_kraken  = {meta['tau_kraken']}
triton_prev = {meta['triton_prev']:.4f}
```

Stage 3 is a deterministic per-encounter Bernoulli (md5 of
`encounter_id` -> uniform [0,1) -> compare to `triton_prev`) so the
marginal Triton/Coral output distribution matches the training
prevalence. The same encounter always gets the same T/C label.

## Quick inference

```python
import pandas as pd
from production.predict import load_model, predict

model = load_model()
X = pd.read_csv("derived/features_triage.csv")   # same schema as training
out = predict(model, X)                          # encounter_id + drug_class + 4 probs
out.to_csv("derived/predictions.csv", index=False)
```

Or from the shell:
```
python production/predict.py path/to/features_triage.csv path/to/out.csv
```

## Versioning

- Python: `{meta['python']}`
- scikit-learn: `{meta['sklearn_version']}`
- numpy: `{meta['numpy_version']}`
- Fitted: `{meta['fitted_utc']}`
- Feature columns hash: `{meta['feature_columns_md5']}` (sha-style fingerprint
  of the `feature_columns` list; use it to detect schema drift in new data)
"""
    (production_dir / "README.md").write_text(md, encoding="utf-8")


def main() -> None:
    print("=" * 78)
    print("Train Task-1 deployment model (Cascade-B, 100% of dataset)")
    print("=" * 78)

    X_df, y, ids = load_features_and_y()
    print(f"X: {X_df.shape}   y class counts: "
          f"{np.bincount(y, minlength=4).tolist()}")

    PRODUCTION.mkdir(exist_ok=True)

    # Preprocessor: fit on the full dataset
    pre = make_preprocessor(X_df)
    pre.fit(X_df)
    Xp = np.asarray(pre.transform(X_df), dtype=float)
    print(f"Preprocessor fitted; transformed shape: {Xp.shape}")

    # Stage 1: tier-1 (drug vs no-drug) on all 261
    y_tier1 = (y != NONE_IDX).astype(int)
    tier1 = make_rforest()
    tier1.fit(Xp, y_tier1)
    p_drug_train = tier1.predict_proba(Xp)[:, 1]
    print(f"Tier-1 fitted on n={len(y_tier1)} "
          f"({y_tier1.sum()} positives), "
          f"in-sample p_drug mean={p_drug_train.mean():.3f}")

    # Stage 2: Kraken-vs-rest on all drug-positive (157)
    drug_mask = y != NONE_IDX
    y_kraken = (y[drug_mask] == KRAKEN_IDX).astype(int)
    kraken = make_rforest()
    kraken.fit(Xp[drug_mask], y_kraken)
    p_k_train = kraken.predict_proba(Xp[drug_mask])[:, 1]
    print(f"Kraken-vs-rest fitted on n={int(drug_mask.sum())} "
          f"({y_kraken.sum()} Kraken), "
          f"in-sample p_kraken mean={p_k_train.mean():.3f}")

    # Stage 3 scalar: training-set Triton prevalence among non-K drug+
    non_k_mask = (y == TRITON_IDX) | (y == CORAL_IDX)
    triton_prev = float((y[non_k_mask] == TRITON_IDX).mean())
    print(f"Triton prevalence (non-K drug+ in training): "
          f"{triton_prev:.4f}  ({int((y[non_k_mask] == TRITON_IDX).sum())} "
          f"of {int(non_k_mask.sum())})")

    # Save artifacts
    joblib.dump(pre, PRODUCTION / "preprocessor.joblib")
    joblib.dump(tier1, PRODUCTION / "tier1_model.joblib")
    joblib.dump(kraken, PRODUCTION / "kraken_vs_rest_model.joblib")

    feature_cols = list(X_df.columns)
    feat_md5 = hashlib.md5(
        "\n".join(feature_cols).encode("utf-8")
    ).hexdigest()
    meta = {
        "model": "cascade_b_rforest_v1",
        "task": "task1_drug_id",
        "class_mapping": {str(i): c for i, c in enumerate(CLASS_NAMES)},
        "tau_drug": TAU_DRUG,
        "tau_kraken": TAU_KRAKEN,
        "triton_prev": triton_prev,
        "feature_columns": feature_cols,
        "feature_columns_md5": feat_md5,
        "n_train_total": int(len(y)),
        "n_train_drug_pos": int(drug_mask.sum()),
        "n_train_non_kraken_drug_pos": int(non_k_mask.sum()),
        "class_counts": np.bincount(y, minlength=4).tolist(),
        "rforest_hyperparams": {
            "n_estimators": 400,
            "max_depth": 8,
            "min_samples_leaf": 4,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "threshold_picking": (
            "tau_drug and tau_kraken were picked on 5-fold OOF "
            "(threshold_cascade_b.py) under the macro-F1 criterion and "
            "are FROZEN here — they are not re-picked at the 100% retrain."
        ),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "fitted_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }
    (PRODUCTION / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )

    write_predict_script(PRODUCTION)
    write_readme(PRODUCTION, meta)

    print()
    print(f"Wrote artifacts to {PRODUCTION}:")
    for p in sorted(PRODUCTION.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:38s} {size_kb:8.1f} KB")
    print()
    print("Smoke test (load + score the training set):")
    sys.path.insert(0, str(PRODUCTION))
    import importlib
    predict_mod = importlib.import_module("predict")
    model = predict_mod.load_model(PRODUCTION)
    # Reload the raw features+ids for a true round-trip test.
    raw = pd.read_csv(DERIVED / "features_triage.csv")
    preds = predict_mod.predict(model, raw)
    truth = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "ground_truth_drug"]]
    merged = preds.merge(truth, on="encounter_id", how="inner")
    acc = (merged["drug_class"] == merged["ground_truth_drug"]).mean()
    print(f"  loaded {len(preds)} predictions; "
          f"in-sample accuracy = {acc:.4f}  "
          f"({int((merged['drug_class'] == merged['ground_truth_drug']).sum())}"
          f"/{len(merged)})")
    print()
    print("NOTE: in-sample accuracy is optimistic (the model has seen "
          "these patients). For honest deployment metrics see "
          "derived/task1_cascade_b_threshold_report.md (OOF + holdout).")


if __name__ == "__main__":
    main()
