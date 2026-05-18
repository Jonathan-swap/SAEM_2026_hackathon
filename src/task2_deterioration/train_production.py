"""Train the deployment Task-2 model on 100% of the chosen cohort and
save the artifacts to `production/task2/`.

Task 2 predicts disposition `encounter_disposition_label` in
{Discharge, Floor, ICU}. The brief's scope is the drug-positive
cohort (ground_truth_drug != 0, n=157). The default Task-2 model per
RUNBOOK §8a is rforest WITH drug-class probability features
(macro ROC-AUC 0.932 OOF, 0.980 holdout).

Class encoding:
  0 = Discharge
  1 = Floor
  2 = ICU

Run:
    .venv/Scripts/python.exe src/task2_deterioration/train_production.py
    .venv/Scripts/python.exe src/task2_deterioration/train_production.py --cohort all

Artifacts written to production/task2/:
  preprocessor.joblib            ColumnTransformer fitted on the cohort
  model.joblib                   rforest, 3-class disposition predictor
  metadata.json                  feature schema, cohort, class mapping,
                                  sklearn/numpy versions, fitted date,
                                  feature_columns_md5 fingerprint
  predict.py                     stand-alone inference script
  README.md                      how to load + use the artifacts
"""
from __future__ import annotations

import argparse
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
PRODUCTION = ROOT / "production" / "task2"

CLASS_NAMES = ["Discharge", "Floor", "ICU"]
PROB_COLS = ["p_kraken", "p_triton", "p_coral", "p_none"]
TEXT_COL = "triage_brief_note"


def load_features_and_y(cohort: str = "drug-positive"
                         ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load 4h features + Task-1 probability features + disposition y.

    Matches the column logic of `train_baseline.py::load_data` exactly
    so the production model trains on the same shape its CV cousin used.
    """
    X = pd.read_csv(DERIVED / "features_fourh.csv")
    outcomes = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "ground_truth_drug",
         "encounter_disposition_label"]]
    probs = pd.read_csv(DERIVED / "probs_avg.csv",
                        keep_default_na=False, na_values=[""])[
        ["encounter_id", *PROB_COLS]]

    for c in ("encounter_disposition_label", "ground_truth_drug",
              "ground_truth_drug_name"):
        if c in X.columns:
            X = X.drop(columns=[c])

    df = X.merge(outcomes, on="encounter_id", how="inner")
    df = df.merge(probs, on="encounter_id", how="inner")

    n_before = len(df)
    if cohort == "drug-positive":
        df = df[df["ground_truth_drug"] != 0].reset_index(drop=True)
        print(f"Cohort filter (drug-positive): {n_before} -> {len(df)}")
    elif cohort == "all":
        print(f"Cohort: all {n_before} patients")
    else:
        raise ValueError(f"Unknown cohort: {cohort!r}")

    label_map = {c: i for i, c in enumerate(CLASS_NAMES)}
    y = df["encounter_disposition_label"].map(label_map).to_numpy()
    if pd.isna(y).any():
        bad = df.loc[pd.isna(y), "encounter_disposition_label"].unique()
        raise ValueError(f"Unrecognised disposition labels: {list(bad)}")

    ids = df["encounter_id"].to_numpy()
    drop = ["encounter_id", "encounter_arrival_date", "ground_truth_drug",
            "ground_truth_drug_name", "encounter_disposition_label"]
    X_df = df.drop(columns=[c for c in drop if c in df.columns])
    return X_df, y.astype(int), ids


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
    """Same hyperparameters as the baseline rforest (RUNBOOK §8a best)."""
    return RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


def write_predict_script(production_dir: Path) -> None:
    code = '''"""Stand-alone Task-2 disposition inference.

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
    python production/task2/predict.py path/to/features_fourh.csv \\
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
        print(f"WARNING: {len(missing)} feature(s) missing from input "
              f"(filled with 0): {missing[:5]}"
              f"{' ...' if len(missing) > 5 else ''}")
        for c in missing:
            Xf[c] = 0
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
            "<probs_avg.csv> <out_predictions.csv>\\n"
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
'''
    (production_dir / "predict.py").write_text(code, encoding="utf-8")


def write_readme(production_dir: Path, meta: dict) -> None:
    md = f"""# Task-2 production model (disposition, rforest)

Trained on **100% of the {meta['cohort']} cohort**
(n={meta['n_train']} encounters, class counts {meta['class_counts']}).
Best Task-2 model per RUNBOOK §8a (rforest WITH drug-class probability
features, macro ROC-AUC 0.932 OOF / 0.980 holdout).

## Contents

| File | Purpose |
|---|---|
| `preprocessor.joblib` | sklearn ColumnTransformer (TF-IDF on `triage_brief_note`, OHE on categoricals, median-impute + StandardScaler on numerics). Fitted on the {meta['cohort']} cohort. |
| `model.joblib` | rforest, 3-class disposition predictor (Discharge / Floor / ICU). |
| `metadata.json` | Feature schema (including the 4 LLM-agent probability features `p_kraken/p_triton/p_coral/p_none`), cohort, class mapping, version pins. |
| `predict.py` | Stand-alone inference: load_model + predict(df, probs) -> encounter_id, disposition_class, 3 probs. |

## Class encoding

| drug_class | Label |
|---:|---|
| 0 | Discharge |
| 1 | Floor |
| 2 | ICU |

## Quick inference

```python
import pandas as pd
from production.task2.predict import load_model, predict

model = load_model()
X = pd.read_csv("derived/features_fourh.csv")
probs = pd.read_csv("derived/probs_avg.csv")
out = predict(model, X, probs)
out.to_csv("derived/task2_predictions.csv", index=False)
```

Or from the shell:
```
python production/task2/predict.py path/to/features_fourh.csv \\
    path/to/probs_avg.csv path/to/out.csv
```

## Required inputs for new data

- **`features_fourh.csv`** — 4-hour-horizon features (Task-2 inputs).
- **`probs_avg.csv`** — 10-agent LLM consensus probabilities
  (`p_kraken, p_triton, p_coral, p_none`). Generated via the
  10-agent step in `run_pipeline.py`. In a Task-1-only deployment
  scenario, replace this with the cascade-derived 4-class
  probabilities from `production/task1/predict.py`.

## Versioning

- Python: `{meta['python']}`
- scikit-learn: `{meta['sklearn_version']}`
- numpy: `{meta['numpy_version']}`
- Fitted: `{meta['fitted_utc']}`
- Feature columns hash: `{meta['feature_columns_md5']}` (md5 of the
  ordered `feature_columns` list — use it to detect schema drift in
  new data)
"""
    (production_dir / "README.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", choices=["drug-positive", "all"],
                    default="drug-positive",
                    help="Cohort to train on. Default: drug-positive "
                         "(matches RUNBOOK §8a default and the brief).")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Train Task-2 deployment model (rforest, 100% of "
          f"{args.cohort} cohort)")
    print("=" * 78)

    X_df, y, ids = load_features_and_y(cohort=args.cohort)
    counts = np.bincount(y, minlength=len(CLASS_NAMES)).tolist()
    print(f"X: {X_df.shape}   y class counts: {counts}  "
          f"({dict(zip(CLASS_NAMES, counts))})")

    PRODUCTION.mkdir(parents=True, exist_ok=True)

    pre = make_preprocessor(X_df)
    pre.fit(X_df)
    Xp = np.asarray(pre.transform(X_df), dtype=float)
    print(f"Preprocessor fitted; transformed shape: {Xp.shape}")

    model = make_rforest()
    model.fit(Xp, y)
    p_train = model.predict_proba(Xp)
    in_sample_acc = float((p_train.argmax(axis=1) == y).mean())
    print(f"rforest fitted; in-sample accuracy = {in_sample_acc:.4f}")

    joblib.dump(pre, PRODUCTION / "preprocessor.joblib")
    joblib.dump(model, PRODUCTION / "model.joblib")

    feature_cols = list(X_df.columns)
    feat_md5 = hashlib.md5(
        "\n".join(feature_cols).encode("utf-8")
    ).hexdigest()
    meta = {
        "model": "task2_rforest_v1",
        "task": "task2_disposition",
        "cohort": args.cohort,
        "class_mapping": {str(i): c for i, c in enumerate(CLASS_NAMES)},
        "feature_columns": feature_cols,
        "feature_columns_md5": feat_md5,
        "prob_columns_required": PROB_COLS,
        "n_train": int(len(y)),
        "class_counts": counts,
        "rforest_hyperparams": {
            "n_estimators": 400,
            "max_depth": 8,
            "min_samples_leaf": 4,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "training_note": (
            "Trained on 100% of the chosen cohort — no fold held out. "
            "For honest deployment metrics see "
            "derived/task2_baseline_summary.csv (5-fold CV macro AUC "
            "0.932) and section 8 of RUNBOOK.md (temporal holdout "
            "macro AUC 0.980)."
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
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name:38s} {size_kb:8.1f} KB")

    print()
    print("Smoke test (load + score the cohort):")
    sys.path.insert(0, str(PRODUCTION))
    import importlib
    predict_mod = importlib.import_module("predict")
    model_pkg = predict_mod.load_model(PRODUCTION)
    X_raw = pd.read_csv(DERIVED / "features_fourh.csv")
    probs_raw = pd.read_csv(DERIVED / "probs_avg.csv",
                              keep_default_na=False, na_values=[""])
    if args.cohort == "drug-positive":
        truth = pd.read_csv(DERIVED / "outcomes.csv")[
            ["encounter_id", "ground_truth_drug",
             "encounter_disposition_label"]]
        cohort_ids = truth.loc[truth["ground_truth_drug"] != 0,
                                "encounter_id"]
        X_raw = X_raw[X_raw["encounter_id"].isin(cohort_ids)]
    preds = predict_mod.predict(model_pkg, X_raw, probs_raw)
    truth = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "encounter_disposition_label"]]
    merged = preds.merge(truth, on="encounter_id", how="inner")
    label_map = {c: i for i, c in enumerate(CLASS_NAMES)}
    merged["truth_idx"] = merged["encounter_disposition_label"].map(label_map)
    correct = (merged["disposition_class"] == merged["truth_idx"]).sum()
    acc = correct / len(merged)
    print(f"  loaded {len(preds)} predictions; "
          f"in-sample accuracy = {acc:.4f}  ({int(correct)}/{len(merged)})")
    print()
    print("NOTE: in-sample accuracy is optimistic. For honest "
          "deployment metrics see RUNBOOK section 8 (CV + temporal "
          "holdout).")


if __name__ == "__main__":
    main()
