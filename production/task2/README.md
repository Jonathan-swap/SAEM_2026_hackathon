# Task-2 production model (disposition, rforest)

Trained on **100% of the drug-positive cohort**
(n=157 encounters, class counts [121, 22, 14]).
Best Task-2 model per RUNBOOK §8a (rforest WITH drug-class probability
features, macro ROC-AUC 0.932 OOF / 0.980 holdout).

## Contents

| File | Purpose |
|---|---|
| `preprocessor.joblib` | sklearn ColumnTransformer (TF-IDF on `triage_brief_note`, OHE on categoricals, median-impute + StandardScaler on numerics). Fitted on the drug-positive cohort. |
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
python production/task2/predict.py path/to/features_fourh.csv \
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

- Python: `3.13.11`
- scikit-learn: `1.8.0`
- numpy: `2.4.4`
- Fitted: `2026-05-18T16:32:28Z`
- Feature columns hash: `d2cef7534b6c84e3b73dd3295734041f` (md5 of the
  ordered `feature_columns` list — use it to detect schema drift in
  new data)
