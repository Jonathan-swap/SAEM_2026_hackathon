# Task-1 production model (Cascade-B, rforest)

Trained on **100% of the dataset** (n=261 encounters,
class counts [104, 58, 51, 48]). Picked decision thresholds frozen
from the macro-F1 optimum on 5-fold OOF — they are NOT re-picked at the
100% retrain, because doing so would optimistically bias the deployment.

## Contents

| File | Purpose |
|---|---|
| `preprocessor.joblib` | sklearn ColumnTransformer (TF-IDF on `triage_brief_note`, OHE on categoricals, median-impute + StandardScaler on numerics). Fitted on all 261 encounters. |
| `tier1_model.joblib` | rforest, predicts `P(drug-positive)`. Trained on all 261 encounters with binary label `ground_truth_drug != 0`. |
| `kraken_vs_rest_model.joblib` | rforest, predicts `P(Kraken \| drug-positive)`. Trained on the 157 drug-positive encounters with binary label `ground_truth_drug == 1`. |
| `metadata.json` | Frozen thresholds, prevalence, feature schema, version pins. |
| `predict.py` | Stand-alone inference: load the model and score new encounters. |

## Frozen decision rule

```text
if  P(drug)     <  tau_drug       -> None       (drug_class = 0)
elif P(K|drug)  >= tau_kraken     -> Kraken     (drug_class = 1)
elif md5(eid)/2^64 <  triton_prev -> Triton     (drug_class = 2)
else                              -> Coral      (drug_class = 3)

tau_drug    = 0.57
tau_kraken  = 0.45
triton_prev = 0.5152
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

- Python: `3.13.11`
- scikit-learn: `1.8.0`
- numpy: `2.4.4`
- Fitted: `2026-05-18T16:22:20Z`
- Feature columns hash: `e155a51a7ed5d127d78b3e0d399e9dcb` (sha-style fingerprint
  of the `feature_columns` list; use it to detect schema drift in new data)
