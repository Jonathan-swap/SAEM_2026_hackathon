# Temporal holdout evaluation

Train on all encounters with `encounter_arrival_date <= 2025-05-21`; test on `2025-05-22` only.

## Task 1 — drug ID at triage (4 classes)

Train n = 187, Test n = 74.

| model | logloss | accuracy | macro_auc | brier_none | brier_kraken | brier_triton | brier_coral |
|---|---|---|---|---|---|---|---|
| logreg | 2.1635 | 0.3514 | 0.6003 | 0.2640 | 0.2805 | 0.2338 | 0.1472 |
| rforest | 1.2078 | 0.4189 | 0.7007 | 0.2065 | 0.2178 | 0.1277 | 0.1107 |
| hgb | 1.6995 | 0.4324 | 0.6849 | 0.2566 | 0.2290 | 0.1703 | 0.1551 |

Best model: **rforest**

## Task 2 — deterioration at 4h (drug-positive cohort, 3 classes)

Train n = 112, Test n = 45.

| model | logloss | accuracy | macro_auc | auc_discharge | auc_floor | auc_icu | brier_discharge | brier_floor | brier_icu |
|---|---|---|---|---|---|---|---|---|---|
| logreg | 0.3244 | 0.8889 | 0.9574 | 0.9747 | 0.9774 | 0.9200 | 0.0665 | 0.0590 | 0.0460 |
| rforest | 0.3130 | 0.8889 | 0.9887 | 0.9924 | 0.9737 | 1.0000 | 0.0514 | 0.0691 | 0.0361 |
| hgb | 0.5840 | 0.8667 | 0.8696 | 0.9293 | 0.8045 | 0.8750 | 0.0346 | 0.1214 | 0.0809 |

Best model: **rforest**
