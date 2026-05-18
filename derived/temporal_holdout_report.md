# Temporal holdout evaluation

Train on all encounters with `encounter_arrival_date <= 2025-05-21`; test on `2025-05-22` only.

## Task 1 — drug ID at triage (4 classes)

Train n = 187, Test n = 74.

| model | logloss | accuracy | macro_auc | macro_prauc | prevalence_none | auc_none | prauc_none | brier_none | bss_none | prevalence_kraken | auc_kraken | prauc_kraken | brier_kraken | bss_kraken | prevalence_triton | auc_triton | prauc_triton | brier_triton | bss_triton | prevalence_coral | auc_coral | prauc_coral | brier_coral | bss_coral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 2.1635 | 0.3514 | 0.6003 | 0.3576 | 0.3919 | 0.6421 | 0.5904 | 0.2640 | -0.1077 | 0.3243 | 0.5400 | 0.3967 | 0.2805 | -0.2802 | 0.1622 | 0.6532 | 0.3024 | 0.2338 | -0.7205 | 0.1216 | 0.5658 | 0.1411 | 0.1472 | -0.3775 |
| rforest | 1.2194 | 0.3784 | 0.6951 | 0.4066 | 0.3919 | 0.7088 | 0.6213 | 0.2114 | 0.1128 | 0.3243 | 0.6142 | 0.4125 | 0.2216 | -0.0114 | 0.1622 | 0.7567 | 0.3521 | 0.1265 | 0.0692 | 0.1216 | 0.7009 | 0.2406 | 0.1114 | -0.0429 |
| hgb | 1.7021 | 0.4324 | 0.6850 | 0.4183 | 0.3919 | 0.7011 | 0.5989 | 0.2544 | -0.0676 | 0.3243 | 0.7258 | 0.5773 | 0.2300 | -0.0496 | 0.1622 | 0.6855 | 0.3084 | 0.1707 | -0.2562 | 0.1216 | 0.6274 | 0.1885 | 0.1585 | -0.4838 |

Best model: **rforest**

## Task 2 — deterioration at 4h (drug-positive cohort, 3 classes)

Train n = 112, Test n = 45.

| model | logloss | accuracy | macro_auc | macro_prauc | prevalence_discharge | auc_discharge | prauc_discharge | brier_discharge | bss_discharge | prevalence_floor | auc_floor | prauc_floor | brier_floor | bss_floor | prevalence_icu | auc_icu | prauc_icu | brier_icu | bss_icu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 0.3244 | 0.8889 | 0.9574 | 0.8845 | 0.7333 | 0.9747 | 0.9918 | 0.0665 | 0.6601 | 0.1556 | 0.9774 | 0.8917 | 0.0590 | 0.5510 | 0.1111 | 0.9200 | 0.7698 | 0.0460 | 0.5339 |
| rforest | 0.3215 | 0.8889 | 0.9783 | 0.9240 | 0.7333 | 0.9899 | 0.9963 | 0.0524 | 0.7322 | 0.1556 | 0.9549 | 0.8329 | 0.0728 | 0.4455 | 0.1111 | 0.9900 | 0.9429 | 0.0383 | 0.6127 |
| hgb | 0.5921 | 0.8667 | 0.8675 | 0.6432 | 0.7333 | 0.9268 | 0.9444 | 0.0350 | 0.8211 | 0.1556 | 0.8008 | 0.4126 | 0.1222 | 0.0699 | 0.1111 | 0.8750 | 0.5726 | 0.0815 | 0.1747 |

Best model: **rforest**
