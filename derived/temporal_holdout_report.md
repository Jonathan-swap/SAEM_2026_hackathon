# Temporal holdout evaluation

Train on all encounters with `encounter_arrival_date <= 2025-05-21`; test on `2025-05-22` only.

## Task 1 — drug ID at triage (4 classes)

Train n = 187, Test n = 74.

| model | logloss | accuracy | macro_auc | macro_prauc | prevalence_none | auc_none | prauc_none | brier_none | bss_none | prevalence_kraken | auc_kraken | prauc_kraken | brier_kraken | bss_kraken | prevalence_triton | auc_triton | prauc_triton | brier_triton | bss_triton | prevalence_coral | auc_coral | prauc_coral | brier_coral | bss_coral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 2.1635 | 0.3514 | 0.6003 | 0.3576 | 0.3919 | 0.6421 | 0.5904 | 0.2640 | -0.1077 | 0.3243 | 0.5400 | 0.3967 | 0.2805 | -0.2802 | 0.1622 | 0.6532 | 0.3024 | 0.2338 | -0.7205 | 0.1216 | 0.5658 | 0.1411 | 0.1472 | -0.3775 |
| rforest | 1.2098 | 0.3919 | 0.6958 | 0.4287 | 0.3919 | 0.7211 | 0.6464 | 0.2064 | 0.1340 | 0.3243 | 0.6308 | 0.4502 | 0.2191 | 0.0001 | 0.1622 | 0.7527 | 0.3350 | 0.1266 | 0.0685 | 0.1216 | 0.6786 | 0.2831 | 0.1124 | -0.0525 |
| hgb | 1.6994 | 0.4324 | 0.6849 | 0.4189 | 0.3919 | 0.6973 | 0.5960 | 0.2566 | -0.0768 | 0.3243 | 0.7300 | 0.5816 | 0.2290 | -0.0452 | 0.1622 | 0.6882 | 0.3103 | 0.1703 | -0.2535 | 0.1216 | 0.6239 | 0.1875 | 0.1551 | -0.4523 |

Best model: **rforest**

## Task 2 — deterioration at 4h (drug-positive cohort, 3 classes)

Train n = 112, Test n = 45.

| model | logloss | accuracy | macro_auc | macro_prauc | prevalence_discharge | auc_discharge | prauc_discharge | brier_discharge | bss_discharge | prevalence_floor | auc_floor | prauc_floor | brier_floor | bss_floor | prevalence_icu | auc_icu | prauc_icu | brier_icu | bss_icu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 0.3244 | 0.8889 | 0.9574 | 0.8845 | 0.7333 | 0.9747 | 0.9918 | 0.0665 | 0.6601 | 0.1556 | 0.9774 | 0.8917 | 0.0590 | 0.5510 | 0.1111 | 0.9200 | 0.7698 | 0.0460 | 0.5339 |
| rforest | 0.3200 | 0.8889 | 0.9849 | 0.9516 | 0.7333 | 0.9924 | 0.9972 | 0.0545 | 0.7215 | 0.1556 | 0.9624 | 0.8575 | 0.0704 | 0.4644 | 0.1111 | 1.0000 | 1.0000 | 0.0369 | 0.6260 |
| hgb | 0.5840 | 0.8667 | 0.8696 | 0.6470 | 0.7333 | 0.9293 | 0.9494 | 0.0346 | 0.8232 | 0.1556 | 0.8045 | 0.4190 | 0.1214 | 0.0759 | 0.1111 | 0.8750 | 0.5726 | 0.0809 | 0.1811 |

Best model: **rforest**
