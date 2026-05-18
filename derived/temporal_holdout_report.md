# Temporal holdout evaluation

Train on all encounters with `encounter_arrival_date <= 2025-05-21`; test on `2025-05-22` only.

## Task 1 — drug ID at triage (4 classes)

Train n = 187, Test n = 74.

| model | logloss | accuracy | macro_auc | macro_prauc | prevalence_none | auc_none | prauc_none | brier_none | bss_none | prevalence_kraken | auc_kraken | prauc_kraken | brier_kraken | bss_kraken | prevalence_triton | auc_triton | prauc_triton | brier_triton | bss_triton | prevalence_coral | auc_coral | prauc_coral | brier_coral | bss_coral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 2.1635 | 0.3514 | 0.6003 | 0.3576 | 0.3919 | 0.6421 | 0.5904 | 0.2640 | -0.1077 | 0.3243 | 0.5400 | 0.3967 | 0.2805 | -0.2802 | 0.1622 | 0.6532 | 0.3024 | 0.2338 | -0.7205 | 0.1216 | 0.5658 | 0.1411 | 0.1472 | -0.3775 |
| rforest | 1.2193 | 0.4189 | 0.6908 | 0.4134 | 0.3919 | 0.7042 | 0.6385 | 0.2111 | 0.1142 | 0.3243 | 0.6292 | 0.4741 | 0.2179 | 0.0057 | 0.1622 | 0.7392 | 0.3196 | 0.1281 | 0.0574 | 0.1216 | 0.6906 | 0.2215 | 0.1118 | -0.0464 |
| hgb | 1.7043 | 0.4324 | 0.6847 | 0.4182 | 0.3919 | 0.7011 | 0.5992 | 0.2539 | -0.0654 | 0.3243 | 0.7250 | 0.5766 | 0.2303 | -0.0511 | 0.1622 | 0.6855 | 0.3084 | 0.1706 | -0.2555 | 0.1216 | 0.6274 | 0.1885 | 0.1597 | -0.4945 |

Best model: **rforest**

## Task 2 — deterioration at 4h (drug-positive cohort, 3 classes)

Train n = 112, Test n = 45.

| model | logloss | accuracy | macro_auc | macro_prauc | prevalence_discharge | auc_discharge | prauc_discharge | brier_discharge | bss_discharge | prevalence_floor | auc_floor | prauc_floor | brier_floor | bss_floor | prevalence_icu | auc_icu | prauc_icu | brier_icu | bss_icu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 0.3244 | 0.8889 | 0.9574 | 0.8845 | 0.7333 | 0.9747 | 0.9918 | 0.0665 | 0.6601 | 0.1556 | 0.9774 | 0.8917 | 0.0590 | 0.5510 | 0.1111 | 0.9200 | 0.7698 | 0.0460 | 0.5339 |
| rforest | 0.3188 | 0.8889 | 0.9849 | 0.9516 | 0.7333 | 0.9924 | 0.9972 | 0.0528 | 0.7302 | 0.1556 | 0.9624 | 0.8575 | 0.0700 | 0.4670 | 0.1111 | 1.0000 | 1.0000 | 0.0378 | 0.6173 |
| hgb | 0.5844 | 0.8667 | 0.8692 | 0.6533 | 0.7333 | 0.9293 | 0.9494 | 0.0353 | 0.8195 | 0.1556 | 0.8083 | 0.4429 | 0.1215 | 0.0748 | 0.1111 | 0.8700 | 0.5675 | 0.0807 | 0.1824 |

Best model: **rforest**
