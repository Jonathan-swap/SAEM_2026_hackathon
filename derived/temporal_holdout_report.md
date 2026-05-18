# Temporal holdout evaluation

Train on all encounters with `encounter_arrival_date <= 2025-05-21`; test on `2025-05-22` only.

## Task 1 — drug ID at triage (4 classes)

Train n = 187, Test n = 74.

| model | logloss | accuracy | macro_auc | macro_prauc | prevalence_none | auc_none | prauc_none | brier_none | bss_none | prevalence_kraken | auc_kraken | prauc_kraken | brier_kraken | bss_kraken | prevalence_triton | auc_triton | prauc_triton | brier_triton | bss_triton | prevalence_coral | auc_coral | prauc_coral | brier_coral | bss_coral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 2.1635 | 0.3514 | 0.6003 | 0.3576 | 0.3919 | 0.6421 | 0.5904 | 0.2640 | -0.1077 | 0.3243 | 0.5400 | 0.3967 | 0.2805 | -0.2802 | 0.1622 | 0.6532 | 0.3024 | 0.2338 | -0.7205 | 0.1216 | 0.5658 | 0.1411 | 0.1472 | -0.3775 |
| rforest | 1.2085 | 0.4054 | 0.6952 | 0.4326 | 0.3919 | 0.7172 | 0.6377 | 0.2072 | 0.1303 | 0.3243 | 0.6300 | 0.4755 | 0.2166 | 0.0115 | 0.1622 | 0.7379 | 0.3221 | 0.1284 | 0.0546 | 0.1216 | 0.6957 | 0.2951 | 0.1102 | -0.0320 |
| hgb | 1.7017 | 0.4324 | 0.6846 | 0.4178 | 0.3919 | 0.7011 | 0.5989 | 0.2544 | -0.0676 | 0.3243 | 0.7258 | 0.5773 | 0.2300 | -0.0496 | 0.1622 | 0.6841 | 0.3067 | 0.1708 | -0.2570 | 0.1216 | 0.6274 | 0.1885 | 0.1583 | -0.4822 |

Best model: **rforest**

## Task 2 — deterioration at 4h (drug-positive cohort, 3 classes)

Train n = 187, Test n = 74.

| model | logloss | accuracy | macro_auc | macro_prauc | prevalence_discharge | auc_discharge | prauc_discharge | brier_discharge | bss_discharge | prevalence_floor | auc_floor | prauc_floor | brier_floor | bss_floor | prevalence_icu | auc_icu | prauc_icu | brier_icu | bss_icu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 0.5123 | 0.8108 | 0.9215 | 0.8442 | 0.6216 | 0.9713 | 0.9816 | 0.0672 | 0.7144 | 0.2297 | 0.8854 | 0.7495 | 0.1220 | 0.3106 | 0.1486 | 0.9076 | 0.8015 | 0.0682 | 0.4609 |
| rforest | 0.2810 | 0.9054 | 0.9881 | 0.9690 | 0.6216 | 0.9938 | 0.9962 | 0.0448 | 0.8094 | 0.2297 | 0.9763 | 0.9401 | 0.0675 | 0.6187 | 0.1486 | 0.9942 | 0.9709 | 0.0275 | 0.7829 |
| hgb | 0.2890 | 0.8784 | 0.9718 | 0.9202 | 0.6216 | 0.9884 | 0.9928 | 0.0431 | 0.8166 | 0.2297 | 0.9474 | 0.8390 | 0.0860 | 0.5142 | 0.1486 | 0.9798 | 0.9288 | 0.0436 | 0.6556 |

Best model: **rforest**
