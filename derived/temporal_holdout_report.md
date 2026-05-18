# Temporal holdout evaluation

Train on all encounters with `encounter_arrival_date <= 2025-05-21`; test on `2025-05-22` only.

## Task 1 — drug ID at triage (4 classes)

Train n = 187, Test n = 74.

| model | logloss | accuracy | macro_auc | macro_prauc | prevalence_none | auc_none | prauc_none | brier_none | bss_none | prevalence_kraken | auc_kraken | prauc_kraken | brier_kraken | bss_kraken | prevalence_triton | auc_triton | prauc_triton | brier_triton | bss_triton | prevalence_coral | auc_coral | prauc_coral | brier_coral | bss_coral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 2.1635 | 0.3514 | 0.6003 | 0.3576 | 0.3919 | 0.6421 | 0.5904 | 0.2640 | -0.1077 | 0.3243 | 0.5400 | 0.3967 | 0.2805 | -0.2802 | 0.1622 | 0.6532 | 0.3024 | 0.2338 | -0.7205 | 0.1216 | 0.5658 | 0.1411 | 0.1472 | -0.3775 |
| rforest | 1.2102 | 0.3784 | 0.6989 | 0.4264 | 0.3919 | 0.7272 | 0.6780 | 0.2063 | 0.1344 | 0.3243 | 0.6275 | 0.4461 | 0.2188 | 0.0015 | 0.1622 | 0.7487 | 0.3391 | 0.1265 | 0.0690 | 0.1216 | 0.6923 | 0.2422 | 0.1114 | -0.0431 |
| hgb | 1.7024 | 0.4324 | 0.6848 | 0.4180 | 0.3919 | 0.7011 | 0.5989 | 0.2544 | -0.0673 | 0.3243 | 0.7267 | 0.5780 | 0.2299 | -0.0492 | 0.1622 | 0.6841 | 0.3067 | 0.1706 | -0.2558 | 0.1216 | 0.6274 | 0.1885 | 0.1586 | -0.4849 |

Best model: **rforest**

## Task 2 — deterioration at 4h (drug-positive cohort, 3 classes)

Train n = 187, Test n = 74.

| model | logloss | accuracy | macro_auc | macro_prauc | prevalence_discharge | auc_discharge | prauc_discharge | brier_discharge | bss_discharge | prevalence_floor | auc_floor | prauc_floor | brier_floor | bss_floor | prevalence_icu | auc_icu | prauc_icu | brier_icu | bss_icu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 0.5125 | 0.8108 | 0.9199 | 0.8355 | 0.6216 | 0.9713 | 0.9813 | 0.0664 | 0.7176 | 0.2297 | 0.8824 | 0.7245 | 0.1223 | 0.3088 | 0.1486 | 0.9062 | 0.8008 | 0.0675 | 0.4668 |
| rforest | 0.2736 | 0.9189 | 0.9888 | 0.9709 | 0.6216 | 0.9938 | 0.9961 | 0.0430 | 0.8171 | 0.2297 | 0.9783 | 0.9457 | 0.0652 | 0.6315 | 0.1486 | 0.9942 | 0.9709 | 0.0267 | 0.7890 |
| hgb | 0.2914 | 0.8649 | 0.9723 | 0.9219 | 0.6216 | 0.9884 | 0.9929 | 0.0433 | 0.8159 | 0.2297 | 0.9474 | 0.8463 | 0.0866 | 0.5108 | 0.1486 | 0.9812 | 0.9267 | 0.0447 | 0.6467 |

Best model: **rforest**
