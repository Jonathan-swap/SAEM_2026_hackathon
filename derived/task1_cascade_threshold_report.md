# Cascade-C threshold-based 4-class predictions

Single model family: **rforest** (best Cascade-C model per RUNBOOK §7h). Thresholds picked on 5-fold OOF (n=261), then applied unchanged to the temporal holdout (test = last day).

## Picked thresholds

| Criterion | Stage | τ | Sensitivity | Specificity | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| youden | Tier-1 (drug vs no-drug) | 0.572 | 0.682 | 0.769 | 0.817 | 0.743 |
| youden | K-vs-rest (Kraken vs Triton/Coral) | 0.517 | 0.655 | 0.859 | 0.731 | 0.691 |
| youden | T-vs-C (Triton vs Coral) | 0.571 | 0.392 | 0.688 | 0.571 | 0.465 |
| max_f1 | Tier-1 (drug vs no-drug) | 0.189 | 1.000 | 0.010 | 0.604 | 0.753 |
| max_f1 | K-vs-rest (Kraken vs Triton/Coral) | 0.517 | 0.655 | 0.859 | 0.731 | 0.691 |
| max_f1 | T-vs-C (Triton vs Coral) | 0.202 | 1.000 | 0.000 | 0.515 | 0.680 |
| sens_at_least_90 | Tier-1 (drug vs no-drug) | 0.296 | 0.904 | 0.163 | 0.620 | 0.736 |
| sens_at_least_90 | K-vs-rest (Kraken vs Triton/Coral) | 0.193 | 0.914 | 0.232 | 0.411 | 0.567 |
| sens_at_least_90 | T-vs-C (Triton vs Coral) | 0.349 | 0.902 | 0.062 | 0.505 | 0.648 |

## 4-class metrics (5-fold OOF, n=261)

| Criterion | Accuracy | Macro F1 | F1 None | F1 Kraken | F1 Triton | F1 Coral |
|---|---:|---:|---:|---:|---:|---:|
| youden | 0.498 | 0.424 | 0.684 | 0.240 | 0.392 | 0.379 |
| max_f1 | 0.326 | 0.232 | 0.019 | 0.420 | 0.489 | 0.000 |
| sens_at_least_90 | 0.272 | 0.220 | 0.250 | 0.324 | 0.306 | 0.000 |

### Confusion matrices — OOF

**youden** — accuracy 0.498, macro F1 0.424

| true \ pred | None | Kraken | Triton | Coral |
|---|---:|---:|---:|---:|
| None | 80 | 6 | 9 | 9 |
| Kraken Candy | 31 | 9 | 5 | 13 |
| Triton Tabs | 8 | 0 | 19 | 24 |
| Coral Dust | 11 | 2 | 13 | 22 |

**max_f1** — accuracy 0.326, macro F1 0.232

| true \ pred | None | Kraken | Triton | Coral |
|---|---:|---:|---:|---:|
| None | 1 | 71 | 32 | 0 |
| Kraken Candy | 0 | 38 | 20 | 0 |
| Triton Tabs | 0 | 5 | 46 | 0 |
| Coral Dust | 0 | 9 | 39 | 0 |

**sens_at_least_90** — accuracy 0.272, macro F1 0.220

| true \ pred | None | Kraken | Triton | Coral |
|---|---:|---:|---:|---:|
| None | 17 | 81 | 6 | 0 |
| Kraken Candy | 12 | 41 | 5 | 0 |
| Triton Tabs | 2 | 36 | 13 | 0 |
| Coral Dust | 1 | 37 | 10 | 0 |

## 4-class metrics (temporal holdout, n=74)

| Criterion | Accuracy | Macro F1 | F1 None | F1 Kraken | F1 Triton | F1 Coral |
|---|---:|---:|---:|---:|---:|---:|
| youden | 0.446 | 0.358 | 0.657 | 0.148 | 0.316 | 0.312 |
| max_f1 | 0.284 | 0.191 | 0.000 | 0.400 | 0.364 | 0.000 |
| sens_at_least_90 | 0.405 | 0.282 | 0.316 | 0.525 | 0.286 | 0.000 |

### Confusion matrices — holdout

**youden** — accuracy 0.446, macro F1 0.358

| true \ pred | None | Kraken | Triton | Coral |
|---|---:|---:|---:|---:|
| None | 23 | 1 | 2 | 3 |
| Kraken Candy | 13 | 2 | 1 | 8 |
| Triton Tabs | 2 | 0 | 3 | 7 |
| Coral Dust | 3 | 0 | 1 | 5 |

**max_f1** — accuracy 0.284, macro F1 0.191

| true \ pred | None | Kraken | Triton | Coral |
|---|---:|---:|---:|---:|
| None | 0 | 18 | 11 | 0 |
| Kraken Candy | 0 | 11 | 13 | 0 |
| Triton Tabs | 0 | 2 | 10 | 0 |
| Coral Dust | 0 | 0 | 9 | 0 |

**sens_at_least_90** — accuracy 0.405, macro F1 0.282

| true \ pred | None | Kraken | Triton | Coral |
|---|---:|---:|---:|---:|
| None | 6 | 21 | 2 | 0 |
| Kraken Candy | 1 | 21 | 2 | 0 |
| Triton Tabs | 1 | 8 | 3 | 0 |
| Coral Dust | 1 | 6 | 2 | 0 |

## Files

- `derived/task1_cascade_thresholds.csv` — picked thresholds + diagnostics
- `derived/task1_cascade_threshold_labels.csv` — per-encounter labels (every criterion × split)
