# Task 1 cascade-variants evaluation

Pure analysis layer over the metrics emitted by `src/task1_drug_id/compare_cascades.py`. No model fitting here — read `task1_cascade_combinations_summary.csv` and compute rankings / win counts / consistency.

## 1. Headline metric wins per architecture

Counts the number of (split × model × metric) cells where each architecture ranks #1 across the four headline metrics (macro ROC-AUC, macro PR-AUC, accuracy, log-loss). Total cells = 2 splits × 3 models × 4 metrics = 24.

| Architecture | Wins | Win rate |
|---|---:|---:|
| Cascade B (tier-1 + K-vs-rest + prev) | 18 | 75% |
| Direct 4-class | 3 | 12% |
| Cascade A (tier-1 + tier-2-multi) | 3 | 12% |
| Cascade C (tier-1 + K-vs-rest + T-vs-C) | 2 | 8% |

## 2. Per-architecture summary (across 3 models)

### CV split

| Architecture | Macro AUC mean | best | std | Macro PR-AUC mean | best | std | Accuracy mean | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct 4-class | 0.6489 | 0.6770 | 0.0206 | 0.3556 | 0.3655 | 0.0096 | 0.4215 | 0.4521 |
| Cascade A (tier-1 + tier-2-multi) | 0.6376 | 0.6786 | 0.0290 | 0.3495 | 0.3854 | 0.0256 | 0.4368 | 0.5057 |
| Cascade B (tier-1 + K-vs-rest + prev) | 0.6599 | 0.6820 | 0.0221 | 0.3662 | 0.3849 | 0.0203 | 0.4381 | 0.4636 |
| Cascade C (tier-1 + K-vs-rest + T-vs-C) | 0.6410 | 0.6740 | 0.0263 | 0.3535 | 0.3837 | 0.0235 | 0.4240 | 0.4751 |

### TEMPORAL split

| Architecture | Macro AUC mean | best | std | Macro PR-AUC mean | best | std | Accuracy mean | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct 4-class | 0.6627 | 0.7029 | 0.0447 | 0.4015 | 0.4286 | 0.0313 | 0.3919 | 0.4324 |
| Cascade A (tier-1 + tier-2-multi) | 0.6662 | 0.7097 | 0.0424 | 0.4109 | 0.4428 | 0.0281 | 0.4234 | 0.4459 |
| Cascade B (tier-1 + K-vs-rest + prev) | 0.6990 | 0.7211 | 0.0275 | 0.4244 | 0.4414 | 0.0234 | 0.4550 | 0.4865 |
| Cascade C (tier-1 + K-vs-rest + T-vs-C) | 0.6673 | 0.7071 | 0.0430 | 0.4084 | 0.4435 | 0.0310 | 0.4459 | 0.4730 |

## 3. Stability across model families

How much does macro AUC vary when you swap logreg ↔ rforest ↔ hgb under the same architecture? Lower spread = more model-agnostic.

### CV split

| Architecture | mean | std | spread |
|---|---:|---:|---:|
| Direct 4-class | 0.6489 | 0.0206 | 0.0487 |
| Cascade B (tier-1 + K-vs-rest + prev) | 0.6599 | 0.0221 | 0.0523 |
| Cascade A (tier-1 + tier-2-multi) | 0.6376 | 0.0290 | 0.0636 |
| Cascade C (tier-1 + K-vs-rest + T-vs-C) | 0.6410 | 0.0263 | 0.0642 |

### TEMPORAL split

| Architecture | mean | std | spread |
|---|---:|---:|---:|
| Cascade B (tier-1 + K-vs-rest + prev) | 0.6990 | 0.0275 | 0.0608 |
| Cascade C (tier-1 + K-vs-rest + T-vs-C) | 0.6673 | 0.0430 | 0.0995 |
| Cascade A (tier-1 + tier-2-multi) | 0.6662 | 0.0424 | 0.1009 |
| Direct 4-class | 0.6627 | 0.0447 | 0.1026 |

## 4. Per-class OvR AUC winners (holdout only)

Which architecture has the highest one-vs-rest ROC-AUC for each class?

| Model | Class | Winning architecture | AUC |
|---|---|---|---:|
| logreg | None | Cascade A (tier-1 + tier-2-multi) | 0.678 |
| logreg | Kraken | Cascade A (tier-1 + tier-2-multi) | 0.576 |
| logreg | Triton | Cascade B (tier-1 + K-vs-rest + prev) | 0.741 |
| logreg | Coral | Cascade B (tier-1 + K-vs-rest + prev) | 0.663 |
| rforest | None | Direct 4-class | 0.715 |
| rforest | Kraken | Cascade A (tier-1 + tier-2-multi) | 0.662 |
| rforest | Triton | Cascade B (tier-1 + K-vs-rest + prev) | 0.777 |
| rforest | Coral | Cascade B (tier-1 + K-vs-rest + prev) | 0.735 |
| hgb | None | Direct 4-class | 0.701 |
| hgb | Kraken | Direct 4-class | 0.726 |
| hgb | Triton | Cascade B (tier-1 + K-vs-rest + prev) | 0.769 |
| hgb | Coral | Cascade B (tier-1 + K-vs-rest + prev) | 0.723 |

## 5. Consistency — does each cascade beat direct on **both** CV and holdout?

| Model | Cascade | Δ AUC (CV) | Δ AUC (holdout) | Beats direct on CV? | Holdout? | Both? |
|---|---|---:|---:|---|---|---|
| logreg | Cascade A (tier-1 + tier-2-multi) | -0.0133 | +0.0084 | no | YES | no |
| logreg | Cascade B (tier-1 + K-vs-rest + prev) | +0.0014 | +0.0600 | YES | YES | YES |
| logreg | Cascade C (tier-1 + K-vs-rest + T-vs-C) | -0.0186 | +0.0073 | no | YES | no |
| rforest | Cascade A (tier-1 + tier-2-multi) | +0.0016 | +0.0067 | YES | YES | YES |
| rforest | Cascade B (tier-1 + K-vs-rest + prev) | +0.0050 | +0.0182 | YES | YES | YES |
| rforest | Cascade C (tier-1 + K-vs-rest + T-vs-C) | -0.0031 | +0.0042 | no | YES | no |
| hgb | Cascade A (tier-1 + tier-2-multi) | -0.0221 | -0.0047 | no | no | no |
| hgb | Cascade B (tier-1 + K-vs-rest + prev) | +0.0267 | +0.0307 | YES | YES | YES |
| hgb | Cascade C (tier-1 + K-vs-rest + T-vs-C) | -0.0021 | +0.0022 | no | YES | no |

## 6. Deployment recommendation

**Recommended: `logreg` × **Cascade B (tier-1 + K-vs-rest + prev)**.**

This (model × architecture) pair beats direct 4-class on *both* the 5-fold CV (Δ = +0.0014) and the temporal holdout (Δ = +0.0600). Among all consistent winners it has the highest mean Δ macro AUC across the two splits (+0.0307).

Other architectures that also beat direct on both splits (for at least one model):
- `rforest` × Cascade A (tier-1 + tier-2-multi): CV Δ +0.0016, holdout Δ +0.0067
- `rforest` × Cascade B (tier-1 + K-vs-rest + prev): CV Δ +0.0050, holdout Δ +0.0182
- `hgb` × Cascade B (tier-1 + K-vs-rest + prev): CV Δ +0.0267, holdout Δ +0.0307
