# Task 1 cascade-variants evaluation

Pure analysis layer over the metrics emitted by `src/task1_drug_id/compare_cascades.py`. No model fitting here — read `task1_cascade_combinations_summary.csv` and compute rankings / win counts / consistency.

## 1. Headline metric wins per architecture

Counts the number of (split × model × metric) cells where each architecture ranks #1 across the four headline metrics (macro ROC-AUC, macro PR-AUC, accuracy, log-loss). Total cells = 2 splits × 3 models × 4 metrics = 24.

| Architecture | Wins | Win rate |
|---|---:|---:|
| Cascade B (tier-1 + K-vs-rest + prev) | 19 | 79% |
| Cascade A (tier-1 + tier-2-multi) | 4 | 17% |
| Direct 4-class | 3 | 12% |
| Cascade C (tier-1 + K-vs-rest + T-vs-C) | 3 | 12% |

## 2. Per-architecture summary (across 3 models)

### CV split

| Architecture | Macro AUC mean | best | std | Macro PR-AUC mean | best | std | Accuracy mean | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct 4-class | 0.6489 | 0.6770 | 0.0206 | 0.3553 | 0.3648 | 0.0093 | 0.4176 | 0.4521 |
| Cascade A (tier-1 + tier-2-multi) | 0.6384 | 0.6809 | 0.0301 | 0.3512 | 0.3905 | 0.0280 | 0.4304 | 0.4866 |
| Cascade B (tier-1 + K-vs-rest + prev) | 0.6600 | 0.6825 | 0.0222 | 0.3671 | 0.3876 | 0.0211 | 0.4444 | 0.4828 |
| Cascade C (tier-1 + K-vs-rest + T-vs-C) | 0.6419 | 0.6767 | 0.0274 | 0.3553 | 0.3892 | 0.0259 | 0.4278 | 0.4866 |

### TEMPORAL split

| Architecture | Macro AUC mean | best | std | Macro PR-AUC mean | best | std | Accuracy mean | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct 4-class | 0.6581 | 0.6894 | 0.0409 | 0.3979 | 0.4186 | 0.0285 | 0.4054 | 0.4324 |
| Cascade A (tier-1 + tier-2-multi) | 0.6688 | 0.7082 | 0.0432 | 0.4134 | 0.4478 | 0.0301 | 0.4189 | 0.4459 |
| Cascade B (tier-1 + K-vs-rest + prev) | 0.7015 | 0.7250 | 0.0292 | 0.4306 | 0.4537 | 0.0279 | 0.4550 | 0.4865 |
| Cascade C (tier-1 + K-vs-rest + T-vs-C) | 0.6730 | 0.7124 | 0.0465 | 0.4175 | 0.4539 | 0.0362 | 0.4414 | 0.4730 |

## 3. Stability across model families

How much does macro AUC vary when you swap logreg ↔ rforest ↔ hgb under the same architecture? Lower spread = more model-agnostic.

### CV split

| Architecture | mean | std | spread |
|---|---:|---:|---:|
| Direct 4-class | 0.6489 | 0.0206 | 0.0487 |
| Cascade B (tier-1 + K-vs-rest + prev) | 0.6600 | 0.0222 | 0.0527 |
| Cascade A (tier-1 + tier-2-multi) | 0.6384 | 0.0301 | 0.0659 |
| Cascade C (tier-1 + K-vs-rest + T-vs-C) | 0.6419 | 0.0274 | 0.0670 |

### TEMPORAL split

| Architecture | mean | std | spread |
|---|---:|---:|---:|
| Cascade B (tier-1 + K-vs-rest + prev) | 0.7015 | 0.0292 | 0.0647 |
| Direct 4-class | 0.6581 | 0.0409 | 0.0891 |
| Cascade A (tier-1 + tier-2-multi) | 0.6688 | 0.0432 | 0.0995 |
| Cascade C (tier-1 + K-vs-rest + T-vs-C) | 0.6730 | 0.0465 | 0.1047 |

## 4. Per-class OvR AUC winners (holdout only)

Which architecture has the highest one-vs-rest ROC-AUC for each class?

| Model | Class | Winning architecture | AUC |
|---|---|---|---:|
| logreg | None | Cascade A (tier-1 + tier-2-multi) | 0.678 |
| logreg | Kraken | Cascade A (tier-1 + tier-2-multi) | 0.576 |
| logreg | Triton | Cascade B (tier-1 + K-vs-rest + prev) | 0.741 |
| logreg | Coral | Cascade B (tier-1 + K-vs-rest + prev) | 0.663 |
| rforest | None | Cascade A (tier-1 + tier-2-multi) | 0.716 |
| rforest | Kraken | Cascade B (tier-1 + K-vs-rest + prev) | 0.649 |
| rforest | Triton | Cascade B (tier-1 + K-vs-rest + prev) | 0.773 |
| rforest | Coral | Cascade B (tier-1 + K-vs-rest + prev) | 0.738 |
| hgb | None | Cascade A (tier-1 + tier-2-multi) | 0.703 |
| hgb | Kraken | Direct 4-class | 0.730 |
| hgb | Triton | Cascade B (tier-1 + K-vs-rest + prev) | 0.761 |
| hgb | Coral | Cascade B (tier-1 + K-vs-rest + prev) | 0.728 |

## 5. Consistency — does each cascade beat direct on **both** CV and holdout?

| Model | Cascade | Δ AUC (CV) | Δ AUC (holdout) | Beats direct on CV? | Holdout? | Both? |
|---|---|---:|---:|---|---|---|
| logreg | Cascade A (tier-1 + tier-2-multi) | -0.0133 | +0.0084 | no | YES | no |
| logreg | Cascade B (tier-1 + K-vs-rest + prev) | +0.0014 | +0.0600 | YES | YES | YES |
| logreg | Cascade C (tier-1 + K-vs-rest + T-vs-C) | -0.0186 | +0.0073 | no | YES | no |
| rforest | Cascade A (tier-1 + tier-2-multi) | +0.0039 | +0.0188 | YES | YES | YES |
| rforest | Cascade B (tier-1 + K-vs-rest + prev) | +0.0054 | +0.0298 | YES | YES | YES |
| rforest | Cascade C (tier-1 + K-vs-rest + T-vs-C) | -0.0003 | +0.0230 | no | YES | no |
| hgb | Cascade A (tier-1 + tier-2-multi) | -0.0222 | +0.0048 | no | YES | no |
| hgb | Cascade B (tier-1 + K-vs-rest + prev) | +0.0266 | +0.0403 | YES | YES | YES |
| hgb | Cascade C (tier-1 + K-vs-rest + T-vs-C) | -0.0022 | +0.0143 | no | YES | no |

## 6. Deployment recommendation

**Recommended: `hgb` × **Cascade B (tier-1 + K-vs-rest + prev)**.**

This (model × architecture) pair beats direct 4-class on *both* the 5-fold CV (Δ = +0.0266) and the temporal holdout (Δ = +0.0403). Among all consistent winners it has the highest mean Δ macro AUC across the two splits (+0.0334).

Other architectures that also beat direct on both splits (for at least one model):
- `logreg` × Cascade B (tier-1 + K-vs-rest + prev): CV Δ +0.0014, holdout Δ +0.0600
- `rforest` × Cascade A (tier-1 + tier-2-multi): CV Δ +0.0039, holdout Δ +0.0188
- `rforest` × Cascade B (tier-1 + K-vs-rest + prev): CV Δ +0.0054, holdout Δ +0.0298
