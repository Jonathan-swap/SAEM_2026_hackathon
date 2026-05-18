# Task 1 — direct 4-class vs cascade (tier-1 binary + tier-2 multiclass)

Two architectures, identical CV folds + identical temporal split. For each {logreg, rforest, hgb}, the fold preprocessor + train indices are shared, so the comparison is paired.

## CV (n=261 5-fold OOF)

| Model | Arch | log-loss | accuracy | macro ROC-AUC | macro PR-AUC | AUC None | AUC Kraken | AUC Triton | AUC Coral |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | direct | 1.9144 | 0.3678 | 0.6283 | 0.3426 | 0.617 | 0.587 | 0.642 | 0.667 |
| logreg | cascade | 1.8560 | 0.4100 | 0.6150 | 0.3274 | 0.624 | 0.533 | 0.657 | 0.645 |
| rforest | direct | 1.2336 | 0.4406 | 0.6803 | 0.3669 | 0.714 | 0.584 | 0.741 | 0.682 |
| rforest | cascade | 1.2142 | 0.4751 | 0.6759 | 0.3794 | 0.709 | 0.555 | 0.745 | 0.694 |
| hgb | direct | 1.9838 | 0.4521 | 0.6412 | 0.3585 | 0.665 | 0.589 | 0.671 | 0.639 |
| hgb | cascade | 1.9944 | 0.3946 | 0.6192 | 0.3357 | 0.669 | 0.540 | 0.676 | 0.591 |

### CV — cascade minus direct (Δ)

| Model | Δ macro AUC | Δ macro PR-AUC | Δ accuracy | Δ AUC None | Δ AUC Kraken | Δ AUC Triton | Δ AUC Coral |
|---|---:|---:|---:|---:|---:|---:|---:|
| logreg | -0.0133 | -0.0152 | +0.0421 | +0.0073 | -0.0537 | +0.0148 | -0.0218 |
| rforest | -0.0044 | +0.0125 | +0.0345 | -0.0055 | -0.0292 | +0.0049 | +0.0120 |
| hgb | -0.0220 | -0.0228 | -0.0575 | +0.0040 | -0.0489 | +0.0055 | -0.0488 |

## TEMPORAL (n=74 last-day holdout)

| Model | Arch | log-loss | accuracy | macro ROC-AUC | macro PR-AUC | AUC None | AUC Kraken | AUC Triton | AUC Coral |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | direct | 2.1635 | 0.3514 | 0.6003 | 0.3576 | 0.642 | 0.540 | 0.653 | 0.566 |
| logreg | cascade | 1.9490 | 0.4324 | 0.6087 | 0.3745 | 0.678 | 0.576 | 0.661 | 0.520 |
| rforest | direct | 1.2184 | 0.3919 | 0.6969 | 0.4080 | 0.714 | 0.621 | 0.750 | 0.703 |
| rforest | cascade | 1.2086 | 0.4595 | 0.7120 | 0.4488 | 0.724 | 0.656 | 0.755 | 0.713 |
| hgb | direct | 1.7037 | 0.4324 | 0.6850 | 0.4183 | 0.701 | 0.726 | 0.685 | 0.627 |
| hgb | cascade | 1.6162 | 0.4054 | 0.6931 | 0.4204 | 0.703 | 0.697 | 0.699 | 0.674 |

### TEMPORAL — cascade minus direct (Δ)

| Model | Δ macro AUC | Δ macro PR-AUC | Δ accuracy | Δ AUC None | Δ AUC Kraken | Δ AUC Triton | Δ AUC Coral |
|---|---:|---:|---:|---:|---:|---:|---:|
| logreg | +0.0084 | +0.0169 | +0.0811 | +0.0360 | +0.0358 | +0.0081 | -0.0462 |
| rforest | +0.0151 | +0.0408 | +0.0676 | +0.0100 | +0.0350 | +0.0054 | +0.0103 |
| hgb | +0.0082 | +0.0021 | -0.0270 | +0.0023 | -0.0292 | +0.0134 | +0.0462 |
