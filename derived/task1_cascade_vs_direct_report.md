# Task 1 — direct 4-class vs cascade (tier-1 binary + tier-2 multiclass)

Two architectures, identical CV folds + identical temporal split. For each {logreg, rforest, hgb}, the fold preprocessor + train indices are shared, so the comparison is paired.

## CV (n=261 5-fold OOF)

| Model | Arch | log-loss | accuracy | macro ROC-AUC | macro PR-AUC | AUC None | AUC Kraken | AUC Triton | AUC Coral |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | direct | 1.9144 | 0.3678 | 0.6283 | 0.3426 | 0.617 | 0.587 | 0.642 | 0.667 |
| logreg | cascade | 1.8560 | 0.4100 | 0.6150 | 0.3274 | 0.624 | 0.533 | 0.657 | 0.645 |
| rforest | direct | 1.2311 | 0.4521 | 0.6812 | 0.3709 | 0.712 | 0.600 | 0.736 | 0.676 |
| rforest | cascade | 1.2137 | 0.4904 | 0.6783 | 0.3774 | 0.707 | 0.564 | 0.747 | 0.696 |
| hgb | direct | 1.9830 | 0.4521 | 0.6413 | 0.3584 | 0.665 | 0.589 | 0.671 | 0.639 |
| hgb | cascade | 2.0013 | 0.3908 | 0.6178 | 0.3345 | 0.667 | 0.536 | 0.674 | 0.593 |

### CV — cascade minus direct (Δ)

| Model | Δ macro AUC | Δ macro PR-AUC | Δ accuracy | Δ AUC None | Δ AUC Kraken | Δ AUC Triton | Δ AUC Coral |
|---|---:|---:|---:|---:|---:|---:|---:|
| logreg | -0.0133 | -0.0152 | +0.0421 | +0.0073 | -0.0537 | +0.0148 | -0.0218 |
| rforest | -0.0029 | +0.0064 | +0.0383 | -0.0057 | -0.0365 | +0.0106 | +0.0201 |
| hgb | -0.0234 | -0.0239 | -0.0613 | +0.0022 | -0.0530 | +0.0032 | -0.0461 |

## TEMPORAL (n=74 last-day holdout)

| Model | Arch | log-loss | accuracy | macro ROC-AUC | macro PR-AUC | AUC None | AUC Kraken | AUC Triton | AUC Coral |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | direct | 2.1635 | 0.3514 | 0.6003 | 0.3576 | 0.642 | 0.540 | 0.653 | 0.566 |
| logreg | cascade | 1.9490 | 0.4324 | 0.6087 | 0.3745 | 0.678 | 0.576 | 0.661 | 0.520 |
| rforest | direct | 1.2074 | 0.4189 | 0.6979 | 0.4276 | 0.710 | 0.613 | 0.769 | 0.699 |
| rforest | cascade | 1.2071 | 0.4595 | 0.7094 | 0.4384 | 0.719 | 0.644 | 0.776 | 0.699 |
| hgb | direct | 1.7011 | 0.4324 | 0.6848 | 0.4180 | 0.701 | 0.727 | 0.684 | 0.627 |
| hgb | cascade | 1.6824 | 0.3784 | 0.6773 | 0.3991 | 0.686 | 0.671 | 0.684 | 0.668 |

### TEMPORAL — cascade minus direct (Δ)

| Model | Δ macro AUC | Δ macro PR-AUC | Δ accuracy | Δ AUC None | Δ AUC Kraken | Δ AUC Triton | Δ AUC Coral |
|---|---:|---:|---:|---:|---:|---:|---:|
| logreg | +0.0084 | +0.0169 | +0.0811 | +0.0360 | +0.0358 | +0.0081 | -0.0462 |
| rforest | +0.0115 | +0.0108 | +0.0405 | +0.0084 | +0.0308 | +0.0067 | +0.0000 |
| hgb | -0.0075 | -0.0189 | -0.0541 | -0.0153 | -0.0558 | +0.0000 | +0.0410 |
