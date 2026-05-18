# Task 1 — Direct vs three cascade variants

Same CV folds + same temporal split for all four architectures. For each {logreg, rforest, hgb} the per-fold preprocessor and train indices are shared, so the comparison is paired.

Architectures:

- **D**: Direct 4-class — one classifier predicts None/K/T/C.
- **A**: tier-1 binary + tier-2 multiclass — `P(drug)` × `P(K/T/C | drug)`.
- **B**: tier-1 binary + Kraken-vs-rest + prevalence — `P(drug)` × `P(K | drug)`; non-K mass split T/C by training prevalence (no T-vs-C model).
- **C**: tier-1 binary + Kraken-vs-rest + Triton-vs-Coral — full hierarchical, three binary models.

## CV (n=261 5-fold OOF)

| Model | Arch | log-loss | accuracy | macro AUC | macro PR-AUC | AUC None | AUC Kraken | AUC Triton | AUC Coral |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | direct | 1.9144 | 0.3678 | 0.6283 | 0.3426 | 0.617 | 0.587 | 0.642 | 0.667 |
| logreg | casc_A_tier12 | 1.8560 | 0.4100 | 0.6150 | 0.3274 | 0.624 | 0.533 | 0.657 | 0.645 |
| logreg | casc_B_K_prev | 1.6051 | 0.4100 | 0.6298 | 0.3380 | 0.624 | 0.515 | 0.699 | 0.681 |
| logreg | casc_C_K_TC | 1.7752 | 0.3985 | 0.6098 | 0.3263 | 0.624 | 0.515 | 0.668 | 0.632 |
| rforest | direct | 1.2323 | 0.4444 | 0.6770 | 0.3655 | 0.708 | 0.595 | 0.736 | 0.670 |
| rforest | casc_A_tier12 | 1.2122 | 0.5057 | 0.6786 | 0.3854 | 0.710 | 0.561 | 0.744 | 0.699 |
| rforest | casc_B_K_prev | 1.2013 | 0.4636 | 0.6820 | 0.3849 | 0.710 | 0.535 | 0.759 | 0.723 |
| rforest | casc_C_K_TC | 1.2149 | 0.4751 | 0.6740 | 0.3837 | 0.710 | 0.535 | 0.749 | 0.701 |
| hgb | direct | 1.9838 | 0.4521 | 0.6413 | 0.3585 | 0.665 | 0.589 | 0.671 | 0.639 |
| hgb | casc_A_tier12 | 1.9944 | 0.3946 | 0.6192 | 0.3357 | 0.669 | 0.540 | 0.676 | 0.591 |
| hgb | casc_B_K_prev | 1.5862 | 0.4406 | 0.6679 | 0.3758 | 0.669 | 0.581 | 0.734 | 0.687 |
| hgb | casc_C_K_TC | 1.7306 | 0.3985 | 0.6391 | 0.3503 | 0.669 | 0.581 | 0.702 | 0.604 |

### CV — cascades minus direct

| Model | Arch | Δ macro AUC | Δ macro PR-AUC | Δ acc | Δ AUC K | Δ AUC T | Δ AUC C |
|---|---|---:|---:|---:|---:|---:|---:|
| logreg | casc_A_tier12 | -0.0133 | -0.0152 | +0.0421 | -0.054 | +0.015 | -0.022 |
| logreg | casc_B_K_prev | +0.0014 | -0.0046 | +0.0421 | -0.072 | +0.056 | +0.014 |
| logreg | casc_C_K_TC | -0.0186 | -0.0163 | +0.0307 | -0.072 | +0.025 | -0.036 |
| rforest | casc_A_tier12 | +0.0016 | +0.0198 | +0.0613 | -0.034 | +0.008 | +0.030 |
| rforest | casc_B_K_prev | +0.0050 | +0.0193 | +0.0192 | -0.060 | +0.024 | +0.054 |
| rforest | casc_C_K_TC | -0.0031 | +0.0182 | +0.0307 | -0.060 | +0.013 | +0.032 |
| hgb | casc_A_tier12 | -0.0221 | -0.0228 | -0.0575 | -0.049 | +0.006 | -0.049 |
| hgb | casc_B_K_prev | +0.0267 | +0.0173 | -0.0115 | -0.008 | +0.063 | +0.048 |
| hgb | casc_C_K_TC | -0.0021 | -0.0081 | -0.0536 | -0.008 | +0.031 | -0.036 |

## TEMPORAL (n=74 last-day holdout)

| Model | Arch | log-loss | accuracy | macro AUC | macro PR-AUC | AUC None | AUC Kraken | AUC Triton | AUC Coral |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | direct | 2.1635 | 0.3514 | 0.6003 | 0.3576 | 0.642 | 0.540 | 0.653 | 0.566 |
| logreg | casc_A_tier12 | 1.9490 | 0.4324 | 0.6087 | 0.3745 | 0.678 | 0.576 | 0.661 | 0.520 |
| logreg | casc_B_K_prev | 1.5801 | 0.4865 | 0.6603 | 0.3913 | 0.678 | 0.559 | 0.741 | 0.663 |
| logreg | casc_C_K_TC | 1.7715 | 0.4730 | 0.6076 | 0.3681 | 0.678 | 0.559 | 0.648 | 0.545 |
| rforest | direct | 1.2035 | 0.3919 | 0.7029 | 0.4286 | 0.715 | 0.629 | 0.751 | 0.716 |
| rforest | casc_A_tier12 | 1.2049 | 0.4459 | 0.7097 | 0.4428 | 0.713 | 0.662 | 0.757 | 0.706 |
| rforest | casc_B_K_prev | 1.1690 | 0.4730 | 0.7211 | 0.4406 | 0.713 | 0.659 | 0.777 | 0.735 |
| rforest | casc_C_K_TC | 1.1802 | 0.4730 | 0.7071 | 0.4435 | 0.713 | 0.659 | 0.757 | 0.699 |
| hgb | direct | 1.7037 | 0.4324 | 0.6850 | 0.4183 | 0.701 | 0.726 | 0.685 | 0.627 |
| hgb | casc_A_tier12 | 1.6569 | 0.3919 | 0.6803 | 0.4154 | 0.678 | 0.673 | 0.703 | 0.667 |
| hgb | casc_B_K_prev | 1.4339 | 0.4054 | 0.7156 | 0.4414 | 0.678 | 0.692 | 0.769 | 0.723 |
| hgb | casc_C_K_TC | 1.5515 | 0.3919 | 0.6871 | 0.4135 | 0.678 | 0.692 | 0.723 | 0.655 |

### TEMPORAL — cascades minus direct

| Model | Arch | Δ macro AUC | Δ macro PR-AUC | Δ acc | Δ AUC K | Δ AUC T | Δ AUC C |
|---|---|---:|---:|---:|---:|---:|---:|
| logreg | casc_A_tier12 | +0.0084 | +0.0169 | +0.0811 | +0.036 | +0.008 | -0.046 |
| logreg | casc_B_K_prev | +0.0600 | +0.0337 | +0.1351 | +0.019 | +0.087 | +0.097 |
| logreg | casc_C_K_TC | +0.0073 | +0.0104 | +0.1216 | +0.019 | -0.005 | -0.021 |
| rforest | casc_A_tier12 | +0.0067 | +0.0142 | +0.0541 | +0.033 | +0.005 | -0.010 |
| rforest | casc_B_K_prev | +0.0182 | +0.0120 | +0.0811 | +0.030 | +0.026 | +0.019 |
| rforest | casc_C_K_TC | +0.0042 | +0.0149 | +0.0811 | +0.030 | +0.005 | -0.017 |
| hgb | casc_A_tier12 | -0.0047 | -0.0029 | -0.0405 | -0.052 | +0.017 | +0.039 |
| hgb | casc_B_K_prev | +0.0307 | +0.0230 | -0.0270 | -0.033 | +0.083 | +0.096 |
| hgb | casc_C_K_TC | +0.0022 | -0.0049 | -0.0405 | -0.033 | +0.038 | +0.027 |
