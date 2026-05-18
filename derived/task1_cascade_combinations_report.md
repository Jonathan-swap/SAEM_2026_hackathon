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
| rforest | direct | 1.2347 | 0.4330 | 0.6770 | 0.3648 | 0.712 | 0.588 | 0.730 | 0.677 |
| rforest | casc_A_tier12 | 1.2118 | 0.4866 | 0.6809 | 0.3905 | 0.711 | 0.570 | 0.739 | 0.703 |
| rforest | casc_B_K_prev | 1.1971 | 0.4828 | 0.6825 | 0.3876 | 0.711 | 0.534 | 0.754 | 0.731 |
| rforest | casc_C_K_TC | 1.2089 | 0.4866 | 0.6767 | 0.3892 | 0.711 | 0.534 | 0.751 | 0.711 |
| hgb | direct | 1.9833 | 0.4521 | 0.6414 | 0.3586 | 0.665 | 0.590 | 0.671 | 0.639 |
| hgb | casc_A_tier12 | 1.9944 | 0.3946 | 0.6192 | 0.3357 | 0.669 | 0.540 | 0.676 | 0.591 |
| hgb | casc_B_K_prev | 1.5862 | 0.4406 | 0.6679 | 0.3758 | 0.669 | 0.581 | 0.734 | 0.687 |
| hgb | casc_C_K_TC | 1.7306 | 0.3985 | 0.6391 | 0.3503 | 0.669 | 0.581 | 0.702 | 0.604 |

### CV — cascades minus direct

| Model | Arch | Δ macro AUC | Δ macro PR-AUC | Δ acc | Δ AUC K | Δ AUC T | Δ AUC C |
|---|---|---:|---:|---:|---:|---:|---:|
| logreg | casc_A_tier12 | -0.0133 | -0.0152 | +0.0421 | -0.054 | +0.015 | -0.022 |
| logreg | casc_B_K_prev | +0.0014 | -0.0046 | +0.0421 | -0.072 | +0.056 | +0.014 |
| logreg | casc_C_K_TC | -0.0186 | -0.0163 | +0.0307 | -0.072 | +0.025 | -0.036 |
| rforest | casc_A_tier12 | +0.0039 | +0.0257 | +0.0536 | -0.018 | +0.009 | +0.026 |
| rforest | casc_B_K_prev | +0.0054 | +0.0228 | +0.0498 | -0.054 | +0.024 | +0.054 |
| rforest | casc_C_K_TC | -0.0003 | +0.0244 | +0.0536 | -0.054 | +0.021 | +0.033 |
| hgb | casc_A_tier12 | -0.0222 | -0.0229 | -0.0575 | -0.049 | +0.006 | -0.049 |
| hgb | casc_B_K_prev | +0.0266 | +0.0172 | -0.0115 | -0.008 | +0.063 | +0.048 |
| hgb | casc_C_K_TC | -0.0022 | -0.0082 | -0.0536 | -0.008 | +0.031 | -0.036 |

## TEMPORAL (n=74 last-day holdout)

| Model | Arch | log-loss | accuracy | macro AUC | macro PR-AUC | AUC None | AUC Kraken | AUC Triton | AUC Coral |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | direct | 2.1635 | 0.3514 | 0.6003 | 0.3576 | 0.642 | 0.540 | 0.653 | 0.566 |
| logreg | casc_A_tier12 | 1.9490 | 0.4324 | 0.6087 | 0.3745 | 0.678 | 0.576 | 0.661 | 0.520 |
| logreg | casc_B_K_prev | 1.5801 | 0.4865 | 0.6603 | 0.3913 | 0.678 | 0.559 | 0.741 | 0.663 |
| logreg | casc_C_K_TC | 1.7715 | 0.4730 | 0.6076 | 0.3681 | 0.678 | 0.559 | 0.648 | 0.545 |
| rforest | direct | 1.2185 | 0.4324 | 0.6894 | 0.4175 | 0.707 | 0.620 | 0.747 | 0.684 |
| rforest | casc_A_tier12 | 1.2094 | 0.4459 | 0.7082 | 0.4478 | 0.716 | 0.645 | 0.762 | 0.709 |
| rforest | casc_B_K_prev | 1.1707 | 0.4459 | 0.7192 | 0.4537 | 0.716 | 0.649 | 0.773 | 0.738 |
| rforest | casc_C_K_TC | 1.1817 | 0.4459 | 0.7124 | 0.4539 | 0.716 | 0.649 | 0.761 | 0.723 |
| hgb | direct | 1.7002 | 0.4324 | 0.6847 | 0.4186 | 0.697 | 0.730 | 0.688 | 0.624 |
| hgb | casc_A_tier12 | 1.6226 | 0.3784 | 0.6894 | 0.4179 | 0.703 | 0.699 | 0.712 | 0.643 |
| hgb | casc_B_K_prev | 1.3931 | 0.4324 | 0.7250 | 0.4467 | 0.703 | 0.708 | 0.761 | 0.728 |
| hgb | casc_C_K_TC | 1.5110 | 0.4054 | 0.6989 | 0.4304 | 0.703 | 0.708 | 0.716 | 0.668 |

### TEMPORAL — cascades minus direct

| Model | Arch | Δ macro AUC | Δ macro PR-AUC | Δ acc | Δ AUC K | Δ AUC T | Δ AUC C |
|---|---|---:|---:|---:|---:|---:|---:|
| logreg | casc_A_tier12 | +0.0084 | +0.0169 | +0.0811 | +0.036 | +0.008 | -0.046 |
| logreg | casc_B_K_prev | +0.0600 | +0.0337 | +0.1351 | +0.019 | +0.087 | +0.097 |
| logreg | casc_C_K_TC | +0.0073 | +0.0104 | +0.1216 | +0.019 | -0.005 | -0.021 |
| rforest | casc_A_tier12 | +0.0188 | +0.0304 | +0.0135 | +0.025 | +0.015 | +0.026 |
| rforest | casc_B_K_prev | +0.0298 | +0.0362 | +0.0135 | +0.029 | +0.026 | +0.055 |
| rforest | casc_C_K_TC | +0.0230 | +0.0365 | +0.0135 | +0.029 | +0.013 | +0.039 |
| hgb | casc_A_tier12 | +0.0048 | -0.0007 | -0.0541 | -0.031 | +0.024 | +0.019 |
| hgb | casc_B_K_prev | +0.0403 | +0.0281 | +0.0000 | -0.022 | +0.073 | +0.104 |
| hgb | casc_C_K_TC | +0.0143 | +0.0118 | -0.0270 | -0.022 | +0.028 | +0.044 |
