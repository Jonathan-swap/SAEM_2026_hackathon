# Cascade-B threshold-tradeoff results

Single model family: **rforest** (best Cascade-B model per RUNBOOK §7h, holdout macro AUC 0.721). Thresholds picked on 5-fold OOF (n=261), applied unchanged to the temporal holdout (test = last day, n=74).

Triton prevalence (training mean across 5 OOF folds): **0.515** → stage-3 assigns Triton with that probability and Coral with **0.485**, per encounter.

Grid: τ_drug × τ_kraken ∈ [0.05, 0.95]² step 0.02 = 2116 cells.

## Picked points — OOF (where thresholds were chosen)

| Criterion | τ_drug | τ_kraken | Macro F1 | Accuracy | Min-class F1 | F1(N) | F1(K) | F1(T) | F1(C) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro_f1 | 0.57 | 0.45 | 0.430 | 0.502 | 0.265 | 0.689 | 0.265 | 0.416 | 0.350 |
| accuracy | 0.57 | 0.45 | 0.430 | 0.502 | 0.265 | 0.689 | 0.265 | 0.416 | 0.350 |
| min_class_f1 | 0.33 | 0.51 | 0.371 | 0.368 | 0.343 | 0.351 | 0.343 | 0.418 | 0.371 |

## Picked points — temporal holdout (deployment metric)

| Criterion | τ_drug | τ_kraken | Macro F1 | Accuracy | Min-class F1 | F1(N) | F1(K) | F1(T) | F1(C) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro_f1 | 0.57 | 0.45 | 0.393 | 0.459 | 0.200 | 0.638 | 0.200 | 0.333 | 0.400 |
| accuracy | 0.57 | 0.45 | 0.393 | 0.459 | 0.200 | 0.638 | 0.200 | 0.333 | 0.400 |
| min_class_f1 | 0.33 | 0.51 | 0.384 | 0.392 | 0.286 | 0.449 | 0.372 | 0.286 | 0.429 |

### Confusion matrices — holdout

**macro_f1** (τ_drug=0.57, τ_kraken=0.45) — accuracy 0.459, macro F1 0.393

| true \ pred | None | Kraken | Triton | Coral |
|---|---:|---:|---:|---:|
| None | 22 | 3 | 2 | 2 |
| Kraken Candy | 13 | 3 | 5 | 3 |
| Triton Tabs | 2 | 0 | 4 | 6 |
| Coral Dust | 3 | 0 | 1 | 5 |

**accuracy** (τ_drug=0.57, τ_kraken=0.45) — accuracy 0.459, macro F1 0.393

| true \ pred | None | Kraken | Triton | Coral |
|---|---:|---:|---:|---:|
| None | 22 | 3 | 2 | 2 |
| Kraken Candy | 13 | 3 | 5 | 3 |
| Triton Tabs | 2 | 0 | 4 | 6 |
| Coral Dust | 3 | 0 | 1 | 5 |

**min_class_f1** (τ_drug=0.33, τ_kraken=0.51) — accuracy 0.392, macro F1 0.384

| true \ pred | None | Kraken | Triton | Coral |
|---|---:|---:|---:|---:|
| None | 11 | 10 | 5 | 3 |
| Kraken Candy | 6 | 8 | 6 | 4 |
| Triton Tabs | 2 | 0 | 4 | 6 |
| Coral Dust | 1 | 1 | 1 | 6 |

## Files

- `derived/task1_cascade_b_threshold_grid.csv` — every (τ_drug, τ_kraken) cell with metrics
- `derived/task1_cascade_b_threshold_picked.csv` — three picked pairs evaluated on OOF and holdout
- `derived/task1_cascade_b_threshold_labels.csv` — per-encounter output: p_drug, p_kraken_given_drug, triton_prev, the four soft probabilities (p_none/p_kraken/p_triton/p_coral), true label, and final candidate label (picked by macro F1).

### Stage-3 behavior

Cascade-B has no T-vs-C model — the §7g ceiling. Instead of collapsing every non-Kraken drug-positive encounter to a single majority class, stage 3 now does a per-encounter Bernoulli draw against the training prevalence (0.515 Triton among non-K drug+). The draw is deterministic (md5 hash of encounter_id → uniform → compare to prevalence), so:

- the marginal Triton/Coral output distribution matches the training prevalence (~51.5% Triton, ~48.5% Coral among non-K drug+ predictions);
- the same encounter always gets the same T/C label across re-runs and across grid cells;
- no RNG state is needed during the grid search.

T vs C discrimination still cannot exceed chance at triage. The prevalence-Bernoulli simply preserves the marginal class balance instead of zeroing one class. For real T-vs-C discrimination, move to the 4-hour-horizon Task-2 features.
