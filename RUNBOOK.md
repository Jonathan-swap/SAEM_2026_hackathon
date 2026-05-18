# Runbook — SAEM Hackathon Pipeline

End-to-end execution from raw xlsx → trained, evaluated Task-1 /
Task-2 / Task-3 models with both **5-fold cross-validation** (§7a /
§8a) and **temporal holdout** (§7b / §8b, train on early days, test
on last day — the deployment-relevant metric). Plus v6 feature
audit, task-aligned clustering. Every command runs from the repo
root (`SAEM_2026_hackathon/`). PowerShell on Windows; substitute
`.venv/bin/python` on macOS/Linux.

## Outcomes vs features (read this first)

Two **supervised targets** live in a single canonical file:

```
derived/outcomes.csv
  encounter_id, ground_truth_drug, ground_truth_drug_name, encounter_disposition_label
```

| Task | Target column | Class set | Source sheet |
|---|---|---|---|
| **Task 1** | `ground_truth_drug` (int 0-3) / `ground_truth_drug_name` | None / Kraken / Triton / Coral | `Task1_Two_Tier_Input_Data.csv` |
| **Task 2** | `encounter_disposition_label` | Discharge / Floor / ICU | xlsx `Disposition` sheet |

**Important to internalise about Task 2**: the target is
**disposition** (where the patient ends up — Discharge / Floor /
ICU), NOT drug class. The drug class only acts as a *cohort filter*
in the default mode (`--cohort drug-positive` → train only on the
157 patients with `ground_truth_drug != 0`). The prediction over
that cohort is still Discharge vs Floor vs ICU.

The two feature tables (`features_triage.csv`, `features_fourh.csv`)
contain **only features** — no outcome columns. The split is
structural: a script that doesn't read `outcomes.csv` literally
cannot see the target.

---

## 0. Prerequisites

- Python 3.10+ (tested on 3.13)
- ~500 MB free disk for `derived/` artifacts
- Claude Code harness — only required for the 10-agent labeling step (§5b)

---

## 1. Set up the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins pandas, numpy, scikit-learn, scipy, openpyxl,
joblib, matplotlib, and `mord` (ordinal regression). `setup.sh`
does the same on bash.

Verify:

```powershell
.venv\Scripts\python.exe -c "import pandas, numpy, sklearn, scipy, openpyxl; print('ok')"
```

---

## 2. Drop in the raw data

Place organizers' files at:

```
data/Hackathon_Data_Release_1_SHARE.xlsx       # raw xlsx — required
data/Task1_Two_Tier_Input_Data.csv             # manual ground truth — required
data/toxidrome_report_v6.pdf                   # reference report — recommended
```

All gitignored. The xlsx must contain three sheets: `Triage_Data`,
`Four_Hour_Data`, `Disposition`. Encounter counts should match (261
in the released dataset).

---

## 3. Audit the 4-hour cap (optional, recommended)

Before training, verify the source xlsx's timeseries are bounded:

```powershell
.venv\Scripts\python.exe src\eda\check_time_horizons.py
```

Pass criterion: `OK: every timeseries record has minute <= 240.`
If any record exceeds 240, the `cap_4h()` guard in
`extract_time_features.py` silently drops it — but you should
investigate, because the source data shouldn't drift.

---

## 4. Feature pipeline

Run the orchestrator with feature-only mode:

```powershell
.venv\Scripts\python.exe run_pipeline.py --skip-agents --features-only
```

Steps executed (in order):

| # | Script | Output |
|---|---|---|
| 1 | `src/features/extract_narratives.py`    | `derived/narratives_triage.jsonl` (triage-only) + `derived/narratives_fourh.jsonl` (triage + 4h narratives); leakage sentinel asserts the triage file is 4h-free |
| 2 | `src/features/extract_structured.py`    | `derived/features_triage.csv` + `features_fourh.csv` (outcomes excluded) |
| 3 | `src/features/extract_time_features.py` | adds Groups A–G time features (capped at min ≤ 240) |
| 4 | `src/features/extract_differentials.py` | adds triage↔4h diffs to `features_fourh.csv` |
| 5 | `src/features/extract_note_features.py` | adds parsed onset minutes + festival flags (triage notes) |
| 6 | `src/features/extract_note_4h_features.py` | adds HPI/MDM word counts + `mdm_severity_tier_4h` + `hpi_has_severe_4h` (4h-only; never lands in `features_triage.csv`) |
| 7 | `src/features/extract_v6_features.py`   | adds PE binaries + peak-lab thresholds + triage keywords (see §4.1 for details) |
| 8 | `src/labels/load_ground_truth.py`       | `derived/ground_truth.csv` (drug class) |
| 9 | `src/labels/build_outcomes.py`          | `derived/outcomes.csv` (drug class + disposition) |

Pass criterion: each step logs `OK` and the time-features script ends
with `OK: no leakage; triage features contain only arrival-time +
minute-0 signals.`

After step 7, **`derived/outcomes.csv`** is the canonical source for
BOTH supervised labels: drug class (Task 1) and disposition (Task 2).
The feature tables (`features_triage.csv`, `features_fourh.csv`)
**do NOT contain any outcome columns** — every Task-1 and Task-2
trainer pulls labels from `outcomes.csv` exclusively.

### 4.1. v6 features (integrated into the pipeline)

As of the latest pipeline, `extract_v6_features.py` runs automatically
as step 6 of §4. You can also invoke it standalone for development:

```powershell
.venv\Scripts\python.exe src\features\extract_v6_features.py
```

Adds three feature families (research: [`research/01_v6_features_and_prompts.md`](research/01_v6_features_and_prompts.md)):

- **PE binaries** (Task 2 only) — 14 finding tokens parsed from
  `physical_exam_pertinent_positives` + the two high-PPV combos
  (Kraken: diaphoretic+tachycardic; Triton: reduced_tracking+slow_responses)
- **Peak-lab + peak-vital threshold flags** (Task 2 only) — v6
  rule-1 anchors (peak_lactate>5, peak_CPK>1000, etc.) plus
  `all_peak_labs_normal` and `kraken_severity_anchor`
- **Triage text keywords + densities** (both tables) —
  `triage_chief_agitation`, `note_arousal_density`,
  `note_perceptual_density`, and seven triage threshold flags
  (AG>20, AG<12, pH>7.35, HR>120, temp>38, glucose>140,
  sympathomimetic combo)

Pass criterion: each feature has a positive count printed and the
leakage sentinel still passes when `cleanup_features.py` runs.

---

## 5. Labels — choose ONE

### 5a. Manual ground truth only (fastest, recommended)

`derived/ground_truth.csv` was already written in §4. Skip 5b unless
you also want the 10-agent drug-probability features for Task 2.

### 5b. 10-agent LLM consensus (v6-aligned prompts)

Requires the Claude Code harness — spawns 10 subagents in parallel,
each reading `derived/narratives_fourh.jsonl` (the full triage + 4h
record; the triage-only `narratives_triage.jsonl` is reserved for
triage-horizon NLP and would starve the agents of the 4h narrative
blocks they need to derive case-fixed toxidrome labels) under a
different reasoning emphasis. **Prompts: [`src/labels/agents/PROMPTS.md`](src/labels/agents/PROMPTS.md).**

The prompts open with a **v6 PRE-READ block** that every agent
consumes: corrected fixed drug→toxidrome mapping (Triton is *not*
classical bradycardic sedative — it's depressant + cardiac awareness +
tinnitus), the 14-rule discriminator hierarchy verbatim, and three
methodology warnings (MDM boilerplate in 44% of cases, pupils absent
from dataset, severity-Kraken correlation is real). Two agents were
repurposed: Agent 4 = MDM minus boilerplate; Agent 7 = autonomic-only
decision tree (no pupils).

Each subagent writes `derived/probs_<N>.csv`. After all 10 complete:

```powershell
.venv\Scripts\python.exe src\labels\merge_probabilities.py
```

Output: `derived/probs_avg.csv` (averaged 4-class probability per
encounter). Without this file, Task 2's "WITH drug-probs" variant
cannot run.

---

## 6. EDA + final cleanup

```powershell
.venv\Scripts\python.exe src\eda\eda_descriptive.py
.venv\Scripts\python.exe src\features\cleanup_features.py
```

`eda_descriptive.py` writes:
- `derived/eda_descriptive_report.md`
- `derived/exploratory_features.csv` (the `cand_*` candidate features)
- `derived/eda_plots/*.png`

`cleanup_features.py`:
- Drops zero-variance + ≥99%-constant columns from both feature tables
- Merges `cand_*` features into both
- Runs the leakage sentinel — forbids `vts_/lts_/itv_/xmod_/stab_/arc_/diff_/*_4h`
  prefixes, plus `encounter_disposition_label` and `arrival_same_day_volume`
  in `features_triage.csv`

Pass criterion: `OK: leakage sentinel passed on features_triage.csv`.

### 6.1. v6 feature audit (optional)

After running §4.1 (v6 features) and §6 (cleanup), audit which v6
features actually pull weight:

```powershell
.venv\Scripts\python.exe src\eda\eda_v6_features.py
```

Outputs:
- Stdout — top-15 features by MI vs Task-1 target, top-15 vs Task-2
  disposition target, and class-conditional fraction-positive for
  every binary feature.
- `derived/v6_feature_audit.md` — full markdown report.

Pass criterion: the class-conditional fractions for `pe_diaphoretic`,
`pe_mild_tremor`, etc. should match the v6 report's stated percentages
(29% / 10% / 4% Kraken / Triton / Coral for `pe_diaphoretic`).
This is your reproducibility check.

### 6.2. Task-2 leakage verification (optional, recommended)

Before trusting Task-2's high AUC, verify the disposition outcome
isn't leaking back in as a feature:

```powershell
.venv\Scripts\python.exe src\eda\check_task2_leakage.py
```

Four checks run automatically:

1. `encounter_disposition_label` IS in `features_fourh.csv` (correct
   — the trainer reads it as target) but **is removed by
   `load_data()` before reaching X**.
2. No alias columns (no `discharge`/`floor`/`icu`/`disposition`/
   `admit` tokens anywhere in `X.columns`).
3. `ground_truth_drug*` (the drug-class label) is also removed.
4. MI scan: every feature's MI vs the disposition target is
   computed. The reference MI(target ↔ target) gives a ceiling
   (~0.69 for the 3-class drug-positive cohort). A leak threshold
   of `0.85 × reference` flags features that are essentially the
   target by another name.

Pass criterion: `OK Task-2 training cannot see the disposition
outcome` (exit code 0). Current state: top feature MI = 0.449
(65% of ceiling) — strong legitimate clinical predictors like
`vts_heart_rate_mean`, `vts_temperature_c_mean`, `vts_gcs_mean`.
**Zero features cross the leak threshold.**

---

## 7. Train + evaluate Task 1 (drug ID at triage)

**What is predicted**: `ground_truth_drug` ∈ {None, Kraken Candy,
Triton Tabs, Coral Dust}. Sourced from `outcomes.csv` (which in turn
sources from `Task1_Two_Tier_Input_Data.csv`).

### Seven architectures (all retained; §7h compares 4 of them)

| § | Architecture | Predicts | Cohort | Train script |
|---|---|---|---|---|
| **7a / 7b** | Direct 4-class | None / Kraken / Triton / Coral | all 261 | `src/task1_drug_id/train_baseline.py` |
| **7c** | Tier-1 binary | drug / no-drug | all 261 | `src/task1_drug_id/train_binary.py` |
| **7d** | Tier-2 multiclass | Kraken / Triton / Coral (given drug+) | 157 drug-positive | `src/task1_drug_id/train_tier2.py` |
| **7e** | **Cascade** (tier-1 × tier-2 → 4-class) | None / Kraken / Triton / Coral | all 261 (via cascade) | `src/task1_drug_id/compare_cascade.py` |
| **7f** | Kraken vs (Triton + Coral) | Kraken / other-drug | 157 drug-positive | `src/task1_drug_id/train_kraken_binary.py` |
| **7g** | Triton vs Coral | Triton / Coral | 99 non-Kraken drug-positive | `src/task1_drug_id/train_triton_coral.py` |
| **7h** | Cascade variants A/B/C vs direct | None / K / T / C (via three different cascade compositions) | all 261 | `src/task1_drug_id/compare_cascades.py` |

### Headline results

**Temporal holdout (last day = test) — 4-class architectures (n=74)
side-by-side:** (last refreshed 2026-05-18 with the upstream task2
features ported in: `additional_labs_drawn_4h` + 4 HPI/MDM 4h-only
features — see §4. Same StratifiedKFold(n_splits=5, shuffle=True,
random_state=42) and temporal holdout split as prior runs.)

| Architecture | Best model | macro ROC-AUC | macro PR-AUC | accuracy |
|---|---|---:|---:|---:|
| Direct 4-class | rforest | 0.703 | 0.429 | 0.392 |
| Cascade-A (tier-1 + tier-2-multi) | rforest | 0.710 | 0.443 | 0.446 |
| Cascade-C (tier-1 + K-vs-rest + T-vs-C model) | rforest | 0.707 | 0.444 | 0.473 |
| **Cascade-B (tier-1 + K-vs-rest + prev)** | **rforest** | **0.721** | **0.441** | **0.473** |

**Cascade-B is still the deployment champion** — wins 18 of 24
split × model × metric cells (vs Direct 3/24, Cascade-A 3/24,
Cascade-C 2/24). The new HPI/MDM 4h features narrowed the lead and
flipped the best Cascade-B model from hgb to rforest on the temporal
holdout (rforest 0.721 vs hgb 0.716). Holdout deltas over direct:
logreg +0.060, rforest +0.018, hgb +0.031. See §7h for the
comparison details and `research/05_cascade_variants.md`.

**Binary sub-problems (different cohorts and class sets, not directly
comparable to the 4-class numbers above):**

| Sub-problem | Best model | n test | ROC-AUC | PR-AUC | accuracy |
|---|---|---:|---:|---:|---:|
| Tier-1: drug vs no-drug | rforest | 74 | 0.723 | 0.791 | 0.649 |
| Tier-2 multiclass: K/T/C (3-class) | hgb | 45 | 0.678 macro | 0.536 macro | 0.556 |
| **Kraken vs rest** | **hgb** | 45 | **0.825** | **0.877** | **0.800** |
| Triton vs Coral | rforest | 21 | **0.426** ⚠ | 0.611 | 0.476 |

**5-fold CV (4-class architectures, n=261):**

| Architecture | Best model | macro ROC-AUC | macro PR-AUC | accuracy |
|---|---|---:|---:|---:|
| Direct 4-class | rforest | 0.677 | 0.366 | 0.444 |
| Cascade-A (tier-1 + tier-2-multi) | rforest | 0.679 | 0.385 | 0.506 |
| Cascade-C (tier-1 + K-vs-rest + T-vs-C model) | rforest | 0.674 | 0.384 | 0.475 |
| **Cascade-B (tier-1 + K-vs-rest + prev)** | rforest | **0.682** | **0.385** | **0.464** |

**5-fold CV (binary sub-problems):**

| Sub-problem | Best model | n | ROC-AUC | PR-AUC | accuracy |
|---|---|---:|---:|---:|---:|
| Tier-1: drug vs no-drug | rforest | 261 | 0.736 | 0.819 | 0.686 |
| Tier-2 multiclass: K/T/C | rforest | 157 | 0.715 macro | 0.598 macro | 0.522 |
| **Kraken vs rest** | **hgb** | 157 | **0.806** | **0.769** | **0.751** |
| Triton vs Coral | rforest | 99 | **0.517** ⚠ | 0.599 | 0.515 |

Notes on comparing these numbers:
- **Direct vs Cascade** are directly comparable — both predict all
  four classes on all 261 patients. **Cascade is the deployment
  recommendation for the full 4-class problem** (see §7e).
- **Tier-1 binary** is a 2-class problem (60% prevalence) on all
  261 patients — screening test for "is there ANY drug?".
- **Tier-2 multiclass** runs on 157 drug-positive patients (3
  classes, ~37/32/31% K/T/C). Feeds the cascade and the Task-3
  calculator's "most likely drug" card.
- **Kraken vs rest** is a 2-class problem (37% Kraken) on the
  drug-positive cohort. The strongest single-decision model in
  the lot — Kraken's sympathomimetic triage signature is uniquely
  visible at minute 0, so binarising the question instead of
  asking K-vs-T-vs-C lifts AUC from ~0.71 to ~0.83 on the
  holdout. Clinically meaningful because Kraken drives a
  completely different ED workup (rhabdomyolysis labs +
  aggressive fluids + cooling) from Triton/Coral (supportive).
- **Triton vs Coral** is a 2-class problem (52% Triton) on the
  non-Kraken drug-positive cohort (n=99). **Documents the
  ceiling**: holdout ROC-AUC 0.32-0.39 across all models (worse
  than chance). Triton's and Coral's defining tokens live in
  HPI/MDM (post-triage); their triage vitals/labs are
  near-identical. Any architecture that has to separate Triton
  from Coral at triage inherits this ceiling. The proper fix is
  to move that decision to Task-2's 4-hour-horizon feature set,
  where the PE binaries finally become visible.

### Common context

Two evaluation modes are available, both leakage-clean and using
the manual ground truth + v6 features:

### 7a. 5-fold stratified cross-validation (random folds, all 261 patients)

```powershell
.venv\Scripts\python.exe src\task1_drug_id\train_baseline.py
```

Three models (logreg, rforest, hgb). Artifacts:
`derived/task1_baseline_summary.csv` + `task1_oof_predictions.csv`.

| Model | log-loss | accuracy | macro ROC-AUC | macro PR-AUC |
|-------|---------:|---------:|--------------:|-------------:|
| logreg | 1.91 | 0.37 | 0.625 | 0.384 |
| **rforest** | **1.23** | **0.44** | **0.695** | **0.456** |
| hgb | 1.98 | 0.45 | 0.642 | 0.406 |

Majority-class baseline accuracy = 0.40. Class prevalence:
None 0.40 / Kraken 0.22 / Triton 0.20 / Coral 0.18.

**Per-class metrics for rforest** (OVR = one-vs-rest):

| Class | Prevalence | ROC-AUC | PR-AUC | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| None | 0.40 | 0.732 | 0.616 | 0.213 | **+0.113** |
| Kraken | 0.22 | 0.603 | 0.339 | 0.173 | −0.005 |
| Triton | 0.20 | 0.743 | 0.457 | 0.143 | **+0.085** |
| Coral | 0.18 | 0.703 | 0.411 | 0.143 | **+0.053** |

Task-1 BSS is near zero or slightly negative for the drug classes
— rforest barely beats predicting the marginal prevalence. The
0.70 macro AUC is real but small: triage-only text fundamentally
lacks Triton/Coral signal (their defining tokens —
"ringing in ears", "time distortion" — appear post-triage only).

### 7b. Temporal holdout (train on early days, test on last day) — **deployment-relevant**

Split: train on every encounter with `encounter_arrival_date < last_day`;
test on the last day's encounters only. With the 5-day release this is
**187 train / 74 test**. Mirrors the Phase-2 deployment scenario where
the model trained on prior days must predict on a fresh wave of
arrivals.

```powershell
.venv\Scripts\python.exe src\eval_temporal.py    # runs Task 1 + Task 2
```

Artifacts (Task 1): `derived/task1_temporal_summary.csv`
+ `task1_temporal_predictions.csv` + section 1 of
`temporal_holdout_report.md`.

| Model | log-loss | accuracy | macro ROC-AUC | macro PR-AUC |
|-------|---------:|---------:|--------------:|-------------:|
| logreg | 2.16 | 0.35 | 0.600 | 0.358 |
| **rforest** | **1.21** | **0.39** | **0.696** | **0.429** |
| hgb | 1.70 | 0.43 | 0.685 | 0.419 |

**Per-class metrics for rforest (holdout):**

| Class | Prevalence | ROC-AUC | PR-AUC | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| None | 0.39 | 0.721 | 0.646 | 0.206 | **+0.134** |
| Kraken | 0.32 | 0.631 | 0.450 | 0.219 | +0.000 |
| Triton | 0.16 | 0.753 | 0.335 | 0.127 | **+0.068** |
| Coral | 0.12 | 0.679 | 0.283 | 0.112 | −0.053 |

Holdout AUC ≈ CV AUC — the Task-1 ceiling is data-bound. See
`research/03_v6_feature_evaluation.md`.

### 7c. Binary tier-1 classifier (drug vs no-drug)

Mirrors the tier-1 question from the organizers' file
`Task1_Two_Tier_Input_Data.csv`: is this patient on ANY festival
drug? Collapses Kraken+Triton+Coral into a single positive class.

```powershell
.venv\Scripts\python.exe src\task1_drug_id\train_binary.py
```

Runs BOTH 5-fold CV and temporal holdout in one go. Artifacts:
- `derived/task1_binary_baseline_summary.csv` (CV)
- `derived/task1_binary_oof_predictions.csv` (CV OOF)
- `derived/task1_binary_temporal_summary.csv` (holdout)
- `derived/task1_binary_temporal_predictions.csv` (holdout preds)

Class prevalence: 60% drug-positive / 40% no-drug.

**5-fold CV (n=261):**

| Model | log-loss | accuracy | ROC-AUC | PR-AUC | Sens | Spec |
|-------|---------:|---------:|--------:|-------:|-----:|-----:|
| logreg | 0.84 | 0.64 | 0.65 | 0.72 | 0.66 | 0.62 |
| **rforest** | **0.60** | **0.69** | **0.74** | **0.82** | 0.71 | 0.66 |
| hgb | 0.95 | 0.63 | 0.66 | 0.76 | 0.69 | 0.55 |

**Temporal holdout (test = last day, n=74):**

| Model | ROC-AUC | PR-AUC | Sens | Spec | PPV | NPV | BSS |
|-------|--------:|-------:|-----:|-----:|----:|----:|----:|
| logreg | 0.678 | 0.731 | 0.67 | 0.59 | 0.71 | 0.53 | −0.025 |
| **rforest** | **0.723** | **0.791** | 0.62 | **0.69** | **0.76** | 0.54 | **+0.120** |
| hgb | 0.703 | 0.789 | 0.71 | 0.48 | 0.68 | 0.52 | −0.057 |

Test-set prevalence 0.61. The binary task is materially easier than
the 4-class drug-ID — 0.72 ROC-AUC vs 0.70 — because Kraken's strong
sympathomimetic triage signature is shared by the easier-to-detect
"drug-positive" superset.

### 7d. Tier-2 classifier — which drug? (Kraken / Triton / Coral)

The natural pair to §7c. Given a patient is drug-positive, classify
which of the three festival drugs. Cohort filter:
`outcomes.csv :: ground_truth_drug != 0` → 157 patients.

```powershell
.venv\Scripts\python.exe src\task1_drug_id\train_tier2.py
```

Runs both 5-fold CV and temporal holdout. Artifacts:
- `derived/task1_tier2_baseline_summary.csv` (CV)
- `derived/task1_tier2_oof_predictions.csv`
- `derived/task1_tier2_temporal_summary.csv`
- `derived/task1_tier2_temporal_predictions.csv`

Class prevalence on cohort: Kraken 0.37 / Triton 0.32 / Coral 0.31.

**5-fold CV (n=157):**

| Model | log-loss | accuracy | macro ROC-AUC | macro PR-AUC |
|-------|---------:|---------:|--------------:|-------------:|
| logreg | 1.65 | 0.50 | 0.641 | 0.530 |
| **rforest** | **0.96** | **0.53** | **0.709** | **0.591** |
| hgb | 1.58 | 0.52 | 0.667 | 0.539 |

**Per-class metrics (rforest, 5-fold CV):**

| Class | Prevalence | ROC-AUC | PR-AUC | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| Kraken | 0.37 | 0.809 | 0.769 | 0.170 | **+0.270** |
| Triton | 0.32 | 0.714 | 0.588 | 0.193 | **+0.120** |
| Coral | 0.31 | 0.605 | 0.417 | 0.208 | +0.018 |

**Temporal holdout (test = last day, n=45):**

| Model | log-loss | accuracy | macro ROC-AUC | macro PR-AUC |
|-------|---------:|---------:|--------------:|-------------:|
| logreg | 2.03 | 0.42 | 0.594 | 0.480 |
| rforest | 1.00 | 0.51 | 0.666 | 0.505 |
| **hgb** | 1.38 | **0.56** | **0.678** | **0.536** |

**Per-class metrics (hgb, holdout):**

| Class | Prevalence | ROC-AUC | PR-AUC | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| Kraken | 0.53 | **0.802** | **0.843** | 0.199 | **+0.200** |
| Triton | 0.27 | 0.687 | 0.525 | 0.239 | −0.224 |
| Coral | 0.20 | 0.546 | 0.240 | 0.243 | −0.518 |

The Kraken column drives most of the lift. Triton and Coral's
defining tokens ("ringing in ears", "time distortion sensation")
appear in HPI/MDM only — not in `triage_brief_note` — so any
triage-only classifier hits a ceiling for those two. Pairing this
tier-2 with a Task-2 4h-horizon model gives much better discrimination
because the 4h PE findings (reduced_tracking, slow_responses for
Triton; unsteady_gait for Coral) finally become visible.

### 7e. Direct 4-class vs cascade (tier-1 + tier-2)

Apples-to-apples comparison of the two architectures for Task 1:
direct 4-class (§7a/7b) versus the cascade (§7c tier-1 → §7d tier-2,
combined as `P(None)=1−P(drug)`, `P(K)=P(drug)·P(K|drug)`, etc.).

```powershell
.venv\Scripts\python.exe src\task1_drug_id\compare_cascade.py
```

Identical CV folds + identical temporal split. Outputs:
- `derived/task1_cascade_vs_direct_summary.csv`
- `derived/task1_cascade_vs_direct_report.md`
- `research/04_cascade_vs_direct.md` (analysis + recommendation)

**Temporal holdout (the deployment metric, n=74, rforest):**

| Architecture | macro ROC-AUC | macro PR-AUC | accuracy | log-loss |
|---|---:|---:|---:|---:|
| Direct 4-class | 0.697 | 0.408 | 0.392 | 1.218 |
| **Cascade** | **0.712** | **0.449** | **0.460** | **1.209** |
| Δ (cascade − direct) | **+0.015** | **+0.041** | **+0.068** | −0.009 |

Cascade wins macro AUC on **all 3** models on the holdout (logreg
+0.008, rforest +0.015, hgb +0.008) and macro PR-AUC on 2/3.
Per-class: cascade adds **+3-4 points to Kraken AUC** for logreg
and rforest. CV is essentially tied (small AUC losses within
fold noise, small accuracy gains).

**Recommendation**: ship the cascade for deployment. It also
produces tier-1 and tier-2 probabilities separately, which is what
the Task-3 triage calculator's two-card UI already wants.

Details: [`research/04_cascade_vs_direct.md`](research/04_cascade_vs_direct.md).

### 7f. Kraken vs (Triton + Coral) binary classifier

Sub-problem on the drug-positive cohort: is this patient on
**Kraken** (sympathomimetic, severity-enriched) or on one of the
other two drugs (Triton depressant / Coral hallucinogen)? Clinically
the highest-stakes single decision — Kraken triggers a completely
different ED workup (rhabdomyolysis labs, IV fluids, cooling) from
the supportive-only Triton/Coral pathway.

```powershell
.venv\Scripts\python.exe src\task1_drug_id\train_kraken_binary.py
```

Both 5-fold CV and temporal holdout in one go. Artifacts:
- `derived/task1_kraken_binary_baseline_summary.csv` (CV)
- `derived/task1_kraken_binary_oof_predictions.csv`
- `derived/task1_kraken_binary_temporal_summary.csv`
- `derived/task1_kraken_binary_temporal_predictions.csv`

Class prevalence on cohort: 37% Kraken / 63% other-drug.

**5-fold CV (n=157):**

| Model | log-loss | accuracy | ROC-AUC | PR-AUC | Sens | Spec | BSS |
|-------|---------:|---------:|--------:|-------:|-----:|-----:|----:|
| logreg | 0.71 | 0.71 | 0.747 | 0.746 | 0.67 | 0.74 | +0.099 |
| rforest | 0.54 | 0.76 | 0.800 | 0.767 | 0.62 | 0.84 | +0.241 |
| **hgb** | **0.61** | **0.75** | **0.806** | **0.769** | 0.65 | 0.81 | **+0.234** |

**Temporal holdout (test = last day, n=45):**

| Model | log-loss | accuracy | ROC-AUC | PR-AUC | Sens | Spec | PPV | NPV | BSS |
|-------|---------:|---------:|--------:|-------:|-----:|-----:|----:|----:|----:|
| logreg | 1.09 | 0.67 | 0.633 | 0.769 | 0.50 | 0.86 | 0.80 | 0.60 | −0.113 |
| rforest | 0.60 | 0.69 | 0.778 | 0.833 | 0.50 | 0.90 | 0.86 | 0.61 | +0.178 |
| **hgb** | **0.65** | **0.80** | **0.825** | **0.877** | 0.75 | 0.86 | 0.86 | 0.75 | **+0.288** |

**Why this works so well**: Kraken's defining v6 signals are *triage-visible* —
high HR, high RR, elevated temperature, wide anion gap, agitation
in the brief note. Triton and Coral lack equivalent triage-time
signatures (their defining tokens — "ringing in ears", "time
distortion" — live in HPI/MDM, post-triage). Binarising the
problem as "Kraken vs everything-else-drug-related" gets all the
signal the data has to offer.

Compare with tier-2 multiclass (§7d, hgb holdout macro AUC 0.68)
— forcing the model to also separate Triton from Coral introduces
mostly-noise decisions that cap macro AUC. If the deployment
question is "should this patient get rhabdo workup?", this
Kraken-vs-rest model is the cleanest answer (PPV 0.86 at
prevalence 0.53).

### 7g. Triton vs Coral binary classifier — **the ceiling**

Companion to §7f. Cohort: 99 patients with `ground_truth_drug
∈ {Triton, Coral}` (excludes None AND Kraken). The last unanswered
sub-decision in Task 1.

```powershell
.venv\Scripts\python.exe src\task1_drug_id\train_triton_coral.py
```

**Result: at-or-below chance on the holdout.** This is the
empirical proof that Triton and Coral cannot be discriminated from
triage data alone.

**5-fold CV (n=99, 52% Triton):**

| Model | ROC-AUC | PR-AUC | accuracy | BSS |
|---|---:|---:|---:|---:|
| logreg | 0.411 | 0.483 | 0.445 | −0.618 |
| **rforest** | **0.506** | **0.600** | **0.515** | **−0.038** |
| hgb | 0.458 | 0.538 | 0.523 | −0.356 |

**Temporal holdout (test = last day, n=21, 57% Triton):**

| Model | ROC-AUC | PR-AUC | accuracy | Sens | Spec | BSS |
|---|---:|---:|---:|---:|---:|---:|
| **logreg** | **0.389** ⚠ | 0.560 | 0.429 | 0.67 | 0.11 | −0.874 |
| rforest | 0.389 ⚠ | 0.609 | 0.476 | 0.58 | 0.33 | −0.165 |
| hgb | 0.324 ⚠ | 0.496 | 0.381 | 0.42 | 0.33 | −0.728 |

**Why this is the ceiling**:

- v6 Triton signature: palpitations / "ringing in ears" /
  psychomotor slowing — all live in HPI/MDM, **post-triage**.
- v6 Coral signature: time distortion / perceptual alteration /
  spatial disorientation — same, **post-triage**.
- Both have **near-normal triage vitals and labs** (HR ~88-90,
  Temp ~37.1-37.2, AG <12, all peak labs near-normal). The two
  classes are essentially identical at minute 0.

All BSS values are negative — every model loses to "always predict
the prevalence". The tiny holdout (n=21) inflates variance but
doesn't explain the directional finding: CV (n=99) also lands at
ROC-AUC 0.41-0.51.

**Implication**: any architecture that needs to separate Triton
from Coral at triage (the tier-2 multiclass §7d, the cascade §7e)
inherits this ceiling. The path forward for Triton/Coral
discrimination is the **4-hour-horizon feature set** (Task 2's
`features_fourh.csv`), where the PE binaries appear:
`pe_reduced_tracking` and `pe_slow_responses` flag Triton;
`pe_unsteady_gait` and `pe_ataxia` flag Coral (per v6 +
`research/03_v6_feature_evaluation.md`). The cascade-variant
comparison in §7h shows that *replacing* the T-vs-C model with a
prevalence-only split actually **improves** holdout macro AUC.

### 7h. Cascade variants — pick the right way to compose the binaries

What's the best way to assemble the binary classifiers (§7c, §7f,
§7g) into a full 4-class prediction? Comparison of three cascade
constructions vs the direct 4-class baseline:

```powershell
.venv\Scripts\python.exe src\task1_drug_id\compare_cascades.py
```

Four architectures, same folds + same temporal split, three model
families:

| Arch | Stage 1 | Stage 2 | Stage 3 (T-vs-C) |
|---|---|---|---|
| **D** Direct | one 4-class | — | — |
| **A** Cascade-A | tier-1 binary | tier-2 multiclass (K/T/C) | (built into stage 2) |
| **B** Cascade-B | tier-1 binary | Kraken-vs-rest binary | **training prevalence (no model)** |
| **C** Cascade-C | tier-1 binary | Kraken-vs-rest binary | Triton-vs-Coral binary |

**Temporal holdout macro ROC-AUC (n=74)**:

| Model | D | A | **B** | C | Best |
|---|---:|---:|---:|---:|---|
| logreg | 0.600 | 0.609 | **0.660** | 0.608 | B (+0.060) |
| rforest | 0.703 | 0.710 | **0.721** | 0.707 | **B (+0.018)** |
| hgb | 0.685 | 0.680 | **0.716** | 0.687 | B (+0.031) |

**Cascade B wins on every model.** After re-running the pipeline
with the upstream task2 features ported in (§4 step 6), the single
best Task-1 configuration is now **rforest + Cascade-B**:
**0.721 macro ROC-AUC** on the holdout, narrowly above hgb +
Cascade-B (0.716). Both beat direct 4-class (0.703 best across the
three models).

**Why prevalence beats a real T-vs-C model**: §7g showed
Triton-vs-Coral at triage is sub-chance (ROC-AUC 0.32-0.39). A
sub-chance model is *anti-calibrated* — confident predictions are
systematically wrong. Replacing it with prevalence (uninformative
but calibrated) preserves the non-K probability mass without
adding noise. Cascade B improves *every class*'s AUC on the
holdout (hgb: ΔNone +0.054, ΔKraken +0.055, ΔTriton +0.020,
ΔCoral +0.022).

**Per-class deltas vs direct, holdout, hgb:**

| Arch | Δ AUC None | Δ AUC Kraken | Δ AUC Triton | Δ AUC Coral |
|---|---:|---:|---:|---:|
| A vs D | +0.002 | −0.029 | +0.013 | +0.046 |
| **B vs D** | **+0.054** | **+0.055** | **+0.020** | **+0.022** |
| C vs D | +0.014 | +0.012 | +0.018 | +0.020 |

**Updated deployment recommendation**: **Cascade-B with rforest**
(0.721 holdout) is the new champion, with hgb (0.716) a close second
that's worth keeping as a calibration alternative. Both require two
trained models (§7c tier-1 + §7f Kraken-vs-rest) plus one scalar
prevalence constant. Smaller surface area, better calibration,
better holdout AUC than the §7e cascade.

Full analysis: [`research/05_cascade_variants.md`](research/05_cascade_variants.md).

#### Deeper analysis via `evaluate_cascades.py`

A separate analysis layer reads the comparison summary and produces
six structured views. No model fitting — pure analysis.

```powershell
.venv\Scripts\python.exe src\task1_drug_id\evaluate_cascades.py
```

Outputs:
- `derived/task1_cascade_evaluation_report.md` (structured analysis)
- `derived/task1_cascade_evaluation_rankings.csv` (long-form rank table)

**Win counts across 24 metric cells** (2 splits × 3 models × 4 metrics:
macro ROC-AUC, macro PR-AUC, accuracy, log-loss):

| Architecture | Wins | Win rate |
|---|---:|---:|
| **Cascade B** (tier-1 + K-vs-rest + prev) | **18** | **75%** |
| Cascade A (tier-1 + tier-2-multi) | 3 | 12% |
| Direct 4-class | 3 | 12% |
| Cascade C (tier-1 + K-vs-rest + T-vs-C model) | 2 | 8% |

**Consistency** — does each cascade beat direct on **both** CV and
holdout?

| Model | Cascade | Δ AUC CV | Δ AUC holdout | Both? |
|---|---|---:|---:|---|
| logreg | A | −0.013 | +0.008 | no |
| **logreg** | **B** | **+0.001** | **+0.060** | **YES** |
| logreg | C | −0.019 | +0.007 | no |
| **rforest** | **A** | **+0.002** | **+0.007** | **YES** |
| **rforest** | **B** | **+0.005** | **+0.018** | **YES** |
| rforest | C | −0.003 | +0.004 | no |
| hgb | A | −0.022 | −0.005 | no |
| **hgb** | **B** | **+0.027** | **+0.031** | **YES** ← biggest mean Δ |
| hgb | C | −0.002 | +0.002 | no |

**Four (model × cascade) pairs are consistent winners; three of the
four are Cascade B.** hgb × Cascade B has the largest mean Δ across
the two splits (+0.033).

**Stability across model families** (lower spread = more
model-agnostic; matters because we don't always know which model
will land best on Phase-2 data):

| Architecture | Mean (holdout) | Std | Spread |
|---|---:|---:|---:|
| **Cascade B** | **0.699** | **0.025** | **0.061** ← tightest |
| Direct | 0.663 | 0.038 | 0.085 |
| Cascade A | 0.666 | 0.041 | 0.101 |
| Cascade C | 0.667 | 0.041 | 0.099 |

Cascade B not only has the highest mean holdout AUC, it has the
**lowest variance** across logreg/rforest/hgb — switching model
families costs you less if you ship Cascade B.

**Per-class OvR-AUC winners on the holdout** (which architecture
wins each class for each model?):

| Class | Winning architecture (most frequent across the 3 models) |
|---|---|
| None | Cascade A — 3/3 model families |
| Kraken | Cascade B — 2/3 (rforest, logreg). Direct wins hgb. |
| **Triton** | **Cascade B — 3/3 model families** |
| **Coral** | **Cascade B — 3/3 model families** |

Cascade B sweeps Triton and Coral OvR AUC across every model — the
prevalence-based T/C split is more discriminating than any fitted
T-vs-C model. Cascade A wins None (because tier-1's binary
predictions feed both architectures equally, but A's softer 4-class
output produces better None calibration than B's sharper
"1 − P(drug)" decision boundary). The `evaluate_cascades.py`
auto-recommendation maximises mean Δ vs direct (favouring the
biggest *improvement*) and lands on **logreg × Cascade-B** (+0.031
mean); the **deployment** recommendation maximises raw holdout AUC
and lands on **rforest × Cascade-B** (0.721 holdout, +0.018 over
direct). Both are documented; pick by your decision criterion.

---

## 8. Train + evaluate Task 2 (4h deterioration)

**What is predicted**: `encounter_disposition_label` ∈ {Discharge,
Floor, ICU}. Sourced from `outcomes.csv` (which in turn sources from
the xlsx `Disposition` sheet — never from a feature file).

**Cohort filter** (default): drug-positive per ground truth
(`outcomes.csv :: ground_truth_drug != 0`), 157 of 261 patients.
Drug class is **only the cohort selector** — the model itself
predicts Discharge / Floor / ICU. Override with
`--cohort all` to train on every encounter.

Two variants per run:
- **WITH** drug-class probs as features (requires §5b)
- **WITHOUT** (clinical features only — runs unconditionally)

### 8a. 5-fold stratified cross-validation

```powershell
# Default — drug-positive cohort (brief's stated scope, n=157)
.venv\Scripts\python.exe src\task2_deterioration\train_baseline.py

# OR: all-patients cohort (n=261, includes None-class)
.venv\Scripts\python.exe src\task2_deterioration\train_baseline.py --cohort all
```

The `--cohort` flag has two values:
- `drug-positive` (default) — patients with `ground_truth_drug != 0`,
  157 of 261. Matches the hackathon brief's Task-2 scope.
- `all` — every encounter (261). None-class patients are
  predominantly Discharge, which inflates the macro AUC.

Outputs land at `task2_baseline_summary.csv` /
`task2_oof_predictions.csv` for the drug-positive cohort and
`task2_baseline_summary_all.csv` / `task2_oof_predictions_all.csv`
for the all-patients cohort.

Artifacts (WITH-probs variant overrides):
`derived/task2_baseline_summary.csv` + `task2_oof_predictions.csv`.

**WITH drug-probs variant** (default-saved, drug-positive cohort n=157):

| Model | log-loss | accuracy | macro ROC-AUC | macro PR-AUC |
|-------|---------:|---------:|--------------:|-------------:|
| logreg | 0.42 | 0.91 | 0.888 | 0.831 |
| **rforest** | **0.34** | **0.91** | **0.932** | **0.897** |
| hgb | 0.61 | 0.83 | 0.882 | 0.713 |

Class prevalence: Discharge 0.77 / Floor 0.14 / ICU 0.09. The new 4h
HPI/MDM features (§4 step 6) lifted rforest macro AUC 0.924 → 0.932
and macro PR-AUC 0.887 → 0.897 vs the prior run.

**Per-class metrics for rforest (drug-positive, n=157):**

| Class | Prevalence | ROC-AUC | PR-AUC | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| Discharge | 0.77 | 0.932 | 0.967 | 0.065 | **+0.629** |
| Floor | 0.14 | 0.945 | 0.856 | 0.060 | **+0.491** |
| ICU | 0.09 | 0.903 | 0.883 | 0.032 | **+0.611** |

**All-patients cohort (n=261, with-probs variant)**:

| Model | log-loss | accuracy | macro ROC-AUC | macro PR-AUC |
|-------|---------:|---------:|--------------:|-------------:|
| logreg | 0.65 | 0.84 | 0.911 | 0.820 |
| **rforest** | **0.34** | **0.91** | **0.950** | **0.893** |
| hgb | 0.46 | 0.89 | 0.934 | 0.868 |

Class prevalence on full cohort: Discharge 0.66 / Floor 0.20 / ICU 0.15.

**Per-class metrics for rforest (all patients, n=261):**

| Class | Prevalence | ROC-AUC | PR-AUC | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| Discharge | 0.66 | 0.952 | 0.956 | 0.059 | **+0.738** |
| Floor | 0.20 | 0.946 | 0.839 | 0.066 | **+0.585** |
| ICU | 0.15 | 0.952 | 0.883 | 0.034 | **+0.727** |

Disposition distribution on the full cohort: Discharge 171 (66%) /
Floor 52 (20%) / ICU 38 (15%). The all-patients macro AUC is
higher than drug-positive (0.955 vs 0.927) because the added
None-class patients are predominantly Discharge — they're easy
calls that pad the macro average. The **drug-positive cohort is
the harder, brief-relevant problem**: predicting deterioration
among festival-drug patients where Discharge is no longer the
dominant class.

### 8b. Temporal holdout — **deployment-relevant**

Last day of festival (peak day) used as test; prior days = train.
With the 5-day release this is **112 train / 45 test** for the
drug-positive cohort, or **187 train / 74 test** for all patients.

```powershell
# Default: drug-positive cohort
.venv\Scripts\python.exe src\eval_temporal.py

# All-patients cohort
.venv\Scripts\python.exe src\eval_temporal.py --cohort all
```

Artifacts (Task 2): `derived/task2_temporal_summary.csv` +
`task2_temporal_predictions.csv` + section 2 of
`temporal_holdout_report.md`.

| Model | log-loss | accuracy | macro ROC-AUC | macro PR-AUC |
|-------|---------:|---------:|--------------:|-------------:|
| logreg | 0.32 | 0.89 | 0.959 | 0.885 |
| **rforest** | **0.32** | **0.89** | **0.980** | **0.933** |
| hgb | 0.54 | 0.87 | 0.888 | 0.669 |

Test-set class prevalence: Discharge 0.73 / Floor 0.16 / ICU 0.11.

**Per-class metrics for rforest (drug-positive holdout) — headline:**

| Class | Prevalence | ROC-AUC | PR-AUC | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| Discharge | 0.73 | 0.990 | 0.996 | 0.054 | **+0.724** |
| Floor | 0.16 | 0.955 | 0.836 | 0.074 | **+0.437** |
| ICU | 0.11 | 0.995 | 0.967 | 0.038 | **+0.615** |

**All-patients holdout (n=261, test = last day n=74)** — `task2_temporal_summary_all.csv`:

| Model | log-loss | accuracy | macro ROC-AUC | macro PR-AUC |
|-------|---------:|---------:|--------------:|-------------:|
| logreg | 0.51 | 0.81 | 0.921 | 0.844 |
| **rforest** | **0.28** | **0.91** | **0.988** | **0.969** |
| hgb | 0.29 | 0.88 | 0.972 | 0.920 |

Task 2 generalises even better when None-class is included
(0.988 vs 0.980 drug-positive) — the additional Discharge-heavy
None patients provide more high-confidence negatives without
hurting Floor/ICU discrimination.

Task-2 holdout rforest hits **0.980 macro ROC-AUC and 0.933 macro
PR-AUC** on the peak festival day. ICU-class ROC-AUC is **0.995**
and PR-AUC **0.967** — every ICU case in the test set ranks at or
near the top of the predicted-ICU score. All three BSS values are
firmly positive (+0.44 to +0.72): the model beats climatology by
roughly half of the theoretical maximum improvement. The
severity-anchored features (peak_lactate, peak_CPK, peak_troponin,
peak_HR, peak_temp) transfer cleanly from training days to the
holdout; the new HPI/MDM 4h features (§4 step 6) added marginal
noise on this specific split but improved CV-side numbers (§8a).

---

## 9. Task 3 — rapid triage tool

Static HTML+JS, no build step, no server:

```powershell
start src\task3_rapid_tool\triage_calculator.html
```

Inputs: vitals (HR / RR / SBP / DBP / SpO2 / Temp / GCS / SuppO2),
age, festival flag, arrival mode, chief complaint, symptom-onset
minutes, ESI, 10 exam-finding checkboxes, optional POC labs.
Output: four class probabilities + an evidence trail of which
inputs pushed which direction. Prints to a 1-page paper card.

---

## 10. Unsupervised clustering — task-aligned cohorts

```powershell
.venv\Scripts\python.exe src\unsupervised\cluster.py
```

Two clustering runs mirroring the supervised models' targets:

- **task1**: all 261 encounters, `features_triage.csv`, KMeans(4),
  truth = `ground_truth_drug_name` (None / Kraken / Triton / Coral)
- **task2**: 157 drug-positive (per `ground_truth_drug != 0`),
  `features_fourh.csv`, KMeans(3), truth =
  **`encounter_disposition_label`** (Discharge / Floor / ICU) —
  the actual Task-2 outcome, not the drug class

For each: HDBSCAN runs too (typically finds 0 dense clusters — real
result, features are high-dim and toxidromes overlap), PCA(2) scatter
plotted with **centroid markers** (white X), **convex-hull cluster
boundaries**, and **per-centroid class-fraction annotations** showing
*every* ground-truth class with its within-cluster fraction (ranked).
For Task 1, two clusters can be dominantly "None" (None covers 40% of
the cohort) — the annotations make the mix visible rather than
masking it with unique-assignment.

Artifacts:
- `derived/clusters_task1.csv`, `clusters_task2.csv` — cluster
  assignments per encounter
- `derived/cluster_pca_task1.png`, `cluster_pca_task2.png` —
  scatter plots
- `derived/cluster_review_task1/cluster_<k>.csv` (4 files) and
  `derived/cluster_review_task2/cluster_<k>.csv` (3 files) —
  per-cluster review CSVs containing encounter_id, assigned
  cluster, ground_truth_drug_name, and distance from EVERY
  cluster's centroid (sorted by distance to own centroid)

Stdout reports cluster ARI / NMI vs the task-specific truth and
top-1 macro purity. Currently:

| Run | n | Truth | KMeans ARI | Macro top-1 purity |
|---|---:|---|---:|---|
| task1 | 261 | drug class | 0.049 | 0.48 |
| task2 | 157 | disposition | **0.281** | **0.789** (C2 = 50% Floor + 38% ICU = 88% admitted) |

Aligning Task-2 clustering with the disposition target (instead of
drug class) tripled both ARI (0.05 → 0.28) and macro purity
(0.52 → 0.79). The features built on the 4h horizon track
disposition severity, not toxidrome identity — which is the
correct answer for the supervised Task-2 model too.

Plot headings have been updated:

- `cluster_pca_task1.png` → "Task 1 (drug ID at triage, n=261)
  | KMeans(4) vs ground-truth drug class"
- `cluster_pca_task2.png` → "Task 2 (4h deterioration) |
  KMeans(3) vs ground-truth disposition"

---

## 11. Full one-shot Phase-2 retrain

When fresh xlsx arrives the day-of:

```powershell
# Drop the new xlsx + ground truth at data/ first, then:
.venv\Scripts\python.exe run_pipeline.py --skip-agents
```

`run_pipeline.py` now runs the full pipeline end-to-end:
feature extraction (incl. v6), label load, outcomes consolidation,
EDA, cleanup, leakage sentinel, and both task baselines. Total
wall-clock today: ~5–10 min on a laptop (excluding the LLM agent
step which is harness-spawned).

Flags:
- `--features-only` — stop after §4, skip training
- `--skip-agents` — reuse existing `probs_<N>.csv` if you already
  ran §5b

---

## Troubleshooting

**`FileNotFoundError: Hackathon_Data_Release_1_SHARE.xlsx`** —
drop the xlsx under `data/`. Filenames are exact; case matters on
Linux.

**`AssertionError: Leakage in features_triage.csv: [...]`** — a
feature script reintroduced a forbidden column. Inspect the listed
names, trace back to which script added them, and either drop them
or move them to the 4h side. The sentinel forbids
`vts_/lts_/itv_/xmod_/stab_/arc_/diff_/abs_diff_/pct_change_/direction_`
prefixes, anything containing `_4h`, `encounter_disposition_label`,
and `arrival_same_day_volume`.

**Task 2 fails on missing `probs_avg.csv`** — you're running the
WITH-probs variant without having merged the 10 agents. Either run
§5b or rely on the WITHOUT-probs numbers that the same training run
also reports.

**`PerformanceWarning: DataFrame is highly fragmented`** —
cosmetic. `extract_differentials.py` adds ~37 columns one at a time;
pandas warns but the output is correct.

**`Input contains NaN` from sklearn metrics in `cluster.py`** —
the None-class rows have NaN `ground_truth_drug_name`. The script
coerces NaN → "None" before passing to metrics; if you see this,
your `ground_truth.csv` is malformed.

**`ImportError: tabulate`** in `eval_temporal.py` markdown step —
fixed in-script via a manual `_df_to_md` formatter; no action needed.

**v6 keyword features come out all-zero** (`triage_chief_ringing_ears`,
`triage_chief_psychomotor_slow`, `triage_chief_perceptual`,
`triage_chief_spatial`, `note_inward_density`) — this is a FINDING,
not a bug. Those Triton/Coral defining tokens appear only in HPI/MDM
(post-triage), not in `triage_brief_note`. `cleanup_features.py`
drops them as constant. Task-1 ceiling for Triton/Coral
discrimination is data-bound, not engineering-bound.

**`No module named 'mord'`** — required by Task 2 ordinal-regression
experiments (currently disabled). Reinstall: `pip install -r requirements.txt`.

---

## File map (what lives where)

```
data/                          ← raw xlsx + ground-truth csv (gitignored)
derived/                       ← all pipeline outputs (tracked)
  narratives_triage.jsonl      ← triage-only notes (chief complaint, brief note, mode of arrival)
  narratives_fourh.jsonl       ← triage + 4h narratives (HPI, PE+, MDM, course, meds/procs) — agent input
  features_triage.csv          ← Task-1 inputs (triage horizon)
  features_fourh.csv           ← Task-2 inputs (4h horizon)
  ground_truth.csv             ← manual drug labels (intermediate)
  outcomes.csv                 ← CANONICAL labels for both tasks:
                                 ground_truth_drug + encounter_disposition_label
                                 (built by src/labels/build_outcomes.py)
  probs_1..10.csv              ← per-agent (LLM) probabilities — optional
  probs_avg.csv                ← averaged agent consensus — optional
  exploratory_features.csv     ← cand_* candidate features
  task1_baseline_summary.csv + task1_oof_predictions.csv
  task2_baseline_summary.csv + task2_oof_predictions.csv
  task1_temporal_summary.csv + task1_temporal_predictions.csv
  task2_temporal_summary.csv + task2_temporal_predictions.csv
  temporal_holdout_report.md
  v6_feature_audit.md
  eda_descriptive_report.md + eda_plots/
  clusters_task1.csv + clusters_task2.csv
  cluster_pca_task1.png + cluster_pca_task2.png
  cluster_review_task1/cluster_{0,1,2,3}.csv
  cluster_review_task2/cluster_{0,1,2}.csv
src/
  features/                    ← extract_* + cleanup_features.py
    extract_v6_features.py     ← PE binaries + peak thresholds + triage keywords (NEW)
  labels/                      ← load_ground_truth.py (drug labels),
                                 build_outcomes.py (canonical outcomes.csv —
                                   drug + disposition merged),
                                 merge_probabilities.py,
                                 agents/agent_01..10 + PROMPTS.md (v6-aligned)
  eda/                         ← eda_descriptive.py, eda_advanced.py,
                                 check_time_horizons.py,
                                 eda_v6_features.py,
                                 check_task2_leakage.py (verifies
                                   Task-2 does NOT use disposition
                                   as a feature; MI scan, alias
                                   scan, target-removal proof)
  task1_drug_id/               ← train_baseline.py (Task 1)
  task2_deterioration/         ← train_baseline.py (Task 2)
  task3_rapid_tool/            ← triage_calculator.html (Task 3)
  unsupervised/                ← cluster.py (task-aligned, centroids + boundaries +
                                 full-class-mix annotations + per-cluster review CSVs)
  eval_temporal.py             ← train-on-early-days / test-on-last-day (NEW)
run_pipeline.py                ← orchestrator (§4 + §12); NOTE: v6 step not yet wired in
PITCH.md                       ← 4-minute pitch deck (8 slides)
RUNBOOK.md                     ← this file
README.md                      ← project overview
research/
  01_v6_features_and_prompts.md  ← Round 1 feature design (32 features)
  02_v6_extended_features.md     ← Round 2 (39 additional features)
  03_v6_feature_evaluation.md    ← build + EDA + train + assess
```
