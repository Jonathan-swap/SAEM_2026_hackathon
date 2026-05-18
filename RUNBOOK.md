# Runbook — SAEM Hackathon Pipeline

End-to-end execution from raw xlsx → trained, evaluated Task-1 /
Task-2 / Task-3 models with both **5-fold cross-validation** (§7a /
§8a) and **temporal holdout** (§7b / §8b, train on early days, test
on last day — the deployment-relevant metric). Plus v6 feature
audit, task-aligned clustering. Every command runs from the repo
root (`SAEM_2026_hackathon/`). PowerShell on Windows; substitute
`.venv/bin/python` on macOS/Linux.

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
| 1 | `src/features/extract_narratives.py`    | `derived/narratives.jsonl` |
| 2 | `src/features/extract_structured.py`    | `derived/features_triage.csv` + `features_fourh.csv` |
| 3 | `src/features/extract_time_features.py` | adds Groups A–G time features (capped at min ≤ 240) |
| 4 | `src/features/extract_differentials.py` | adds triage↔4h diffs to `features_fourh.csv` |
| 5 | `src/features/extract_note_features.py` | adds parsed onset minutes + festival flags |
| 6 | `src/labels/load_ground_truth.py`       | `derived/ground_truth.csv` |

Pass criterion: each step logs `OK` and the time-features script ends
with `OK: no leakage; triage features contain only arrival-time +
minute-0 signals.`

### 4.1. v6 features (run manually after §4)

`run_pipeline.py` does not yet include this step. Run it explicitly:

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
each reading `derived/narratives.jsonl` under a different reasoning
emphasis. **Prompts: [`src/labels/agents/PROMPTS.md`](src/labels/agents/PROMPTS.md).**

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
| Triton | 0.20 | 0.743 | 0.457 | 0.144 | **+0.085** |
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
| **rforest** | **1.22** | **0.42** | **0.691** | **0.413** |
| hgb | 1.70 | 0.43 | 0.685 | 0.418 |

**Per-class metrics for rforest (holdout):**

| Class | Prevalence | ROC-AUC | PR-AUC | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| None | 0.39 | 0.704 | 0.639 | 0.211 | **+0.114** |
| Kraken | 0.32 | 0.629 | 0.474 | 0.218 | +0.006 |
| Triton | 0.16 | 0.739 | 0.320 | 0.128 | **+0.057** |
| Coral | 0.12 | 0.691 | 0.221 | 0.112 | −0.046 |

Holdout AUC ≈ CV AUC — the Task-1 ceiling is data-bound. See
`research/03_v6_feature_evaluation.md`.

---

## 8. Train + evaluate Task 2 (4h deterioration)

Cohort: drug-positive per ground truth (`ground_truth_drug != 0`),
157 of 261 patients. Two variants per run:
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

Clinical-only variant:

| Model | log-loss | accuracy | macro ROC-AUC | macro PR-AUC |
|-------|---------:|---------:|--------------:|-------------:|
| logreg | 0.41 | 0.91 | 0.888 | 0.830 |
| **rforest** | **0.35** | **0.91** | **0.927** | **0.895** |
| hgb | 0.64 | 0.82 | 0.879 | 0.702 |

Class prevalence: Discharge 0.77 / Floor 0.14 / ICU 0.09.

**Per-class metrics for rforest (drug-positive, n=157):**

| Class | Prevalence | ROC-AUC | PR-AUC | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| Discharge | 0.77 | 0.933 | 0.966 | 0.066 | **+0.624** |
| Floor | 0.14 | 0.934 | 0.834 | 0.061 | **+0.483** |
| ICU | 0.09 | 0.915 | 0.887 | 0.032 | **+0.613** |

**Macro metrics for all-patients cohort (n=261, rforest)**:

| Variant | macro ROC-AUC | macro PR-AUC |
|---|---:|---:|
| WITHOUT drug-probs (clinical-only) | **0.955** | **0.905** |
| WITH drug-probs | **0.955** | **0.904** |

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
drug-positive cohort.

```powershell
.venv\Scripts\python.exe src\eval_temporal.py    # Task 1 + Task 2 in one run
```

Artifacts (Task 2): `derived/task2_temporal_summary.csv` +
`task2_temporal_predictions.csv` + section 2 of
`temporal_holdout_report.md`.

| Model | log-loss | accuracy | macro ROC-AUC | macro PR-AUC |
|-------|---------:|---------:|--------------:|-------------:|
| logreg | 0.32 | 0.89 | 0.957 | 0.884 |
| **rforest** | **0.32** | **0.89** | **0.985** | **0.952** |
| hgb | 0.58 | 0.87 | 0.869 | 0.653 |

Test-set class prevalence: Discharge 0.73 / Floor 0.16 / ICU 0.11.

**Per-class metrics for rforest (holdout) — headline numbers:**

| Class | Prevalence | ROC-AUC | PR-AUC | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| Discharge | 0.73 | 0.992 | 0.997 | 0.053 | **+0.730** |
| Floor | 0.16 | 0.962 | 0.858 | 0.070 | **+0.467** |
| ICU | 0.11 | **1.000** | **1.000** | 0.038 | **+0.617** |

Task-2 holdout rforest hits **0.985 macro ROC-AUC and 0.952 macro
PR-AUC** on the peak festival day. ICU-class ROC-AUC and PR-AUC
both hit **1.00** — every ICU case in the test set ranks above
every non-ICU case. All three BSS values are firmly positive
(+0.47 to +0.73): the model beats climatology by roughly half of
the theoretical maximum improvement. The severity-anchored
features (peak_lactate, peak_CPK, peak_troponin, peak_HR,
peak_temp) transfer cleanly from training days to the holdout.

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
# Manually re-run the v6 step (not yet in run_pipeline.py):
.venv\Scripts\python.exe src\features\extract_v6_features.py
.venv\Scripts\python.exe src\features\cleanup_features.py
.venv\Scripts\python.exe src\task1_drug_id\train_baseline.py
.venv\Scripts\python.exe src\task2_deterioration\train_baseline.py
```

Steps to add to `run_pipeline.py` before Phase 2 (TODO): insert
`extract_v6_features.py` between step 5 (`extract_note_features`)
and step 6 (`load_ground_truth`) so the v6 features are available
to `cleanup_features.py`'s leakage sentinel + the training scripts.

Total wall-clock today: ~5–10 min on a laptop (excluding the LLM
agent step which is harness-spawned).

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
  narratives.jsonl
  features_triage.csv          ← Task-1 inputs (triage horizon)
  features_fourh.csv           ← Task-2 inputs (4h horizon)
  ground_truth.csv             ← manual labels
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
  labels/                      ← load_ground_truth.py, merge_probabilities.py,
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
