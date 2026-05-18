# Runbook — SAEM Hackathon Pipeline

End-to-end execution from raw xlsx → trained, evaluated Task-1 /
Task-2 / Task-3 models, plus the v6 feature audit, task-aligned
clustering, and temporal-holdout evaluation. Every command runs from
the repo root (`SAEM_2026_hackathon/`). PowerShell on Windows;
substitute `.venv/bin/python` on macOS/Linux.

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

---

## 7. Train + evaluate Task 1 (drug ID at triage)

```powershell
.venv\Scripts\python.exe src\task1_drug_id\train_baseline.py
```

5-fold stratified CV. Three models (logreg, rforest, hgb).

Artifacts:
- `derived/task1_baseline_summary.csv`
- `derived/task1_oof_predictions.csv`

Current numbers (leakage-clean, manual ground truth, v6 features):

| Model | log-loss | accuracy | macro AUC |
|-------|---------:|---------:|----------:|
| logreg | 1.91 | 0.37 | 0.63 |
| **rforest** | **1.23** | **0.44** | **0.70** |
| hgb | 1.98 | 0.45 | 0.64 |

Majority-class baseline accuracy = 0.40.

---

## 8. Train + evaluate Task 2 (4h deterioration)

```powershell
.venv\Scripts\python.exe src\task2_deterioration\train_baseline.py
```

5-fold stratified CV. Two variants:
- **WITH** drug-class probs as features (requires §5b)
- **WITHOUT** (clinical features only — runs unconditionally)

Cohort: drug-positive per ground truth (`ground_truth_drug != 0`).
157 of 261 patients.

Artifacts (WITH-probs variant overrides):
- `derived/task2_baseline_summary.csv`
- `derived/task2_oof_predictions.csv`

Current numbers (clinical-only variant, v6 features):

| Model | log-loss | accuracy | macro AUC | AUC disp / floor / ICU |
|-------|---------:|---------:|----------:|-----------------------:|
| logreg | 0.41 | 0.91 | 0.89 | 0.90 / 0.91 / 0.86 |
| **rforest** | **0.34** | **0.91** | **0.93** | 0.93 / 0.94 / 0.91 |
| hgb | 0.64 | 0.82 | 0.87 | 0.90 / 0.85 / 0.87 |

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

## 10. Temporal holdout — train on early days, test on last day

Mirrors a Phase-2 deployment scenario where the model trained on
prior days must predict on a fresh wave of arrivals.

```powershell
.venv\Scripts\python.exe src\eval_temporal.py
```

Split: train on `encounter_arrival_date < last_day`; test on the
last day only. With the 5-day release: **187 train / 74 test** for
Task 1; **112 train / 45 test** for Task 2 (drug-positive cohort).

Artifacts:
- `derived/task1_temporal_summary.csv` + `task1_temporal_predictions.csv`
- `derived/task2_temporal_summary.csv` + `task2_temporal_predictions.csv`
- `derived/temporal_holdout_report.md`

Current holdout numbers:

| Task | Model | Accuracy | Macro AUC |
|---|---|---:|---:|
| 1 | rforest | 0.41 | 0.69 |
| 2 | **rforest** | **0.89** | **0.98** |
| 2 | logreg | 0.89 | 0.96 |

Task 2 generalizes spectacularly to the holdout day — severity
anchors lock in. Task 1 is on par with CV (triage data fundamentally
caps at ~0.70 AUC; see `research/03_v6_feature_evaluation.md`).

---

## 11. Unsupervised clustering — task-aligned cohorts

```powershell
.venv\Scripts\python.exe src\unsupervised\cluster.py
```

Two clustering runs mirroring the supervised models' cohorts:

- **task1**: all 261 encounters, `features_triage.csv`, KMeans(4)
  (3 drugs + None)
- **task2**: 157 drug-positive (per `ground_truth_drug != 0`),
  `features_fourh.csv`, KMeans(3)

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

Stdout reports cluster ARI / NMI vs ground truth and top-1 macro
purity. Currently:

| Run | n | KMeans ARI | Macro top-1 purity |
|---|---:|---:|---:|
| task1 | 261 | 0.05 | 0.48 |
| task2 | 157 | 0.04 | 0.52 (cluster 2 = 69% Kraken) |

---

## 12. Full one-shot Phase-2 retrain

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
                                 eda_v6_features.py (NEW)
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
