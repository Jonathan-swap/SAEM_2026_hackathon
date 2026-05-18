# v6 Feature Boost + Prompt Realignment

Source: [`data/toxidrome_report_v6.pdf`](../data/toxidrome_report_v6.pdf)
(SAEM General ED Operations Team, n=157 ground-truth cases).

This document (1) lists the features the v6 report shows are
discriminating but that our current pipeline either doesn't compute
or doesn't use, and (2) summarizes the changes made to
[`src/labels/agents/PROMPTS.md`](../src/labels/agents/PROMPTS.md) so
the 10-agent consensus aligns with the manually annotated outcomes
in [`data/Task1_Two_Tier_Input_Data.csv`](../data/Task1_Two_Tier_Input_Data.csv).

---

## What the v6 report establishes

| Drug | Inferred class | n (/157) | Defining HPI signal | Defining PE signal | Defining labs |
|---|---|---:|---|---|---|
| **Kraken Candy** | Sympathomimetic (PCP/amph-like) | 58 | agitation chief; marked restlessness; rapid escalation | diaphoretic (p=0.001), mild_tremor (p<0.001), tachycardic (p=0.008), agitated, restless, fatigued | peak_lactate>5.0; peak_CPK>1000; peak_troponin>0.15; AG>20 |
| **Triton Tabs** | CNS depressant + cardiac awareness (THC/benzo-like) | 51 | palpitations / cardiac-awareness chief; ringing in ears | reduced_tracking (p=0.026), slow_responses (p=0.022); diaphoresis & agitation **absent** | all peak labs near-normal; pH > 7.35; AG < 12 |
| **Coral Dust** | Hallucinogen (LSD/psilocybin-like) | 48 | time-distortion sensation; perceptual alteration; spatial disorientation; waxing/waning + perceptual | unsteady gait (29%), ataxia (23%), intermittent disorientation; diaphoresis/agitation **absent** | all peak labs near-normal |

**Critical methodology warnings** (carried over to every agent):

- The phrase *"benzodiazepine for agitation/sympathetic excess"* appears in
  **44% of MDMs across all three drugs**. It is a templated treatment line,
  not drug signal. All behavioral scoring must use HPI text only.
- **Pupil findings (mydriasis / miosis) are absent from the dataset.** Any
  agent reasoning that depends on pupil size cannot run on this data.
- Kraken is enriched among high-severity cases (18/24 = 75%) — this is a
  real pharmacological correlation, not a labeling artifact. Don't try to
  decorrelate severity from drug class.

---

## Features identified (>10 — counted: 32)

The discriminator power of these features is established in the v6
report's "Full Discriminator Table" and the 14-rule hierarchy. I've
split them by which Task they're legal for (the leakage sentinel in
`cleanup_features.py` will reject anything 4h-only that lands in
`features_triage.csv`).

### Task-1 (triage-horizon, must come from `triage_brief_note` + `triage_chief_complaint` + triage vitals/labs)

These are parseable from the triage record alone — no leakage.

| # | Feature | Source | Drug signal |
|---:|---|---|---|
| 1 | `triage_chief_agitation` | regex on chief_complaint OR brief_note for `agitat*`, `combative`, `restless`, `uncontainable` | **Kraken DEFINING** |
| 2 | `triage_chief_palpitations` | regex for `palpitation`, `racing heart`, `cardiac awareness`, `feel my heart` | **Triton STRONG** |
| 3 | `triage_chief_ringing_ears` | regex for `ringing`, `tinnitus`, `ears`, `auditory` | **Triton DEFINING** |
| 4 | `triage_chief_perceptual` | regex for `time distort*`, `perceptual`, `altered`, `wave`, `visual`, `blurry vision`, `unreal` | **Coral DEFINING/STRONG** |
| 5 | `triage_chief_spatial` | regex for `unsteady`, `ataxia`, `dizzy`, `disoriented` | Coral lean |
| 6 | `triage_chief_psychomotor_slow` | regex for `slow*`, `lethargic`, `withdrawn`, `disengaged` | Triton |
| 7 | `triage_ag_above_20` | binary: `triage_lab_anion_gap > 20` | **Kraken STRONG** (rule #14 in hierarchy) |
| 8 | `triage_ag_below_12` | binary: `triage_lab_anion_gap < 12` | Triton lean |
| 9 | `triage_ph_above_735_with_slowing` | binary: `triage_lab_ph > 7.35 AND triage_chief_psychomotor_slow` | **Triton STRONG** (rule #8) |
| 10 | `triage_hr_above_120` | binary: `triage_heart_rate > 120` | Kraken supporting |
| 11 | `triage_temp_above_38` | binary: `triage_temperature_c > 38.0` | Kraken supporting |
| 12 | `triage_sympathomimetic_combo` | binary: `HR>110 AND RR>22 AND T>37.5` | Kraken composite |
| 13 | `triage_glucose_above_140` | binary: `triage_lab_glucose > 140` (Kraken mean 115, Triton/Coral ~100) | Kraken supporting |

### Task-2 (4h-horizon, can additionally use PE + peak labs)

These add the structured PE findings and peak-lab thresholds that
v6 calls out by p-value.

| # | Feature | Source | Drug signal |
|---:|---|---|---|
| 14 | `pe_diaphoretic` | token in `physical_exam_pertinent_positives` | **Kraken p=0.001** |
| 15 | `pe_tachycardic` | token | Kraken p=0.008 |
| 16 | `pe_mild_tremor` | token (`mild_tremor`) | **Kraken p<0.001** |
| 17 | `pe_agitated` | token | Kraken p=0.003 |
| 18 | `pe_restless` | token | Kraken p=0.042 |
| 19 | `pe_fatigued_appearance` | token | Kraken p=0.001 |
| 20 | `pe_reduced_tracking` | token | **Triton p=0.026** |
| 21 | `pe_slow_responses` | token | **Triton p=0.022** |
| 22 | `pe_distractible` | token | Triton |
| 23 | `pe_unsteady_gait` | token | Coral (29%) |
| 24 | `pe_ataxia` | token | Coral (23%) |
| 25 | `pe_intermittent_disorientation` | token | Coral (8%) |
| 26 | `pe_kraken_combo` | `pe_diaphoretic AND pe_tachycardic` | **Kraken 64% PPV** (rule #2) |
| 27 | `pe_triton_combo` | `pe_reduced_tracking AND pe_slow_responses` | **Triton 53% PPV** (rule #5) |
| 28 | `peak_lactate_5plus` | `vts_/lts_lactate_max_value > 5.0` | **Kraken VERY STRONG** (rule #1) |
| 29 | `peak_cpk_1000plus` | `lts_cpk_max_value > 1000` | **Kraken VERY STRONG** (rule #1) |
| 30 | `peak_troponin_015plus` | `lts_troponin_max_value > 0.15` | Kraken (rule from table) |
| 31 | `peak_hr_150plus` | `vts_heart_rate_max > 150` | Kraken |
| 32 | `peak_temp_385plus` | `vts_temperature_c_max > 38.5` | Kraken |
| 33 | `all_peak_labs_normal` | `lts_cpk_max < 200 AND lts_lactate_max < 1.5 AND lts_troponin_max < 0.05` | **Triton/Coral confirmation** |
| 34 | `kraken_severity_anchor` | rule-bundle: any of `peak_lactate>5` / `peak_CPK>1000` / `peak_troponin>0.15` | Maps to rule #1 of v6 hierarchy |

**Implementation hint:** features 14–27 require a new
`src/features/extract_pe_features.py` that parses the semicolon-delimited
`physical_exam_pertinent_positives` field from the `Four_Hour_Data`
sheet. Features 28–34 are one-line threshold flags derived from the
existing `vts_/lts_` columns in `features_fourh.csv` and can be added
to `cleanup_features.py` or a new `extract_threshold_flags.py`.

**Privacy note:** none of these features leak row data — they're
structured binary flags computed from columns already in the xlsx.

---

## Prompt changes (`src/labels/agents/PROMPTS.md`)

The current prompts mis-described Triton ("bradycardia, hypotension,
hypoventilation, miosis") — v6 reveals Triton presents with normal HR
and patient-reported cardiac awareness, not the classical sedative
profile. Six structural fixes:

1. **New "v6 Pre-Read" block** at the top of `PROMPTS.md` that every
   agent must consume before reasoning: the corrected fixed mapping,
   the 14-rule discriminator hierarchy verbatim, the MDM-boilerplate
   warning, and the no-pupils caveat.
2. **Fixed mapping rewritten** across all 10 agents:
   - Triton no longer described as bradycardic/miotic; described as
     CNS-depressant with subjective tachycardia awareness, ringing
     in ears, and slowed responses.
   - Kraken peak-lab thresholds added (lactate/CPK/troponin).
   - Coral emphasized via absence of Kraken/Triton PE findings.
3. **Agent 4 (MDM-led) repurposed → MDM-minus-boilerplate.** Original
   "weight MDM heaviest" is contaminated. Now: scan MDM for
   *differential diagnosis statements* and *severity tier* only, and
   explicitly *strip* the boilerplate phrase before reasoning.
4. **Agent 7 (pupil + autonomic) repurposed → autonomic-only.** Removes
   the pupil branch entirely. Adds diaphoresis/tremor branches and
   peak-lab branches as the dominant autonomic signal.
5. **Agent 10 (token-cluster) updated** to use the actual 14 PE
   findings in the dataset (was using textbook tokens like
   `mydriasis`/`miosis`/`hypertensive`/`flushed` that don't appear).
   Added a peak-lab threshold cluster.
6. **Paths corrected** — every absolute path now points into
   `SAEM_2026_hackathon/derived/` instead of the old outer repo.

The 10 agents stay independent (each runs in its own context, no
cross-reads). The diversity-of-reasoning structure (Agents 1–5 vary
field weighting; 6–10 vary reasoning paradigm) is preserved — only the
substrate clinical knowledge is corrected.

---

## Validation plan (after agents rerun)

Once the 10 agents are re-spawned with v6 prompts and the new
features are wired into the training pipeline, validate alignment
with the manual ground truth via:

- **Per-class precision/recall vs `Task1_Two_Tier_Input_Data.csv` final_pred_label** — the manual file already contains organizer probabilities; compare `probs_avg.csv` argmax to `ground_truth_drug_name`.
- **Brier per class** — most direct calibration metric.
- **High-confidence rule audit** — for every encounter the v6
  hierarchy classifies via rules 1–4 (Kraken peak-lab anchors,
  diaphoretic+tachycardic, marked restlessness, agitation+escalation),
  check that the agent consensus assigns p_kraken ≥ 0.7. Same for
  rules 5–8 (Triton) and rules 9–13 (Coral).
- **Severity-Kraken correlation preserved** — high-severity rate
  among predicted Kraken should sit near 18/58 = 31% (current
  ground-truth distribution).
