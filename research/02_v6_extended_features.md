# v6 Extended Feature Catalogue (Round 2)

Source: [`data/toxidrome_report_v6.pdf`](../data/toxidrome_report_v6.pdf)
+ derived insights from running the v6 discriminator hierarchy against
the manually annotated outcomes in
[`Task1_Two_Tier_Input_Data.csv`](../data/Task1_Two_Tier_Input_Data.csv).

Round 1 ([`01_v6_features_and_prompts.md`](01_v6_features_and_prompts.md))
identified 32 direct features (binary PE tokens, peak-lab thresholds,
chief-complaint keywords). This round adds **39 additional features**
that v6 doesn't name explicitly but that follow from its discriminator
logic — composites, interactions, trajectories, and text-structure
signals.

Same leakage discipline as Round 1: Task-1 features must be derivable
from triage signals only; Task-2 features may use the 4h horizon.

---

## A. Physiologic composites (triage-time — Task 1 + Task 2)

These are derived from existing triage vitals; no narrative or 4h
data needed.

| # | Feature | Formula | Drug signal |
|---:|---|---|---|
| 1 | `pulse_pressure` | `SBP - DBP` | Kraken: wide PP from sympathetic surge |
| 2 | `mean_arterial_pressure` | `(SBP + 2*DBP) / 3` | Kraken: elevated MAP |
| 3 | `shock_index` | `HR / SBP` | Kraken severity (>1.0 = decompensation) |
| 4 | `rate_pressure_product` | `HR * SBP` | Kraken: demand-ischemia surrogate |
| 5 | `news2_total` | composite NEWS2 from HR/RR/SBP/SpO2/Temp/GCS | Kraken: high; Coral/Triton: low-moderate |
| 6 | `age_predicted_max_hr` | `220 - age` | Used as denominator for #7 |
| 7 | `tachy_relative_to_max` | `HR / age_predicted_max_hr` | Kraken: >0.7 supports; controls for pediatric cohort |
| 8 | `hyperdynamic_combo` | `HR>110 AND wide_PP (>50) AND temp>37.5` | Kraken composite |
| 9 | `respiratory_distress_combo` | `RR>22 AND SpO2<94` | Generic severity (Kraken/Triton lean) |
| 10 | `cardiopulmonary_uncoupling` | `HR/RR ratio > 6.0` | Kraken: tachycardia outpaces RR |

**Source columns**: `triage_heart_rate`, `triage_respiratory_rate`,
`triage_snapshot.systolic_bp`, `triage_snapshot.diastolic_bp`,
`triage_snapshot.oxygen_saturation`, `triage_temperature_c`,
`triage_gcs`, `triage_age`. All already in `features_triage.csv`.

---

## B. VBG / metabolic patterns (triage-time — Task 1 + Task 2)

Beyond the simple `AG > 20` and `pH > 7.35` rules from Round 1, v6's
discriminator table implies pattern-level metabolic signatures.

| # | Feature | Formula | Drug signal |
|---:|---|---|---|
| 11 | `metabolic_acidosis_pattern` | `pH<7.35 AND AG>16` | Kraken: lactic acidosis from rhabdo (rule #14 lite) |
| 12 | `respiratory_alkalosis_pattern` | `pH>7.45 AND RR>22` | Kraken: catecholamine-driven hyperventilation |
| 13 | `compensated_normal_pattern` | `7.35<pH<7.45 AND 8<AG<12` | Triton/Coral lean |
| 14 | `hypokalemia_flag` | `K+ < 3.5` | Kraken: catecholamine intracellular shift |
| 15 | `hyperkalemia_flag` | `K+ > 5.0` | Kraken severe: rhabdo-driven |
| 16 | `stress_hyperglycemia` | `glucose > 140 AND HR > 110` | Kraken: sympathomimetic + adrenergic |
| 17 | `na_extreme` | `Na < 132 OR Na > 148` | None lean (specific medical) |
| 18 | `hemoglobin_anemia_flag` | `Hgb < 11` (sex-adjusted) | None lean (specific medical) |

---

## C. 4-hour vital trajectory features (Task 2 only)

Beyond `peak_*` thresholds from Round 1, v6's PK profiles (rapid
resolution for Kraken, persistent slowing for Triton, gradual wave
for Coral) imply trajectory-shape features.

| # | Feature | Formula | Drug signal |
|---:|---|---|---|
| 19 | `vitals_normalization_4h` | count of {HR, RR, BP, Temp, SpO2} returned to normal by 4h | Kraken: high (rapid resolution); Triton: low (persistent) |
| 20 | `persistent_tachycardia_4h` | `vts_heart_rate_last30_mean > 100` | Triton with cardiac awareness; Kraken not yet resolved |
| 21 | `gcs_improvement_slope` | slope of GCS over 0-240 min | Triton: positive slope (sedation lifting); Kraken: stable; Coral: flat-then-up |
| 22 | `temp_resolution_time` | minute when temp first returned to <38°C after peak | Kraken-specific (others rarely hyperthermic) |
| 23 | `lactate_clearance_slope` | linear regression slope of lactate over 0-4h | Kraken: positive (rising) in severe; clearing if treated |
| 24 | `intervention_density_first_hour` | count of `itv_*` events with minute<60 | Kraken severity proxy |
| 25 | `escalation_velocity` | minutes from first benzo to next escalation step | Kraken if rapid; sedative-failure if Triton (rare) |
| 26 | `gcs_nadir_minute` | `vts_gcs_nadir_minute` | Triton: early (peak sedation); Coral: variable |
| 27 | `temperature_volatility` | `vts_temperature_c_range` | Kraken: high; Triton/Coral: stable |
| 28 | `hr_oscillation_count` | `stab_hr_oscillations` | Coral: waxing/waning; Kraken: monotonic surge |

---

## D. Text-derived features (HPI + triage_brief_note)

The Round 1 keyword flags covered defining tokens. Round 2 adds
density / structural / interaction signals.

| # | Feature | Formula | Drug signal |
|---:|---|---|---|
| 29 | `note_emotion_arousal_density` | count of arousal tokens (`agitated`, `racing`, `restless`, `combative`) / word_count | Kraken |
| 30 | `note_emotion_inward_density` | count of inward tokens (`slow`, `withdrawn`, `quiet`, `distant`) / word_count | Triton |
| 31 | `note_perceptual_density` | count of perceptual tokens (`wave`, `distort`, `altered`, `vivid`, `unreal`, `spatial`) / word_count | Coral |
| 32 | `note_onset_x_arousal` | `note_onset_minutes < 60 AND note_emotion_arousal_density > 0.05` | Kraken (rapid onset + agitation) |
| 33 | `note_onset_x_perceptual` | `60 < note_onset_minutes < 180 AND note_perceptual_density > 0.03` | Coral (medium onset + perception) |
| 34 | `note_onset_x_palpitations` | `note_onset_minutes > 60 AND keyword(palpitation)` | Triton (delayed cardiac awareness) |
| 35 | `note_first_sentence_signal` | binary: any defining token in the FIRST sentence of brief_note | Strong drug-positive (urgency-of-mention) |
| 36 | `note_drug_lexical_density` | total drug-suggestive tokens / total tokens | Drug-positive (vs None) |

---

## E. MDM-derived (post-boilerplate, Task 2 only)

The MDM contains 44% boilerplate but the remaining 56% has signal.
After stripping the boilerplate phrase, several features become
useful.

| # | Feature | Formula | Drug signal |
|---:|---|---|---|
| 37 | `mdm_length_minus_boilerplate` | char_count(mdm) minus length of stripped phrase | High: complex case → tox or specific dx; Low: pure-boilerplate-only MDM is wide prior |
| 38 | `mdm_alternative_dx_named` | binary: any keyword from {CAD, sepsis, appendicitis, sprain, UTI, stroke, MI, asthma, COPD, pneumonia, gastroenteritis} in stripped MDM | Strong **None** |
| 39 | `mdm_severity_tier` | categorical: low/moderate/high from MDM language (`low risk`, `moderate`, `high acuity`, `ICU candidate`) | High: Kraken-enriched (75% of high-severity = Kraken) |

---

## Implementation guidance

All 39 are **structural** — computable from columns already in the
xlsx or in `narratives.jsonl`. No external data needed.

**Where each feature plugs into the pipeline**:

| Feature group | Add to | Script |
|---|---|---|
| A. Physiologic composites (1–10) | both feature tables | new `src/features/extract_composites.py` or extend `eda_descriptive.py`'s candidate-feature block (they're a natural extension of the existing `cand_*` family — `cand_shock_index`, `cand_pulse_pressure`, `cand_news_total`, `cand_map`, `cand_rate_pressure_product` are already present; just need #6, #7, #8, #9, #10 added) |
| B. VBG patterns (11–18) | both feature tables | extend `cand_*` set in `eda_descriptive.py` |
| C. 4h trajectory (19–28) | `features_fourh.csv` only | extend `src/features/extract_time_features.py` (Groups F-G already compute similar metrics — extend them) |
| D. Text-derived (29–36) | both feature tables | extend `src/features/extract_note_features.py` |
| E. MDM-derived (37–39) | `features_fourh.csv` only | new `src/features/extract_mdm_features.py` (note: this DOES read MDM which is 4h-horizon — gated to Task 2; never lands in features_triage) |

**Estimated discriminator strength** (prior on the v6 hierarchy):

- High-priority (likely top-10 by MI): #3 shock_index, #5 news2_total,
  #11 metabolic_acidosis_pattern, #19 vitals_normalization_4h,
  #23 lactate_clearance_slope, #29 arousal_density, #30 inward_density,
  #31 perceptual_density, #35 first_sentence_signal,
  #38 mdm_alternative_dx_named.
- Medium-priority: composites that cross-reference vitals + labs
  (#8, #12, #15, #16) — useful for the boosted Kraken cluster.
- Low-priority but cheap: features #17–18, #26–28 — small lift each,
  may help in ambiguous-case tie-breaks.

**Total feature inventory after both rounds**:

```
Round 1 (research/01):  32 features (PE tokens, peak-lab thresholds, chief-complaint keywords)
Round 2 (this doc):     39 features (composites, trajectories, text density, MDM-derived)
                        ──────
                        71 features identified for boost
```

Current `features_triage.csv` is 74 columns. Adding the ~30 Round 1 +
Round 2 features that fit the triage horizon would push it to ~100.
`features_fourh.csv` is 438 columns; adding Round 1 + Round 2's
4h-side features would push to ~500. Both fit comfortably within
sklearn's column budget on a 261-row dataset (curse of dimensionality
is the real risk — feature selection via `cand_*`-style MI screening
will be necessary).

---

## Validation plan (when implemented)

1. Add features to the appropriate scripts.
2. Re-run `run_pipeline.py --skip-agents` (the full feature half +
   training).
3. Compare Task 1 / Task 2 macro AUC to current baselines (0.690 /
   0.926).
4. Use the existing `eda_advanced.py` MI ranking to identify which
   new features actually pull weight — drop the rest in
   `cleanup_features.py`'s near-constant pass.
5. Rerun `src/unsupervised/cluster.py` — expect improved purity in
   the Task 2 Kraken cluster (currently 69%) once trajectory and
   composite features are added.
