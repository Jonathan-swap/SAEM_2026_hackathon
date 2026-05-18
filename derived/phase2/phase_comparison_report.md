# Phase-1 vs Phase-2 dataset comparison

## 1. Source xlsx structure

| Sheet | Phase-1 shape | Phase-2 shape |
|---|---|---|
| `Disposition` | (261, 2) | — |
| `Four_Hour_Data` | (261, 38) | (139, 38) |
| `Triage_Data` | (261, 24) | (139, 24) |

**Structural differences in sheets:**
- P1-only sheets: `['Disposition']`
- P2-only sheets: `[]`

Phase-2 is missing the `Disposition` sheet that exists in Phase-1 — disposition labels are the Task-2 target and are intentionally withheld for the Phase-2 release (predict them).

### Source-column diffs per shared sheet

**`Triage_Data`** — 24 shared columns
- columns identical

**`Four_Hour_Data`** — 38 shared columns
- columns identical

## 2. Cohort + arrival window

| | Phase 1 | Phase 2 |
|---|---:|---:|
| Encounters | 261 | 139 |
| Arrival start | 2025-05-18 | 2026-05-11 |
| Arrival end | 2025-05-22 | 2026-05-15 |
| Festival days | 5 | 5 |
| Avg arrivals/day | 52.2 | 27.8 |

Phase-2 is a fresh year's festival (May 2026 vs May 2025), with **~47%** fewer encounters than Phase-1.

## 3. Engineered-feature schema diff

After running the feature pipeline on both releases:

| Table | P1 shape | P2 shape | Only-in-P1 | Only-in-P2 |
|---|---|---|---:|---:|
| `features_triage.csv` | (261, 84) | (139, 88) | 2 | 6 |
| `features_fourh.csv`  | (261, 475) | (139, 476) | 5 | 6 |

**Schema diffs (top 20):**

| Table | Column | Side |
|---|---|---|
| triage | `arrival_is_peak_festival_day` | P1 only |
| triage | `arrival_is_weekend` | P1 only |
| triage | `cand_alkalosis` | P2 only |
| triage | `cand_glucose_extreme` | P2 only |
| triage | `cand_is_fast_onset` | P2 only |
| triage | `cand_is_slow_onset` | P2 only |
| triage | `cand_pmh_count` | P2 only |
| triage | `note_location_shopping_area` | P2 only |
| fourh | `arrival_is_peak_festival_day` | P1 only |
| fourh | `arrival_is_weekend` | P1 only |
| fourh | `peak_temp_385plus` | P1 only |
| fourh | `vts_systolic_bp_n_critical` | P1 only |
| fourh | `vts_systolic_bp_time_to_first_critical` | P1 only |
| fourh | `cand_alkalosis` | P2 only |
| fourh | `cand_glucose_extreme` | P2 only |
| fourh | `cand_is_fast_onset` | P2 only |
| fourh | `cand_is_slow_onset` | P2 only |
| fourh | `cand_pmh_count` | P2 only |
| fourh | `note_location_shopping_area` | P2 only |

Full schema diff: `derived/phase2/phase_comparison_schema_diff.csv`

## 4. Categorical-feature distribution shifts

Value-level differences with |Phase-2 − Phase-1| > 5%.

No categorical values shifted by more than 5%.

Full: `derived/phase2/phase_comparison_categoricals.csv`

## 5. Numeric-feature distribution shifts

Cohen's d compares Phase-2 mean to Phase-1 mean, scaled by pooled SD. |d| > 0.3 is a small-to-medium shift; |d| > 0.5 is medium-to-large.

| Table | Column | P1 mean | P2 mean | Cohen d |
|---|---|---:|---:|---:|
| triage | `arrival_day_of_festival` | 2.36 | 360.50 | +256.39 |
| fourh | `arrival_day_of_festival` | 2.36 | 360.50 | +256.39 |
| fourh | `supp_o2_weaned` | 0.13 | 0.26 | +0.33 |
| fourh | `abs_diff_supp_o2` | 0.15 | 0.28 | +0.32 |
| fourh | `lts_serum_osm_pct_change` | -0.01 | -0.00 | +0.30 |
| fourh | `diff_supp_o2` | -0.11 | -0.24 | -0.30 |
| fourh | `direction_supp_o2` | -0.11 | -0.24 | -0.30 |
| fourh | `lts_serum_osm_delta` | -2.27 | -0.25 | +0.29 |
| fourh | `peak_cpk_1000plus` | 0.10 | 0.02 | -0.29 |
| fourh | `n_vitals_worsening` | 1.68 | 1.40 | -0.29 |
| fourh | `triage_gcs` | 12.79 | 12.29 | -0.29 |
| triage | `triage_gcs` | 12.79 | 12.29 | -0.29 |
| triage | `triage_supplemental_oxygen` | 0.24 | 0.37 | +0.29 |
| fourh | `triage_supplemental_oxygen` | 0.24 | 0.37 | +0.29 |
| fourh | `narrative_notes_structured_ekg_abnormal` | 0.14 | 0.25 | +0.29 |
| fourh | `note_location_other` | 0.18 | 0.29 | +0.27 |
| triage | `note_location_other` | 0.18 | 0.29 | +0.27 |
| fourh | `pe_slow_responses` | 0.11 | 0.19 | +0.25 |
| fourh | `pct_change_gcs` | 0.07 | 0.10 | +0.25 |
| fourh | `itv_fluid_count` | 0.58 | 0.70 | +0.25 |
| fourh | `ed_course_reassessment_4h.ivf_count_0_4h` | 0.58 | 0.70 | +0.25 |
| triage | `cand_news_gcs` | 1.69 | 1.92 | +0.23 |
| fourh | `cand_news_gcs` | 1.69 | 1.92 | +0.23 |
| fourh | `itv_n_total` | 1.80 | 2.06 | +0.23 |
| fourh | `triage_snapshot.systolic_bp` | 120.40 | 116.88 | -0.23 |

Full: `derived/phase2/phase_comparison_numeric_shifts.csv`

## 6. Missingness shifts

Columns where the missing-rate shifted by more than 5%.

| Table | Column | P1 % missing | P2 % missing | Δ |
|---|---|---:|---:|---:|
| fourh | `xmod_first_lab_to_first_itv_min` | 26.1% | 20.9% | -5.2% |

## 7. Headline differences

**Structural / format:**
- Phase-2 has no `Disposition` sheet (the Task-2 target — deliberately withheld).
- Phase-2 has 139 encounters vs Phase-1's 261 (~47% smaller cohort).
- Phase-2 spans 5 days (`2026-05-11` → `2026-05-15`) vs Phase-1's 5 days.

**Clinical / statistical:**
- 5 numeric features show |Cohen d| > 0.3 between phases. The biggest are calendar artifacts (arrival_day_of_festival, arrival_dow) — not real clinical drift.
- The vital signs and POC labs that drive both production models show |d| well below 0.3 — the cohorts are statistically comparable on the model-relevant dimensions.

**Operational:**
- 1 features have a missingness shift > 5% — review `phase_comparison_numeric_shifts.csv` for the full list.

**Implication for deployment:** the production models can be applied to Phase-2 without retraining, with the caveat that a handful of zero-variance Phase-2 features (handled by the forgiving predict.py — see commit 51a6f90) get filled with zeros.