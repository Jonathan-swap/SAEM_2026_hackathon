"""v6-derived features.

Adds three feature families informed by data/toxidrome_report_v6.pdf
and `research/01_v6_features_and_prompts.md`:

  Group 1 — PE structured tokens (Task-2 only)
      14 binary flags parsed from physical_exam_pertinent_positives
      + 2 high-PPV combos (Kraken: diaphoretic+tachycardic,
      Triton: reduced_tracking+slow_responses)

  Group 2 — Peak-lab + peak-vital threshold flags (Task-2 only)
      Derived from the existing lts_*_max_value / vts_*_max columns
      in features_fourh.csv. Captures v6 rule #1 (rhabdomyolysis
      anchors) and the "all peak labs near-normal" pattern.

  Group 3 — Triage-text discriminator keywords + densities
      (Task-1 + Task-2). Parses triage_brief_note +
      triage_chief_complaint for the v6 defining tokens (Kraken
      agitation/restlessness, Triton palpitations/ringing,
      Coral perceptual/spatial). Also emits arousal /
      inward / perceptual densities per word.

Idempotent: drops any prior v6_/pe_/peak_/chief_/note_dens_ columns
before merging so reruns produce stable outputs.

All features here are TASK-LEGAL by design:
  - Group 1 + Group 2 land only in features_fourh.csv.
  - Group 3 lands in both feature tables (parsed from triage-time
    text only).

The leakage sentinel in cleanup_features.py will reject anything
mis-routed.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
XLSX = ROOT / "data" / "Hackathon_Data_Release_1_SHARE.xlsx"

PE_COL = "narrative_notes_structured_physical_exam_pertinent_positives"

# The 14-token vocabulary confirmed against the source xlsx
# (every row non-null; semicolon-delimited).
PE_TOKENS = [
    "agitated", "ataxia", "diaphoretic", "distractible", "dry_mucosa",
    "fatigued_appearance", "intermittent_disorientation", "mild_tremor",
    "reduced_tracking", "restless", "slow_responses", "tachycardic",
    "tachypneic_effort", "unsteady_gait",
]

# v6 keyword vocabularies (case-insensitive substring search).
# Note: "(?:...)" is non-capturing.
KW_KRAKEN = re.compile(
    r"\b(?:agitat\w*|combative|uncontainable|restless\w*|impulsive|"
    r"police|security|escalat\w*|pressur\w*\s+speech|erratic)\b",
    re.IGNORECASE,
)
KW_TRITON_PALP = re.compile(
    r"\b(?:palpitation\w*|racing\s+heart|cardiac\s+aware\w*|"
    r"feel(?:s|ing)?\s+(?:my\s+|his\s+|her\s+|the\s+)?heart)\b",
    re.IGNORECASE,
)
KW_TRITON_EARS = re.compile(
    r"\b(?:ringing\b.*?\bear|tinnitus|auditory)\b",
    re.IGNORECASE,
)
KW_TRITON_SLOW = re.compile(
    r"\b(?:slow(?:ed)?\s+response|psychomotor\s+slow|withdrawn|"
    r"disengag\w*|somnolen\w*|staring|drift\w*|lethargic)\b",
    re.IGNORECASE,
)
KW_CORAL_PERCEPTUAL = re.compile(
    r"\b(?:perceptual|time\s+distort\w*|distort\w*\s+time|altered\s+"
    r"(?:reality|perception)|wave-like|vivid|unreal|hallucinat\w*)\b",
    re.IGNORECASE,
)
KW_CORAL_SPATIAL = re.compile(
    r"\b(?:unsteady|ataxi\w*|spatial\s+disorient\w*|blurry\s+vision|"
    r"dizzy)\b",
    re.IGNORECASE,
)

# Densities — broader bag-of-tokens, normalised by word count.
TOKENS_AROUSAL = re.compile(
    r"\b(?:agitat\w*|restless\w*|racing|combative|pressured|"
    r"erratic|impulsive|escalat\w*|fight\w*|thrash\w*)\b",
    re.IGNORECASE,
)
TOKENS_INWARD = re.compile(
    r"\b(?:slow\w*|withdrawn|disengag\w*|quiet|distant|somnolen\w*|"
    r"staring|lethargic|drowsy|sluggish)\b",
    re.IGNORECASE,
)
TOKENS_PERCEPTUAL = re.compile(
    r"\b(?:wave\w*|distort\w*|altered|vivid|unreal|hallucinat\w*|"
    r"spatial|blurry|perceptual|surreal|kaleidoscop\w*)\b",
    re.IGNORECASE,
)


# ---------- Group 1: PE structured tokens -----------------------------

def parse_pe(cell: object) -> dict[str, int]:
    """Return one binary per token in PE_TOKENS plus the two combos."""
    out = {f"pe_{t}": 0 for t in PE_TOKENS}
    if isinstance(cell, str) and cell.strip():
        present = {tok.strip().lower() for tok in cell.split(";")
                   if tok.strip()}
        for t in PE_TOKENS:
            if t in present:
                out[f"pe_{t}"] = 1
    # High-PPV combos from v6
    out["pe_kraken_combo"] = int(out["pe_diaphoretic"]
                                  and out["pe_tachycardic"])
    out["pe_triton_combo"] = int(out["pe_reduced_tracking"]
                                  and out["pe_slow_responses"])
    return out


# ---------- Group 2: peak-lab / peak-vital threshold flags -----------

def peak_threshold_flags(df_fourh: pd.DataFrame) -> pd.DataFrame:
    """Derive v6 rule-1 anchors + the all-normal pattern.

    Reads existing lts_*_max_value / vts_*_max columns; emits one
    row per encounter with peak_* binary flags.
    """
    out = pd.DataFrame({"encounter_id": df_fourh["encounter_id"]})

    def safe_gt(col: str, threshold: float) -> pd.Series:
        if col not in df_fourh.columns:
            return pd.Series(0, index=df_fourh.index, dtype=int)
        return (pd.to_numeric(df_fourh[col], errors="coerce")
                  .fillna(-np.inf) > threshold).astype(int)

    def safe_lt(col: str, threshold: float) -> pd.Series:
        if col not in df_fourh.columns:
            return pd.Series(0, index=df_fourh.index, dtype=int)
        return (pd.to_numeric(df_fourh[col], errors="coerce")
                  .fillna(np.inf) < threshold).astype(int)

    # v6 rule-1 Kraken anchors (peak-lab thresholds)
    out["peak_lactate_5plus"] = safe_gt("lts_lactate_max_value", 5.0)
    out["peak_cpk_1000plus"] = safe_gt("lts_cpk_max_value", 1000.0)
    out["peak_troponin_015plus"] = safe_gt("lts_troponin_max_value",
                                             0.15)
    out["peak_hr_150plus"] = safe_gt("vts_heart_rate_max", 150)
    out["peak_temp_385plus"] = safe_gt("vts_temperature_c_max", 38.5)

    # Composite: kraken_severity_anchor — v6 rule #1 OR rule #14
    out["kraken_severity_anchor"] = (
        out["peak_lactate_5plus"]
        | out["peak_cpk_1000plus"]
        | out["peak_troponin_015plus"]
        | out["peak_hr_150plus"]
        | out["peak_temp_385plus"]
    ).astype(int)

    # All peak labs near-normal — supports Triton/Coral
    near_normal_cpk = safe_lt("lts_cpk_max_value", 200)
    near_normal_lac = safe_lt("lts_lactate_max_value", 1.5)
    near_normal_trop = safe_lt("lts_troponin_max_value", 0.05)
    out["all_peak_labs_normal"] = (near_normal_cpk
                                    & near_normal_lac
                                    & near_normal_trop).astype(int)

    return out


# ---------- Group 3: triage text keywords + densities ----------------

def parse_triage_text(chief: str, brief: str) -> dict[str, float]:
    chief = chief if isinstance(chief, str) else ""
    brief = brief if isinstance(brief, str) else ""
    combined = f"{chief}\n{brief}"
    word_count = max(len(combined.split()), 5)  # smooth denominator

    out: dict[str, float] = {}
    out["triage_chief_agitation"] = int(bool(KW_KRAKEN.search(combined)))
    out["triage_chief_palpitations"] = int(
        bool(KW_TRITON_PALP.search(combined)))
    out["triage_chief_ringing_ears"] = int(
        bool(KW_TRITON_EARS.search(combined)))
    out["triage_chief_psychomotor_slow"] = int(
        bool(KW_TRITON_SLOW.search(combined)))
    out["triage_chief_perceptual"] = int(
        bool(KW_CORAL_PERCEPTUAL.search(combined)))
    out["triage_chief_spatial"] = int(
        bool(KW_CORAL_SPATIAL.search(combined)))

    # Densities (per-word counts, smoothed)
    n_arousal = len(TOKENS_AROUSAL.findall(combined))
    n_inward = len(TOKENS_INWARD.findall(combined))
    n_perceptual = len(TOKENS_PERCEPTUAL.findall(combined))
    out["note_arousal_density"] = n_arousal / word_count
    out["note_inward_density"] = n_inward / word_count
    out["note_perceptual_density"] = n_perceptual / word_count

    return out


# ---------- Triage threshold flags (Task-1 legal) --------------------

def triage_threshold_flags(df_triage: pd.DataFrame) -> pd.DataFrame:
    """Threshold-based binaries derivable at minute 0 of arrival."""
    out = pd.DataFrame({"encounter_id": df_triage["encounter_id"]})

    def col_or_zero(c: str) -> pd.Series:
        if c not in df_triage.columns:
            return pd.Series(np.nan, index=df_triage.index)
        return pd.to_numeric(df_triage[c], errors="coerce")

    ag = col_or_zero("triage_lab_anion_gap")
    ph = col_or_zero("triage_lab_ph")
    hr = col_or_zero("triage_heart_rate")
    rr = col_or_zero("triage_respiratory_rate")
    temp = col_or_zero("triage_temperature_c")
    glucose = col_or_zero("triage_lab_glucose")

    out["triage_ag_above_20"] = (ag > 20).fillna(False).astype(int)
    out["triage_ag_below_12"] = (ag < 12).fillna(False).astype(int)
    out["triage_ph_above_735"] = (ph > 7.35).fillna(False).astype(int)
    out["triage_hr_above_120"] = (hr > 120).fillna(False).astype(int)
    out["triage_temp_above_38"] = (temp > 38.0).fillna(False).astype(int)
    out["triage_glucose_above_140"] = (
        (glucose > 140).fillna(False).astype(int))
    out["triage_sympathomimetic_combo"] = (
        ((hr > 110) & (rr > 22) & (temp > 37.5))
        .fillna(False).astype(int))
    return out


# ---------- Orchestration --------------------------------------------

V6_PREFIXES_TRIAGE = ("triage_chief_", "note_arousal_density",
                       "note_inward_density", "note_perceptual_density",
                       "triage_ag_above_", "triage_ph_above_",
                       "triage_hr_above_", "triage_temp_above_",
                       "triage_glucose_above_",
                       "triage_sympathomimetic_combo")
V6_PREFIXES_FOURH_ONLY = ("pe_", "peak_lactate_", "peak_cpk_",
                           "peak_troponin_", "peak_hr_", "peak_temp_",
                           "kraken_severity_anchor",
                           "all_peak_labs_normal")


def _drop_prior(df: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    drop = [c for c in df.columns
            if any(c.startswith(p) or c == p for p in prefixes)]
    if drop:
        df = df.drop(columns=drop)
    return df


def main() -> None:
    print("Reading xlsx + existing feature tables...")
    triage_xlsx = pd.read_excel(XLSX, sheet_name="Triage_Data",
                                  engine="openpyxl")
    fourh_xlsx = pd.read_excel(XLSX, sheet_name="Four_Hour_Data",
                                engine="openpyxl")
    triage = pd.read_csv(DERIVED / "features_triage.csv")
    fourh = pd.read_csv(DERIVED / "features_fourh.csv")

    # ---- Group 1: PE binaries (Task-2 only) ----
    print("\nGroup 1 — PE binaries from physical_exam_pertinent_positives")
    pe_rows = [parse_pe(v) for v in fourh_xlsx[PE_COL]]
    pe_df = pd.DataFrame(pe_rows)
    pe_df.insert(0, "encounter_id", fourh_xlsx["encounter_id"].values)
    # Coverage diagnostics
    print(f"  PE rows non-null: {fourh_xlsx[PE_COL].notna().sum()} / "
          f"{len(fourh_xlsx)}")
    for col in [f"pe_{t}" for t in PE_TOKENS] + [
            "pe_kraken_combo", "pe_triton_combo"]:
        n_hits = int(pe_df[col].sum())
        print(f"    {col:42s}  positives = {n_hits:>3d} "
              f"({n_hits/len(pe_df)*100:.1f}%)")

    # ---- Group 2: peak-threshold flags (Task-2 only) ----
    print("\nGroup 2 — peak-lab / peak-vital thresholds")
    peak_df = peak_threshold_flags(fourh)
    for col in peak_df.columns[1:]:
        n_hits = int(peak_df[col].sum())
        print(f"    {col:30s}  positives = {n_hits:>3d} "
              f"({n_hits/len(peak_df)*100:.1f}%)")

    # ---- Group 3: triage text features (Task-1 + Task-2) ----
    print("\nGroup 3a — triage text keywords + densities")
    text_rows = [parse_triage_text(c, b)
                 for c, b in zip(triage_xlsx["triage_chief_complaint"],
                                  triage_xlsx["triage_brief_note"])]
    text_df = pd.DataFrame(text_rows)
    text_df.insert(0, "encounter_id", triage_xlsx["encounter_id"].values)
    for col in text_df.columns[1:]:
        s = text_df[col]
        if s.dtype.kind in "iu" or set(s.unique()) <= {0, 1}:
            n_hits = int(s.sum())
            print(f"    {col:32s}  positives = {n_hits:>3d} "
                  f"({n_hits/len(text_df)*100:.1f}%)")
        else:
            print(f"    {col:32s}  mean = {s.mean():.4f}  "
                  f"max = {s.max():.3f}")

    print("\nGroup 3b — triage threshold flags")
    thr_df = triage_threshold_flags(triage)
    for col in thr_df.columns[1:]:
        n_hits = int(thr_df[col].sum())
        print(f"    {col:32s}  positives = {n_hits:>3d} "
              f"({n_hits/len(thr_df)*100:.1f}%)")

    # ---- Merge: features_triage ← Group 3a + 3b ----
    triage = _drop_prior(triage, V6_PREFIXES_TRIAGE)
    triage = (triage.merge(text_df, on="encounter_id", how="left")
                      .merge(thr_df, on="encounter_id", how="left"))
    triage.to_csv(DERIVED / "features_triage.csv", index=False)
    print(f"\nfeatures_triage.csv: {triage.shape}")

    # ---- Merge: features_fourh ← Group 1 + 2 + 3a + 3b ----
    fourh = _drop_prior(fourh, V6_PREFIXES_TRIAGE)
    fourh = _drop_prior(fourh, V6_PREFIXES_FOURH_ONLY)
    fourh = (fourh.merge(pe_df, on="encounter_id", how="left")
                    .merge(peak_df, on="encounter_id", how="left")
                    .merge(text_df, on="encounter_id", how="left")
                    .merge(thr_df, on="encounter_id", how="left"))
    fourh.to_csv(DERIVED / "features_fourh.csv", index=False)
    print(f"features_fourh.csv:  {fourh.shape}")


if __name__ == "__main__":
    main()
