"""Phase A.2 — Build the structured feature table.

Parses the released xlsx into a single wide feature matrix:
  - Triage vitals + demos + chief complaint + ESI + MH flags
  - Parsed triage point-of-care labs (iStat)
  - 4-hour reassessment vitals + labs + delta features
  - Intervention flag columns (already structured)
  - Imaging abnormal flags (already structured)
  - Aggregates from the embedded time-series JSON-repr blobs
    (vitals_timeseries, labs_timeseries, interventions)
  - Composite festival flag: mode_of_arrival OR keyword-in-note
  - Disposition label (Task 2 target)

Writes `derived/features.csv`. The Task-1 (drug) label is NOT in
this file — it comes from `derived_labels.csv` after Phase B.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "data" / "Hackathon_Data_Release_1_SHARE.xlsx"
OUT = ROOT / "derived"
OUT.mkdir(exist_ok=True)


FESTIVAL_KEYWORDS = re.compile(
    r"\b(?:festival|main stage|campground|music|soaking|tent|vortex|"
    r"poseidrium|leviathan|kraken|triton|coral)\b",
    re.IGNORECASE,
)


def safe_parse(s: object) -> list[dict]:
    """Parse a Python-repr list-of-dicts string. Empty/'[]' -> []."""
    if not isinstance(s, str) or not s.strip() or s.strip() == "[]":
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def parse_triage_labs(s: object) -> dict[str, float]:
    """triage.labs is a 1-element list with the iStat panel."""
    rows = safe_parse(s)
    if not rows:
        return {}
    r = rows[0]
    return {
        "triage_lab_glucose": r.get("fingerstick_glucose"),
        "triage_lab_ph": r.get("ph"),
        "triage_lab_sodium": r.get("sodium"),
        "triage_lab_potassium": r.get("potassium"),
        "triage_lab_hemoglobin": r.get("hemoglobin"),
        "triage_lab_anion_gap": r.get("anion_gap"),
    }


def aggregate_vitals_ts(s: object) -> dict[str, float]:
    """Aggregate the ed_course.vitals_timeseries blob into summary stats."""
    rows = safe_parse(s)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {"vts_n_obs": len(df)}
    for col in ["heart_rate", "respiratory_rate", "systolic_bp",
                "diastolic_bp", "oxygen_saturation", "temperature_c",
                "gcs", "end_tidal_co2"]:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        out[f"vts_{col}_min"] = series.min()
        out[f"vts_{col}_max"] = series.max()
        out[f"vts_{col}_mean"] = series.mean()
        out[f"vts_{col}_last"] = series.iloc[-1]
    if "supplemental_oxygen" in df.columns:
        out["vts_supp_o2_any"] = int(
            pd.to_numeric(df["supplemental_oxygen"], errors="coerce")
              .fillna(0).max() > 0)
    return out


def aggregate_labs_ts(s: object) -> dict[str, float]:
    """Aggregate ed_course.labs_timeseries into per-analyte summary stats."""
    rows = safe_parse(s)
    out: dict[str, float] = {"lts_n_obs": len(rows)}
    if not rows:
        return out
    df = pd.DataFrame(rows)
    for col in ["cbc_wbc", "bmp_sodium", "bmp_potassium", "bmp_bicarb",
                "lft_ast", "vbg_ph", "troponin", "lactate", "cpk",
                "esr", "crp", "poct_glucose", "serum_osm"]:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        out[f"lts_{col}_min"] = series.min()
        out[f"lts_{col}_max"] = series.max()
        out[f"lts_{col}_last"] = series.iloc[-1]
    for binc in ["hcg_positive", "ua_abnormal", "serum_tox_positive"]:
        if binc in df.columns:
            out[f"lts_{binc}_any"] = int(
                pd.to_numeric(df[binc], errors="coerce")
                  .fillna(0).max() > 0)
    return out


def aggregate_interventions(s: object) -> dict[str, float]:
    rows = safe_parse(s)
    out = {"itv_n_events": len(rows)}
    if not rows:
        return out
    # Frequency of common intervention keywords
    names = " | ".join(str(r.get("event_name", "")) for r in rows).lower()
    for kw in ["benzodiazepine", "fluid", "intubat", "naloxone",
               "flumazenil", "antipyretic", "monitor", "cool",
               "physostigmine", "vasopressor"]:
        out[f"itv_kw_{kw}"] = int(kw in names)
    return out


def main() -> None:
    triage = pd.read_excel(XLSX, sheet_name="Triage_Data", engine="openpyxl")
    fourh = pd.read_excel(XLSX, sheet_name="Four_Hour_Data", engine="openpyxl")
    dispo = pd.read_excel(XLSX, sheet_name="Disposition", engine="openpyxl")

    # ---- TRIAGE FEATURES (Task 1 inputs — no time leakage) ----
    # Keep all triage columns INCLUDING the brief_note text (written
    # at triage). Drop only the JSON-blob lab column; it gets parsed.
    triage_block = triage.drop(columns=["triage.labs"]).copy()
    triage_labs_parsed = pd.DataFrame(
        triage["triage.labs"].map(parse_triage_labs).tolist()
    )
    triage_block = pd.concat([triage_block, triage_labs_parsed], axis=1)

    # Festival flag — composite, derivable at triage (mode + note)
    note_lc = triage["triage_brief_note"].fillna("").astype(str)
    triage_block["is_festival_patient"] = (
        (triage["triage_mode_of_arrival"] == "Festival Medical Tent Transfer")
        | note_lc.str.contains(FESTIVAL_KEYWORDS)
    ).astype(int)
    triage_block["festival_note_keyword_hit"] = (
        note_lc.str.contains(FESTIVAL_KEYWORDS)
    ).astype(int)

    # `encounter_disposition_label` is the Task-2 target and is NOT
    # available at triage. Excluded from features_triage to make the
    # leakage protection structural (so naive `df.drop(["encounter_id"])`
    # cannot accidentally feed it into a Task-1 model).
    features_triage = triage_block

    triage_path = OUT / "features_triage.csv"
    features_triage.to_csv(triage_path, index=False)
    print(f"Wrote TRIAGE features: {features_triage.shape[0]} rows × "
          f"{features_triage.shape[1]} cols -> {triage_path}")

    # ---- 4-HOUR FEATURES (Task 2 inputs — minute 0 to 240) ----
    # Triage block + 4h reassessment block + ED-course aggregates.
    # Narrative notes are post-triage but also post-clinical-decision;
    # exclude them from features here (they're labels' source, not
    # model inputs). Keep imaging-abnormal flags (binary outcomes by
    # 4h).
    drop_4h = [
        "ed_course.vitals_timeseries",
        "ed_course.labs_timeseries",
        "ed_course.interventions",
        "ed_course.reassessment_4h.encounter_id",
        "narrative_notes_structured_brief_hpi",
        "narrative_notes_structured_hpi",
        "narrative_notes_structured_physical_exam_pertinent_positives",
        "narrative_notes_structured_mdm",
        "narrative_notes_structured_clinical_course",
        "narrative.notes_structured_ed_meds_procedures",
    ]
    fourh_struct = fourh.drop(columns=drop_4h).copy()

    vts = pd.DataFrame(fourh["ed_course.vitals_timeseries"]
                       .map(aggregate_vitals_ts).tolist())
    lts = pd.DataFrame(fourh["ed_course.labs_timeseries"]
                       .map(aggregate_labs_ts).tolist())
    itv = pd.DataFrame(fourh["ed_course.interventions"]
                       .map(aggregate_interventions).tolist())

    fourh_full = pd.concat(
        [fourh_struct.reset_index(drop=True),
         vts.reset_index(drop=True),
         lts.reset_index(drop=True),
         itv.reset_index(drop=True)],
        axis=1,
    )

    features_fourh = (
        triage_block
        .merge(fourh_full, on="encounter_id", how="left")
        .merge(dispo, on="encounter_id", how="left")
    )

    fourh_path = OUT / "features_fourh.csv"
    features_fourh.to_csv(fourh_path, index=False)
    print(f"Wrote 4-HOUR features: {features_fourh.shape[0]} rows × "
          f"{features_fourh.shape[1]} cols -> {fourh_path}")

    print(f"\nFestival flag positive rate: "
          f"{features_triage['is_festival_patient'].mean() * 100:.1f}%  "
          f"({features_triage['is_festival_patient'].sum()} of "
          f"{len(features_triage)})")
    print(f"  via mode_of_arrival only: "
          f"{(features_triage['triage_mode_of_arrival'] == 'Festival Medical Tent Transfer').sum()}")
    print(f"  via note keyword only:    "
          f"{features_triage['festival_note_keyword_hit'].sum()}")
    print(f"\nDisposition class counts (from features_fourh):\n"
          f"{features_fourh['encounter_disposition_label'].value_counts()}")

    # ---- Leakage sentinel ----
    forbidden_in_triage = [
        c for c in features_triage.columns
        if c.startswith(("ed_course", "narrative_notes_", "narrative.notes_"))
        or "_4h" in c or "delta_" in c
        or c == "encounter_disposition_label"
    ]
    assert not forbidden_in_triage, (
        f"Leakage: forbidden columns leaked into triage features: "
        f"{forbidden_in_triage}"
    )
    print("\nOK: triage feature set contains no "
          "ed_course/narrative/4h/delta columns.")


if __name__ == "__main__":
    main()
