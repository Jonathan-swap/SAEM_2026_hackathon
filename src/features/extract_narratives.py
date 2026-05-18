"""Phase A.1 — Export narrative text per encounter to JSONL.

Writes TWO files so the triage-horizon and 4-hour-horizon views are
structurally separated (mirrors `extract_structured.py`'s
`features_triage.csv` / `features_fourh.csv` split):

  - `derived/narratives_triage.jsonl` — triage-horizon only.
    Fields written at minute 0 (Triage_Data sheet): chief complaint,
    triage brief note, mode of arrival. Safe to feed into any
    triage-horizon model.
  - `derived/narratives_fourh.jsonl` — full record: triage fields
    PLUS the 4-hour narrative blocks (HPI, PE positives, MDM, clinical
    course, ED meds/procedures). Consumed by the label-extraction
    subagents in Phase B.

A leakage sentinel at the end asserts the triage file contains no
4h-derived columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "data" / "Hackathon_Data_Release_1_SHARE.xlsx"
OUT = ROOT / "derived"
OUT.mkdir(exist_ok=True)


NARRATIVE_FIELDS = [
    "narrative_notes_structured_brief_hpi",
    "narrative_notes_structured_hpi",
    "narrative_notes_structured_physical_exam_pertinent_positives",
    "narrative_notes_structured_mdm",
    "narrative_notes_structured_clinical_course",
    "narrative.notes_structured_ed_meds_procedures",
]

TRIAGE_ONLY_KEYS = {
    "encounter_id",
    "chief_complaint",
    "triage_brief_note",
    "mode_of_arrival",
    "disposition",
}

FORBIDDEN_TRIAGE_PREFIXES = ("narrative_notes_", "narrative.notes_")
FORBIDDEN_TRIAGE_KEYS = {
    "brief_hpi",
    "hpi",
    "physical_exam_pertinent_positives",
    "mdm",
    "clinical_course",
    "ed_meds_procedures",
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    triage = pd.read_excel(XLSX, sheet_name="Triage_Data", engine="openpyxl")
    fourh = pd.read_excel(XLSX, sheet_name="Four_Hour_Data", engine="openpyxl")
    dispo = pd.read_excel(XLSX, sheet_name="Disposition", engine="openpyxl")

    triage_block = triage[[
        "encounter_id",
        "triage_chief_complaint",
        "triage_brief_note",
        "triage_mode_of_arrival",
    ]].merge(dispo, on="encounter_id")

    triage_records: list[dict] = []
    for _, row in triage_block.iterrows():
        triage_records.append({
            "encounter_id": row["encounter_id"],
            "chief_complaint": row["triage_chief_complaint"],
            "triage_brief_note": row["triage_brief_note"],
            "mode_of_arrival": row["triage_mode_of_arrival"],
            "disposition": row["encounter_disposition_label"],
        })

    triage_path = OUT / "narratives_triage.jsonl"
    _write_jsonl(triage_path, triage_records)

    fourh_block = (
        triage_block
        .merge(fourh[["encounter_id", *NARRATIVE_FIELDS]], on="encounter_id")
    )

    fourh_records: list[dict] = []
    for _, row in fourh_block.iterrows():
        fourh_records.append({
            "encounter_id": row["encounter_id"],
            "chief_complaint": row["triage_chief_complaint"],
            "triage_brief_note": row["triage_brief_note"],
            "mode_of_arrival": row["triage_mode_of_arrival"],
            "brief_hpi": row["narrative_notes_structured_brief_hpi"],
            "hpi": row["narrative_notes_structured_hpi"],
            "physical_exam_pertinent_positives":
                row["narrative_notes_structured_physical_exam_pertinent_positives"],
            "mdm": row["narrative_notes_structured_mdm"],
            "clinical_course": row["narrative_notes_structured_clinical_course"],
            "ed_meds_procedures": row["narrative.notes_structured_ed_meds_procedures"],
            "disposition": row["encounter_disposition_label"],
        })

    fourh_path = OUT / "narratives_fourh.jsonl"
    _write_jsonl(fourh_path, fourh_records)

    # ---- Leakage sentinel: triage file must be free of 4h-derived keys ----
    if triage_records:
        sample_keys = set(triage_records[0].keys())
        forbidden_hits = (
            (sample_keys & FORBIDDEN_TRIAGE_KEYS)
            | {k for k in sample_keys
               if k.startswith(FORBIDDEN_TRIAGE_PREFIXES)}
        )
        assert not forbidden_hits, (
            f"Leakage: 4h-derived keys leaked into "
            f"narratives_triage.jsonl: {sorted(forbidden_hits)}"
        )
        extra = sample_keys - TRIAGE_ONLY_KEYS
        assert not extra, (
            f"Leakage: unexpected keys in narratives_triage.jsonl: "
            f"{sorted(extra)}"
        )

    print(f"Wrote {len(triage_records)} records to {triage_path}")
    print(f"  file size: {triage_path.stat().st_size / 1024:.1f} KB")
    print(f"Wrote {len(fourh_records)} records to {fourh_path}")
    print(f"  file size: {fourh_path.stat().st_size / 1024:.1f} KB")
    print("OK: narratives_triage.jsonl contains no 4h/narrative keys.")


if __name__ == "__main__":
    main()
