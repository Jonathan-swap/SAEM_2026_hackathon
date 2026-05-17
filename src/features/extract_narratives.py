"""Phase A.1 — Export narrative text per encounter to JSONL.

Reads the released xlsx, keeps only narrative + a few non-leaky
anchor fields, writes one JSON object per line to
`derived/narratives.jsonl`. This is the file each of the three
label-extraction subagents will read in Phase B.
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


def main() -> None:
    triage = pd.read_excel(XLSX, sheet_name="Triage_Data", engine="openpyxl")
    fourh = pd.read_excel(XLSX, sheet_name="Four_Hour_Data", engine="openpyxl")
    dispo = pd.read_excel(XLSX, sheet_name="Disposition", engine="openpyxl")

    df = (
        triage[[
            "encounter_id",
            "triage_chief_complaint",
            "triage_brief_note",
            "triage_mode_of_arrival",
        ]]
        .merge(fourh[["encounter_id", *NARRATIVE_FIELDS]], on="encounter_id")
        .merge(dispo, on="encounter_id")
    )

    out_path = OUT / "narratives.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
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
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} records to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
