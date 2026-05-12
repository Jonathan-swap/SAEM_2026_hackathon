"""
Count festival attendees without intoxication-related terms in their notes.
Two checks: (1) triage brief note only, (2) brief note + HPI from 4-hour data.
"""

import pandas as pd

DATA_PATH = "data/Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = "task1_drug_identifier/out"

INTOX_TERMS = [
    "intox", "drug", "overdose", "ingest", "substance", "tox",
    "altered", "stimulant", "hallucin", "psychoactive", "recreational",
    "ecstasy", "mdma", "lsd", "ketamine", "amphetamine", "opioid", "opiate",
]

triage    = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
four_hour = pd.read_excel(DATA_PATH, sheet_name="Four_Hour_Data")
flags     = pd.read_csv(f"{OUT_DIR}/festival_flags.csv")

festival_ids = flags.loc[flags["is_festival"] == 1, "encounter_id"]
brief = triage[triage["encounter_id"].isin(festival_ids)] \
    .set_index("encounter_id")["triage_brief_note"].fillna("")
hpi = four_hour[four_hour["encounter_id"].isin(festival_ids)] \
    .set_index("encounter_id")["narrative_notes_structured_hpi"].fillna("")


def has_intox(text):
    t = str(text).lower()
    return any(term in t for term in INTOX_TERMS)


match_brief    = brief.apply(has_intox)
match_combined = (brief + " " + hpi.reindex(brief.index).fillna("")).apply(has_intox)

print(f"Festival patients: {len(brief)}\n")
print(f"Check 1 (brief note only):   {(~match_brief).sum()} without intox terms")
print(f"Check 2 (brief note + HPI):  {(~match_combined).sum()} without intox terms")
print(f"\nCheck 1 missing IDs: {brief.index[~match_brief].tolist()}")
print(f"\nCheck 2 missing IDs: {brief.index[~match_combined].tolist()}")