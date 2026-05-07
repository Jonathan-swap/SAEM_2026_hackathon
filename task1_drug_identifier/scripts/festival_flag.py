"""
SAEM26 Hackathon — Task 1 Step 1.

Flags each patient as festival or community based on triage data only.
"""

import pandas as pd


# ---- 1. Paths ----
DATA_PATH = "data/Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = "task1_drug_identifier/out"


# ---- 2. Load triage data ----
triage = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
print("Triage rows:", len(triage))


# ---- 3. Festival flag rules ----
FESTIVAL_KEYWORDS = [
    "festival", "main stage", "side stage", "campground",
    "food court", "soaking man", "festival attendee",
]
COMMUNITY_KEYWORDS = ["community patient", "community"]


def classify(row):
    note = str(row["triage_brief_note"]).lower() if pd.notna(row["triage_brief_note"]) else ""

    # Rule 1: Community keyword in note overrides everything
    if any(kw in note for kw in COMMUNITY_KEYWORDS):
        return 0, "community_keyword"
    # Rule 2: Festival transfer mode
    if row["triage_mode_of_arrival"] == "Festival Medical Tent Transfer":
        return 1, "festival_transfer"
    # Rule 3: Festival keyword in note
    if any(kw in note for kw in FESTIVAL_KEYWORDS):
        return 1, "festival_keyword"
    # Default: community, but track these
    return 0, "unmatched"


results = triage.apply(classify, axis=1, result_type="expand")
results.columns = ["is_festival", "rule"]
flags = pd.concat([triage[["encounter_id"]], results], axis=1)


# ---- 4. Summary ----
print("\nClassification breakdown:")
print(flags["rule"].value_counts())
print(f"\nTotal festival: {int(flags['is_festival'].sum())} of {len(flags)}")

unmatched = flags.loc[flags["rule"] == "unmatched", "encounter_id"].tolist()
print(f"\nUnmatched encounters ({len(unmatched)}) defaulted to community:")
print(unmatched)


# ---- 5. Save ----
out_path = f"{OUT_DIR}/festival_flags.csv"
flags[["encounter_id", "is_festival"]].to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")