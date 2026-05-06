"""
SAEM26 Hackathon — Step 1 starter code.

Loads the three data sheets, takes a quick look, and flags which
patients are festival-related so later steps only model on those.
"""

import pandas as pd


# ---- 1. Load the data ----
# Three sheets: triage info, 4-hour info, final disposition.
DATA_PATH = "/data/Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = "/task1_drug_identifier/out"

triage = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
four_hour = pd.read_excel(DATA_PATH, sheet_name="Four_Hour_Data")
disposition = pd.read_excel(DATA_PATH, sheet_name="Disposition")

print("Triage rows:    ", len(triage))
print("Four-hour rows: ", len(four_hour))
print("Disposition rows:", len(disposition))
print("\nDisposition counts:")
print(disposition["encounter_disposition_label"].value_counts())


# ---- 2. Flag festival-related encounters ----
# Two signals: how the patient arrived, and what the HPI mentions.
# A patient is flagged as festival-related if EITHER signal hits.

FESTIVAL_ARRIVAL_MODES = {"Festival Medical Tent Transfer"}

FESTIVAL_KEYWORDS = [
    "festival",
    "main stage",
    "side stage",
    "campground",
    "food court",
    "soaking man",
]


def has_festival_keyword(text):
    """Return True if any festival keyword appears in the text."""
    if pd.isna(text):
        return False
    text = str(text).lower()
    return any(kw in text for kw in FESTIVAL_KEYWORDS)


# Signal 1: arrival mode (from triage sheet)
arrival_flag = triage["triage_mode_of_arrival"].isin(FESTIVAL_ARRIVAL_MODES)

# Signal 2: HPI keywords (from 4-hour sheet)
hpi_flag = four_hour["narrative_notes_structured_hpi"].apply(has_festival_keyword)

# Combine the two flags on encounter_id
flags = pd.DataFrame({
    "encounter_id": triage["encounter_id"],
    "arrival_flag": arrival_flag.values,
}).merge(
    pd.DataFrame({
        "encounter_id": four_hour["encounter_id"],
        "hpi_flag": hpi_flag.values,
    }),
    on="encounter_id",
)

flags["is_festival"] = flags["arrival_flag"] | flags["hpi_flag"]

print("\nFestival-related encounters:",
      int(flags["is_festival"].sum()), "of", len(flags))


# ---- 3. Save the result ----
# Simple two-column file ready for Step 2 (clustering on festival patients).
out_path = f"{OUT_DIR}/festival_flags.csv"
flags[["encounter_id", "is_festival"]].to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")