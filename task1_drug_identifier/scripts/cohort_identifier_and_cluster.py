"""
New cohort rules + clustering on included patients.

Precedence:
  1. serum_tox_positive == 1                           -> EXCLUDE
  2. MDM contains "acute undifferentiated..." phrase   -> INCLUDE
  3. Festival Medical Tent Transfer                    -> INCLUDE
  4. Festival keyword in brief note OR HPI             -> INCLUDE
  5. "community patient" in brief note                 -> EXCLUDE
  6. Otherwise                                         -> REVIEW
"""

import ast
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = "task1_drug_identifier/out"

FESTIVAL_KEYWORDS = ["festival", "main stage", "side stage", "campground",
                     "food court", "soaking man", "festival attendee"]
MDM_TARGET = "acute undifferentiated festival-related tox-metabolic presentation"


# ---- Load + merge needed fields ----
triage    = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
four_hour = pd.read_excel(DATA_PATH, sheet_name="Four_Hour_Data")

combined = triage[["encounter_id", "triage_brief_note", "triage_mode_of_arrival"]].merge(
    four_hour[["encounter_id", "ed_course.labs_timeseries",
               "narrative_notes_structured_mdm",
               "narrative_notes_structured_hpi"]],
    on="encounter_id", how="left",
).fillna({"triage_brief_note": "",
          "narrative_notes_structured_mdm": "",
          "narrative_notes_structured_hpi": ""})


def has_positive_tox(cell):
    """Parse labs_timeseries and check for serum_tox_positive == 1."""
    if pd.isna(cell): return False
    try:
        parsed = ast.literal_eval(str(cell))
    except Exception:
        return False
    if isinstance(parsed, dict):
        return parsed.get("serum_tox_positive") == 1
    if isinstance(parsed, list):
        return any(d.get("serum_tox_positive") == 1 for d in parsed if isinstance(d, dict))
    return False


def classify(row):
    brief = str(row["triage_brief_note"]).lower()
    mdm   = str(row["narrative_notes_structured_mdm"]).lower()
    hpi   = str(row["narrative_notes_structured_hpi"]).lower()
    mode  = row["triage_mode_of_arrival"]

    if has_positive_tox(row["ed_course.labs_timeseries"]):
        return "exclude", "tox_positive"
    if MDM_TARGET in mdm:
        return "include", "mdm_match"
    if mode == "Festival Medical Tent Transfer":
        return "include", "festival_transfer"
    if any(kw in brief or kw in hpi for kw in FESTIVAL_KEYWORDS):
        return "include", "festival_keyword"
    if "community patient" in brief:
        return "exclude", "community_keyword"
    return "review", "unmatched"


classified = combined.apply(classify, axis=1, result_type="expand")
classified.columns = ["bucket", "reason"]
results = pd.concat([combined[["encounter_id"]], classified], axis=1)

print("=== Cohort breakdown ===")
print(results.groupby(["bucket", "reason"]).size().to_string())
print(f"\nReview IDs: {results.loc[results['bucket']=='review', 'encounter_id'].tolist()}")

# results[results["bucket"]=="include"][["encounter_id", "reason"]].to_csv(f"{OUT_DIR}/cohort_include.csv", index=False)
# results[results["bucket"]=="exclude"][["encounter_id", "reason"]].to_csv(f"{OUT_DIR}/cohort_exclude.csv", index=False)
# results[results["bucket"]=="review"][["encounter_id", "reason"]].to_csv(f"{OUT_DIR}/cohort_review.csv", index=False)


# ---- Cluster on included patients ----
include_ids = results.loc[results["bucket"]=="include", "encounter_id"]
triage_inc    = triage[triage["encounter_id"].isin(include_ids)].copy()
four_hour_inc = four_hour[four_hour["encounter_id"].isin(include_ids)].copy()

vitals_cols = [
    "triage_heart_rate", "triage_respiratory_rate",
    "triage_snapshot.systolic_bp", "triage_snapshot.diastolic_bp",
    "triage_snapshot.oxygen_saturation", "triage_supplemental_oxygen",
    "triage_temperature_c", "triage_gcs", "triage_pain_scale",
]
vitals = triage_inc[["encounter_id"] + vitals_cols].copy()

def parse_labs(cell):
    if pd.isna(cell): return {}
    parsed = ast.literal_eval(cell)
    return parsed[0] if parsed else {}

labs = triage_inc["triage.labs"].apply(parse_labs).apply(pd.Series)
labs["encounter_id"] = triage_inc["encounter_id"].values
lab_cols = ["fingerstick_glucose", "ph", "sodium", "potassium", "hemoglobin", "anion_gap"]
labs = labs[["encounter_id"] + lab_cols]

exam = four_hour_inc[["encounter_id",
                      "narrative_notes_structured_physical_exam_pertinent_positives"]].copy()
exam.columns = ["encounter_id", "exam_text"]
exam["exam_text"] = exam["exam_text"].fillna("").astype(str)
exam_dummies = exam["exam_text"].str.get_dummies(sep=";")
exam_dummies.columns = [f"exam_{c.strip()}" for c in exam_dummies.columns]
exam = pd.concat([exam[["encounter_id"]], exam_dummies], axis=1)

features = vitals.merge(labs, on="encounter_id").merge(exam, on="encounter_id")
feature_cols = [c for c in features.columns if c != "encounter_id"]
X_scaled = StandardScaler().fit_transform(features[feature_cols])

print(f"\nIncluded patients clustered: {len(features)}")
print("\n=== Elbow + silhouette ===")
print(f"{'K':>3} {'inertia':>10} {'silhouette':>11}")
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    print(f"{k:>3} {km.inertia_:>10.1f} {silhouette_score(X_scaled, km.labels_):>11.3f}")

features["cluster_k3"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled)
features["cluster_k4"] = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X_scaled)
print("\nK=3 cluster sizes:")
print(features["cluster_k3"].value_counts().sort_index().to_string())
print("\nK=4 cluster sizes:")
print(features["cluster_k4"].value_counts().sort_index().to_string())


# ---- Review df for resident ----
review_df = features[["encounter_id", "cluster_k3", "cluster_k4"]].merge(
    results[["encounter_id", "reason"]], on="encounter_id",
).merge(
    triage[["encounter_id", "triage_brief_note"]], on="encounter_id",
).merge(
    four_hour[["encounter_id",
               "narrative_notes_structured_mdm",
               "narrative_notes_structured_hpi"]], on="encounter_id",
)

review_path = f"{OUT_DIR}/cluster_review_for_resident.csv"
review_df.to_csv(review_path, index=False)
print(f"\nSaved: {review_path}")