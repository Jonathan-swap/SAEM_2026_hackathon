"""
SAEM26 Hackathon — Step 2 starter code.

Clusters festival patients into 3 groups using triage vitals, triage labs,
and physical exam findings. The hope is that the 3 clusters correspond
to the 3 drugs (Kraken, Triton, Coral) — we map cluster -> drug name
afterward by inspecting each cluster's profile.
"""

import ast
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---- 1. Load data + festival flags from Step 1 ----
DATA_PATH = "/data/Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = "/task1_drug_identifier/out"

triage = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
four_hour = pd.read_excel(DATA_PATH, sheet_name="Four_Hour_Data")
flags = pd.read_csv(f"{OUT_DIR}/festival_flags.csv")

# Keep only festival patients
festival_ids = flags.loc[flags["is_festival"], "encounter_id"]
triage = triage[triage["encounter_id"].isin(festival_ids)].copy()
four_hour = four_hour[four_hour["encounter_id"].isin(festival_ids)].copy()

print("Festival patients:", len(triage))


# ---- 2. Pull triage vitals ----
vitals_cols = [
    "triage_heart_rate",
    "triage_respiratory_rate",
    "triage_snapshot.systolic_bp",
    "triage_snapshot.diastolic_bp",
    "triage_snapshot.oxygen_saturation",
    "triage_temperature_c",
    "triage_gcs",
    "triage_pain_scale",
]
vitals = triage[["encounter_id"] + vitals_cols].copy()


# ---- 3. Parse triage labs ----
# Each cell is a string like "[{'ph': 7.4, 'sodium': 140, ...}]" — convert to columns.
def parse_labs(cell):
    if pd.isna(cell):
        return {}
    parsed = ast.literal_eval(cell)  # safely evaluate the string
    return parsed[0] if parsed else {}

labs = triage["triage.labs"].apply(parse_labs).apply(pd.Series)
labs["encounter_id"] = triage["encounter_id"].values

lab_cols = ["fingerstick_glucose", "ph", "sodium", "potassium",
            "hemoglobin", "anion_gap"]
labs = labs[["encounter_id"] + lab_cols]


# ---- 4. One-hot encode physical exam findings ----
# Tokens are semicolon-separated, e.g. "diaphoretic;restless;tachycardic".
exam = four_hour[["encounter_id",
                  "narrative_notes_structured_physical_exam_pertinent_positives"]].copy()
exam.columns = ["encounter_id", "exam_text"]

exam["exam_text"] = exam["exam_text"].fillna("").astype(str)
exam_dummies = exam["exam_text"].str.get_dummies(sep=";")
exam_dummies.columns = [f"exam_{c.strip()}" for c in exam_dummies.columns]
exam = pd.concat([exam[["encounter_id"]], exam_dummies], axis=1)


# ---- 5. Build the feature matrix ----
features = vitals.merge(labs, on="encounter_id").merge(exam, on="encounter_id")
feature_cols = [c for c in features.columns if c != "encounter_id"]

# Checked and found no missingness in vitals, labs, and exam findings.
X = features[feature_cols]


# ---- 6. Standardize and cluster ----
# Standardizing puts everything on the same scale so KMeans isn't dominated
# by big-number features (like BP) over small ones (like pH).
X_scaled = StandardScaler().fit_transform(X)

# Elbow method: try K=2..7 and see how much "inertia" (within-cluster spread)
# drops at each step. Look for the bend in the curve — that's a good K.
# Silhouette score also reported (higher = better-separated clusters, max 1.0).
from sklearn.metrics import silhouette_score

print("\nElbow method (lower inertia = tighter clusters):")
print(f"{'K':>3} {'inertia':>10} {'silhouette':>12}")
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_)
    print(f"{k:>3} {km.inertia_:>10.1f} {sil:>12.3f}")

NUM_CLUSTERS = 3   # don't hard-code this elsewhere — easy to change for the twist
kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=10, random_state=42)
features["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster sizes:")
print(features["cluster"].value_counts().sort_index())


# ---- 7. Profile each cluster ----
# Print the mean of each feature by cluster so we can match clusters to drugs.
profile = features.groupby("cluster")[feature_cols].mean().round(2)
print("\nCluster profiles (mean of each feature):")
print(profile.T)

# Save profile to a txt file for later reference.
profile_path = f"{OUT_DIR}/cluster_profile.txt"
profile.T.to_csv(profile_path, sep="\t")
print(f"\nSaved: {profile_path}")


# ---- 8. Save cluster assignments ----
labels_path = f"{OUT_DIR}/cluster_labels.csv"
features[["encounter_id", "cluster"]].to_csv(labels_path, index=False)
print(f"\nSaved: {labels_path}")