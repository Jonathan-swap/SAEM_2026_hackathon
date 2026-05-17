"""
Step 0 — Data prep (step0_data_prep.py)
Inputs:

data/Hackathon_Data_Release_1_SHARE.xlsx — Triage_Data sheet (vitals, labs dict, brief note, chief complaint), Four_Hour_Data sheet (only used in disposition, not here)
data/tox_ground_truth_v5.csv — Yohan's labels (encounter_id, predicted_drug, severity_score)

Outputs:

Printed only (nothing saved to disk):

Cohort breakdown (drug vs no_drug counts)
Onset hypothesis check (% of each bucket with onset_minutes)
Exploratory KMeans diagnostics (silhouette across K=2..7) — not used downstream
Head of the feature matrix


The in-memory features dataframe (24 triage-only features per patient, all 261 patients) is the conceptual handoff to step 1
"""

import ast
import re
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---- Config ----
DATA_PATH = "data/Hackathon_Data_Release_1_SHARE.xlsx"
GT_PATH   = "data/tox_ground_truth_v5.csv"

ONSET_RE = re.compile(r"symptom onset\s+~?(\d+)\s+minutes?\s+before arrival",
                      re.IGNORECASE)

VITALS = [
    "triage_heart_rate", "triage_respiratory_rate",
    "triage_snapshot.systolic_bp", "triage_snapshot.diastolic_bp",
    "triage_snapshot.oxygen_saturation", "triage_temperature_c",
    "triage_gcs", "triage_pain_scale",
]
LAB_COLS = ["fingerstick_glucose", "ph", "sodium", "potassium",
            "hemoglobin", "anion_gap"]


# ---- Load ----
triage = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
gt     = pd.read_csv(GT_PATH)


# ---- Cohort identification: drug if in GT file, else no_drug ----
drug_ids = set(gt["encounter_id"])
triage["bucket"] = triage["encounter_id"].isin(drug_ids).map(
    {True: "drug", False: "no_drug"}
)
print("=== Cohort breakdown ===")
print(triage["bucket"].value_counts().to_string())


# ---- Triage-only feature extraction ----
def parse_labs(cell):
    if pd.isna(cell): return {}
    parsed = ast.literal_eval(cell)
    return parsed[0] if parsed else {}


def extract_onset(text):
    if pd.isna(text): return float("nan")
    m = ONSET_RE.search(str(text))
    return float(m.group(1)) if m else float("nan")


# 6 labs broken out from the dict column
labs = triage["triage.labs"].apply(parse_labs).apply(pd.Series)
labs["encounter_id"] = triage["encounter_id"].values
labs = labs[["encounter_id"] + LAB_COLS]

# Onset minutes from brief note
triage["onset_minutes"] = triage["triage_brief_note"].apply(extract_onset)

# Chief complaint one-hot
chief = pd.get_dummies(
    triage[["encounter_id", "triage_chief_complaint"]],
    columns=["triage_chief_complaint"], drop_first=True,
)

features = (
    triage[["encounter_id", "bucket", "onset_minutes"] + VITALS]
    .merge(labs,  on="encounter_id")
    .merge(chief, on="encounter_id")
    .merge(gt[["encounter_id", "predicted_drug"]], on="encounter_id", how="left")
)
feature_cols = [c for c in features.columns
                if c not in {"encounter_id", "bucket", "predicted_drug"}]


# ---- Validate Yohan's onset hypothesis ----
print("\n=== Onset time presence by bucket ===")
for b, g in features.groupby("bucket"):
    n_on = g["onset_minutes"].notna().sum()
    print(f"{b:>9}: {n_on:>3}/{len(g):>3} have onset time ({100*n_on/len(g):.1f}%)")


# ---- Exploratory clustering on drug patients ----
drug = features[features["bucket"] == "drug"].copy()
X = drug[feature_cols].fillna(drug[feature_cols].median(numeric_only=True))
X_scaled = StandardScaler().fit_transform(X)

print(f"\n=== Clustering {len(drug)} drug patients on {len(feature_cols)} features ===")
print(f"{'K':>3} {'inertia':>10} {'silhouette':>11}")
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    print(f"{k:>3} {km.inertia_:>10.1f} {silhouette_score(X_scaled, km.labels_):>11.3f}")

drug["cluster_k3"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled)
drug["cluster_k4"] = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X_scaled)


# ---- Preview ----
print("\n=== features (head) ===")
print(features.head().to_string())
print(f"Shape: {features.shape}")

print("\n=== drug w/ clusters (head, key cols) ===")
print(drug[["encounter_id", "bucket", "predicted_drug", "onset_minutes",
            "cluster_k3", "cluster_k4"]].head().to_string())
print(f"Shape: {drug.shape}")