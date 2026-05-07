"""
SAEM26 Hackathon — Task 1 cluster validation.
Run from repo root: python task1_drug_identifier/scripts/cluster_validation.py
"""

import ast
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_selection import f_classif
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score)
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = "task1_drug_identifier/out"
K_FINAL = 3


# ---- Load + filter to festival ----
triage    = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
four_hour = pd.read_excel(DATA_PATH, sheet_name="Four_Hour_Data")
flags     = pd.read_csv(f"{OUT_DIR}/festival_flags.csv")

festival_ids = flags.loc[flags["is_festival"] == 1, "encounter_id"]
triage    = triage[triage["encounter_id"].isin(festival_ids)].copy()
four_hour = four_hour[four_hour["encounter_id"].isin(festival_ids)].copy()


# ---- Build feature matrix ----
vitals_cols = [
    "triage_heart_rate", "triage_respiratory_rate",
    "triage_snapshot.systolic_bp", "triage_snapshot.diastolic_bp",
    "triage_snapshot.oxygen_saturation", "triage_supplemental_oxygen",
    "triage_temperature_c", "triage_gcs", "triage_pain_scale",
]
vitals = triage[["encounter_id"] + vitals_cols].copy()

def parse_labs(cell):
    if pd.isna(cell): return {}
    parsed = ast.literal_eval(cell)
    return parsed[0] if parsed else {}

labs = triage["triage.labs"].apply(parse_labs).apply(pd.Series)
labs["encounter_id"] = triage["encounter_id"].values
lab_cols = ["fingerstick_glucose", "ph", "sodium", "potassium", "hemoglobin", "anion_gap"]
labs = labs[["encounter_id"] + lab_cols]

exam = four_hour[["encounter_id",
                  "narrative_notes_structured_physical_exam_pertinent_positives"]].copy()
exam.columns = ["encounter_id", "exam_text"]
exam["exam_text"] = exam["exam_text"].fillna("").astype(str)
exam_dummies = exam["exam_text"].str.get_dummies(sep=";")
exam_dummies.columns = [f"exam_{c.strip()}" for c in exam_dummies.columns]
exam = pd.concat([exam[["encounter_id"]], exam_dummies], axis=1)

features = vitals.merge(labs, on="encounter_id").merge(exam, on="encounter_id")
feature_cols = [c for c in features.columns if c != "encounter_id"]
X_scaled = StandardScaler().fit_transform(features[feature_cols])


# ---- 1. Validity metrics across K (silhouette/calinski higher=better, davies lower=better) ----
print("=== K validation ===")
print(f"{'K':>3} {'silhouette':>11} {'davies_b':>10} {'calinski_h':>11}")
for k in range(2, 7):
    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).labels_
    print(f"{k:>3} {silhouette_score(X_scaled, lab):>11.3f} "
          f"{davies_bouldin_score(X_scaled, lab):>10.3f} "
          f"{calinski_harabasz_score(X_scaled, lab):>11.1f}")


# ---- 2. Stability across 30 random seeds ----
labels = [KMeans(n_clusters=K_FINAL, n_init=10, random_state=s).fit(X_scaled).labels_
          for s in range(30)]
aris = [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]
print(f"\nStability (mean pairwise ARI, 30 seeds): {np.mean(aris):.3f}")


# ---- 3. Feature F-stat + top-N reclustering ----
y_cluster = KMeans(n_clusters=K_FINAL, n_init=10, random_state=42).fit(X_scaled).labels_
f_scores, _ = f_classif(X_scaled, y_cluster)
feature_f = pd.DataFrame({"feature": feature_cols, "f_stat": f_scores.round(2)}) \
    .sort_values("f_stat", ascending=False).reset_index(drop=True)

print("\n=== Top 10 features ===")
print(feature_f.head(10).to_string(index=False))
print("\n=== Bottom 10 features ===")
print(feature_f.tail(10).to_string(index=False))

print("\n=== Recluster on top-N features ===")
print(f"{'n':>5} {'silhouette':>11} {'ARI vs full':>13}")
total = len(feature_cols)
for n in [total, 20, 15, 10, 5]:
    if n > total: continue
    Xk = StandardScaler().fit_transform(features[feature_f.head(n)["feature"].tolist()])
    lab_k = KMeans(n_clusters=K_FINAL, n_init=10, random_state=42).fit(Xk).labels_
    print(f"{n:>5} {silhouette_score(Xk, lab_k):>11.3f} "
          f"{adjusted_rand_score(y_cluster, lab_k):>13.3f}")


# ---- 4. Hierarchical sanity check ----
agg_lab = AgglomerativeClustering(n_clusters=K_FINAL).fit(X_scaled).labels_
print(f"\nHierarchical vs KMeans ARI: {adjusted_rand_score(y_cluster, agg_lab):.3f}")


feature_f.to_csv(f"{OUT_DIR}/cluster_feature_f_stats.csv", index=False)