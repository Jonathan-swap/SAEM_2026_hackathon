"""
SAEM26 Hackathon — Task 2 feature pruning.
Drops features below 0.001 / 0.005 / 0.01 importance and reports metrics. 
Run from repo root: python task2_disposition/scripts/feature_pruning.py
"""

import ast
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


DATA_PATH = "data/Hackathon_Data_Release_1_SHARE.xlsx"
TASK1_OUT = "task1_drug_identifier/out"
BEST_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)


# ---- Load + filter to festival ----
triage      = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
four_hour   = pd.read_excel(DATA_PATH, sheet_name="Four_Hour_Data")
disposition = pd.read_excel(DATA_PATH, sheet_name="Disposition")
clusters    = pd.read_csv(f"{TASK1_OUT}/cluster_labels.csv")

festival_ids = clusters["encounter_id"]
triage      = triage[triage["encounter_id"].isin(festival_ids)]
four_hour   = four_hour[four_hour["encounter_id"].isin(festival_ids)]
disposition = disposition[disposition["encounter_id"].isin(festival_ids)]


# ---- Build feature matrix ----
triage_vitals = [
    "triage_heart_rate", "triage_respiratory_rate",
    "triage_snapshot.systolic_bp", "triage_snapshot.diastolic_bp",
    "triage_snapshot.oxygen_saturation", "triage_temperature_c",
    "triage_gcs", "triage_pain_scale",
]
fourhr_vitals = [
    "ed_course_reassessment_4h.heart_rate_4h",
    "ed_course_reassessment_4h.respiratory_rate_4h",
    "ed_course_reassessment_4h.systolic_bp_4h",
    "ed_course_reassessment_4h.diastolic_bp_4h",
    "ed_course_reassessment_4h.oxygen_saturation_4h",
    "ed_course_reassessment_4h.supplemental_oxygen_4h",
    "ed_course_reassessment_4h.temperature_c_4h",
    "ed_course_reassessment_4h.gcs_4h",
    "ed_course_reassessment_4h.end_tidal_co2_4h",
    "ed_course_reassessment_4h.delta_hr",
    "ed_course_reassessment_4h.delta_temp",
    "ed_course_reassessment_4h.delta_gcs",
]
lab_cols = ["fingerstick_glucose", "ph", "sodium", "potassium",
            "hemoglobin", "anion_gap"]
pmh_cols = ["triage_mh_psych", "triage_mh_cardiac", "triage_mh_pulm",
            "triage_mh_renal", "triage_mh_substance_use"]
extra_labs = ["ed_course_reassessment_4h.lactate_4h",
              "ed_course_reassessment_4h.cpk_4h",
              "ed_course_reassessment_4h.vbg_ph_4h",
              "ed_course_reassessment_4h.troponin_4h"]

def parse_labs(cell):
    if pd.isna(cell): return {}
    parsed = ast.literal_eval(cell)
    return parsed[0] if parsed else {}

labs = triage["triage.labs"].apply(parse_labs).apply(pd.Series)
labs["encounter_id"] = triage["encounter_id"].values
labs = labs[["encounter_id"] + lab_cols]

triage_extras = triage[["encounter_id", "triage_esi",
                        "triage_supplemental_oxygen"] + pmh_cols].copy()
chief_complaint = pd.get_dummies(
    triage[["encounter_id", "triage_chief_complaint"]],
    columns=["triage_chief_complaint"], drop_first=True,
)
additional_labs = pd.DataFrame({
    "encounter_id": four_hour["encounter_id"].values,
    "additional_labs_drawn": four_hour[extra_labs].notna().any(axis=1).astype(int).values,
})
demo = pd.get_dummies(
    triage[["encounter_id", "triage_age", "triage_sex_gender"]],
    columns=["triage_sex_gender"], drop_first=True,
)
cluster_features = pd.concat(
    [clusters["encounter_id"], pd.get_dummies(clusters["cluster"], prefix="cluster")],
    axis=1,
)

features = (
    triage[["encounter_id"] + triage_vitals]
    .merge(four_hour[["encounter_id"] + fourhr_vitals], on="encounter_id")
    .merge(labs, on="encounter_id")
    .merge(triage_extras, on="encounter_id")
    .merge(chief_complaint, on="encounter_id")
    .merge(additional_labs, on="encounter_id")
    .merge(demo, on="encounter_id")
    .merge(cluster_features, on="encounter_id")
)
feature_cols = [c for c in features.columns if c != "encounter_id"]
X_full = features[feature_cols].astype(float)

disp_map = {"Discharge": 0, "Floor": 1, "ICU": 2}
y = disposition.set_index("encounter_id").loc[features["encounter_id"]]
y = y["encounter_disposition_label"].map(disp_map).values


# ---- Baseline importances + evaluation loop ----
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
importances = pd.Series(
    GradientBoostingClassifier(**BEST_PARAMS).fit(X_full, y).feature_importances_,
    index=feature_cols,
)

def evaluate(X, label):
    preds = cross_val_predict(GradientBoostingClassifier(**BEST_PARAMS), X, y, cv=cv)
    cm = confusion_matrix(y, preds)
    return {
        "config":        label,
        "n_features":    X.shape[1],
        "accuracy":      round((preds == y).mean(), 3),
        "icu_recall":    round(recall_score(y, preds, labels=[2], average="macro"), 3),
        "icu_f1":        round(f1_score(y, preds, labels=[2], average="macro"), 3),
        "floor_recall":  round(recall_score(y, preds, labels=[1], average="macro"), 3),
        "icu_sent_home": int(cm[2, 0]),
    }

results = [evaluate(X_full, "baseline")]
for thr in [0.001, 0.005, 0.01]:
    keep = importances[importances >= thr].index.tolist()
    results.append(evaluate(X_full[keep], f"drop < {thr}"))

print(pd.DataFrame(results).to_string(index=False))