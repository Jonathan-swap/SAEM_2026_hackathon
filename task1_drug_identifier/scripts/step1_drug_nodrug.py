"""
Step 1 — Drug vs. no-drug binary classifier.

Random forest, 10-fold stratified CV. Triage-only features.

Target: bucket from Yohan's GT (drug = in GT file, else no_drug).

Compares two feature sets:
  - "baseline":  vitals + 6 labs + onset_minutes + chief complaint  (~24 features)
  - "+ extras":  adds triage_age, sex, ESI, 5 PMH flags             (~34 features)

Benchmark: ~90% accuracy (Yohan).
"""

import ast
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# ---- Config ----
DATA_PATH = "data/Hackathon_Data_Release_1_SHARE.xlsx"
GT_PATH   = "data/ground_truth_labels_v4.csv"

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
PMH_COLS = ["triage_mh_psych", "triage_mh_cardiac", "triage_mh_pulm",
            "triage_mh_renal", "triage_mh_substance_use"]


# ---- Load + build feature matrix ----
triage = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
gt     = pd.read_csv(GT_PATH)


def parse_labs(cell):
    if pd.isna(cell): return {}
    parsed = ast.literal_eval(cell)
    return parsed[0] if parsed else {}


def extract_onset(text):
    if pd.isna(text): return float("nan")
    m = ONSET_RE.search(str(text))
    return float(m.group(1)) if m else float("nan")


triage["onset_minutes"] = triage["triage_brief_note"].apply(extract_onset)
triage["bucket"] = triage["encounter_id"].isin(set(gt["encounter_id"])).astype(int)

labs = triage["triage.labs"].apply(parse_labs).apply(pd.Series)
labs["encounter_id"] = triage["encounter_id"].values
labs = labs[["encounter_id"] + LAB_COLS]

chief = pd.get_dummies(
    triage[["encounter_id", "triage_chief_complaint"]],
    columns=["triage_chief_complaint"], drop_first=True,
)

# New: demographics + ESI + PMH
sex = pd.get_dummies(
    triage[["encounter_id", "triage_sex_gender"]],
    columns=["triage_sex_gender"], drop_first=True,
)
extras = triage[["encounter_id", "triage_age", "triage_esi"] + PMH_COLS]

data = (triage[["encounter_id", "bucket", "onset_minutes"] + VITALS]
        .merge(labs,   on="encounter_id")
        .merge(chief,  on="encounter_id")
        .merge(extras, on="encounter_id")
        .merge(sex,    on="encounter_id"))

y = data["bucket"]
print(f"Patients: {len(data)}   drug: {int(y.sum())}   no_drug: {int((y==0).sum())}")


# ---- Define feature sets ----
extras_cols = ["triage_age", "triage_esi"] + PMH_COLS + [
    c for c in data.columns if c.startswith("triage_sex_gender_")
]
all_features      = [c for c in data.columns if c not in {"encounter_id", "bucket"}]
baseline_features = [c for c in all_features if c not in extras_cols]


# ---- Model + CV ----
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def evaluate(X, y, name):
    X = X.copy()
    if "onset_minutes" in X.columns:
        X["onset_minutes"] = X["onset_minutes"].fillna(0)
    X = X.fillna(X.median(numeric_only=True))
    preds = cross_val_predict(rf, X, y, cv=kf)
    probs = cross_val_predict(rf, X, y, cv=kf, method="predict_proba")[:, 1]
    return {
        "config":    name,
        "n_feat":    X.shape[1],
        "accuracy":  round(accuracy_score(y, preds), 3),
        "precision": round(precision_score(y, preds), 3),
        "recall":    round(recall_score(y, preds), 3),
        "f1":        round(f1_score(y, preds), 3),
        "roc_auc":   round(roc_auc_score(y, probs), 3),
        "pr_auc":    round(average_precision_score(y, probs), 3),
    }


results = [
    evaluate(data[baseline_features], y, "baseline"),
    evaluate(data[all_features],      y, "+ extras"),
]
print("\n=== Random forest, 10-fold CV ===")
print(pd.DataFrame(results).to_string(index=False))


# ---- Feature importance (full feature set) ----
X_full = data[all_features].copy()
X_full["onset_minutes"] = X_full["onset_minutes"].fillna(0)
X_full = X_full.fillna(X_full.median(numeric_only=True))
rf.fit(X_full, y)

imp = pd.DataFrame({
    "feature":    all_features,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False).head(15)
print("\n=== Top 15 features ===")
print(imp.to_string(index=False))