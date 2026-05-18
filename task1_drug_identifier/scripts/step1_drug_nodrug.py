"""
Step 1 — Drug vs no-drug (step1_drug_nodrug.py)
Inputs:

Same two files as step 0
data/llm_features.csv (optional — Rupesh's LLM-derived features, merged if present)
Builds its own copy of the feature matrix internally

Features (~29+, triage-only):

8 vitals: HR, RR, systolic BP, diastolic BP, SpO2, temperature, GCS, pain
6 labs (split from dict): glucose, pH, sodium, potassium, hemoglobin, anion gap
1 onset feature: onset_minutes from brief note
1 festival template flag (regex on brief note)
~9 chief complaint one-hot columns
~4 mode-of-arrival one-hot columns
N LLM-derived features (optional — varies based on Rupesh's CSV)

Target: binary bucket (1 if in Yohan's GT, 0 if not)
Outputs:

Printed only:

10-fold CV metrics for RF, GBT, LR: accuracy, precision, recall, F1, ROC-AUC, PR-AUC
Top 15 RF feature importances
"""

import ast
import os
import re
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---- Config ----
DATA_PATH = "data/Hackathon_Data_Release_1_SHARE.xlsx"
GT_PATH   = "data/ground_truth_labels_v6.csv"
LLM_PATH  = "data/llm_features.csv"

ONSET_RE = re.compile(r"symptom onset\s+~?(\d+)\s+minutes?\s+before arrival",
                      re.IGNORECASE)
FESTIVAL_TEMPLATE_RE = re.compile(
    r"festival attendee\s+from\s+([a-zA-Z\s]+?)\s+with\s+symptom",
    re.IGNORECASE,
)

VITALS = [
    "triage_heart_rate", "triage_respiratory_rate",
    "triage_snapshot.systolic_bp", "triage_snapshot.diastolic_bp",
    "triage_snapshot.oxygen_saturation", "triage_temperature_c",
    "triage_gcs", "triage_pain_scale",
]
LAB_COLS = ["fingerstick_glucose", "ph", "sodium", "potassium",
            "hemoglobin", "anion_gap"]


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
triage["note_is_festival_template"] = triage["triage_brief_note"].apply(
    lambda t: 1 if isinstance(t, str) and FESTIVAL_TEMPLATE_RE.search(t) else 0
).astype(int)
triage["bucket"] = triage["encounter_id"].isin(set(gt["encounter_id"])).astype(int)

labs = triage["triage.labs"].apply(parse_labs).apply(pd.Series)
labs["encounter_id"] = triage["encounter_id"].values
labs = labs[["encounter_id"] + LAB_COLS]

chief = pd.get_dummies(
    triage[["encounter_id", "triage_chief_complaint"]],
    columns=["triage_chief_complaint"], drop_first=True,
)
mode = pd.get_dummies(
    triage[["encounter_id", "triage_mode_of_arrival"]],
    columns=["triage_mode_of_arrival"], drop_first=True,
)

data = (triage[["encounter_id", "bucket", "onset_minutes",
                "note_is_festival_template"] + VITALS]
        .merge(labs,  on="encounter_id")
        .merge(chief, on="encounter_id")
        .merge(mode,  on="encounter_id"))

if os.path.exists(LLM_PATH):
    llm = pd.read_csv(LLM_PATH)
    data = data.merge(llm, on="encounter_id", how="left")
    print(f"Merged {llm.shape[1] - 1} LLM feature(s) from {LLM_PATH}")
else:
    print(f"No LLM features at {LLM_PATH} — skipping")

y = data["bucket"]
print(f"Patients: {len(data)}   drug: {int(y.sum())}   no_drug: {int((y==0).sum())}")

all_features = [c for c in data.columns if c not in {"encounter_id", "bucket"}]


# ---- Models + CV ----
rf  = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
gbt = GradientBoostingClassifier(random_state=42)
lr  = Pipeline([("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000))])

models = [("RF", rf), ("GBT", gbt), ("LR", lr)]
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def evaluate(model, X, y, name):
    X = X.copy()
    if "onset_minutes" in X.columns:
        X["onset_minutes"] = X["onset_minutes"].fillna(0)
    X = X.fillna(X.median(numeric_only=True))
    preds = cross_val_predict(model, X, y, cv=kf)
    probs = cross_val_predict(model, X, y, cv=kf, method="predict_proba")[:, 1]
    return {
        "model":     name,
        "n_feat":    X.shape[1],
        "accuracy":  round(accuracy_score(y, preds), 3),
        "precision": round(precision_score(y, preds), 3),
        "recall":    round(recall_score(y, preds), 3),
        "f1":        round(f1_score(y, preds), 3),
        "roc_auc":   round(roc_auc_score(y, probs), 3),
        "pr_auc":    round(average_precision_score(y, probs), 3),
    }


results = [evaluate(m, data[all_features], y, name) for name, m in models]
print("\n=== 10-fold CV ===")
print(pd.DataFrame(results).to_string(index=False))


# ---- RF feature importance ----
X_full = data[all_features].copy()
X_full["onset_minutes"] = X_full["onset_minutes"].fillna(0)
X_full = X_full.fillna(X_full.median(numeric_only=True))
rf.fit(X_full, y)
imp = pd.DataFrame({
    "feature":    all_features,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False).head(15)
print("\n=== RF top 15 features ===")
print(imp.to_string(index=False))