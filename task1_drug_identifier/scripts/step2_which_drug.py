"""
Step 2 — Which drug (step2_which_drug.py)
Inputs:

Same two files as step 0
data/llm_features.csv (optional — merged if present)
Filters to drug patients (inner join with v6 GT)

Features (~29+, triage-only):

Same feature set as step 1 (vitals, labs, onset, festival template, chief complaint, mode of arrival)
N LLM-derived features (optional)

Target: multiclass predicted_drug from v6 (1=Kraken, 2=Triton, 3=Coral)
Outputs:

Printed only:

10-fold CV metrics for RF and LR: accuracy, macro precision/recall/F1, macro ROC-AUC, macro PR-AUC
Per-class classification report for each model
Confusion matrix for each model
Top 10 RF feature importances
"""

import ast
import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                              classification_report, confusion_matrix,
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


# ---- Load + filter to drug patients ----
triage = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
gt     = pd.read_csv(GT_PATH)

triage = triage[triage["encounter_id"].isin(set(gt["encounter_id"]))].copy()


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

data = (triage[["encounter_id", "onset_minutes",
                "note_is_festival_template"] + VITALS]
        .merge(labs,  on="encounter_id")
        .merge(chief, on="encounter_id")
        .merge(mode,  on="encounter_id"))

data = data.merge(gt[["encounter_id", "predicted_drug"]], on="encounter_id")

if os.path.exists(LLM_PATH):
    llm = pd.read_csv(LLM_PATH)
    data = data.merge(llm, on="encounter_id", how="left")
    print(f"Merged {llm.shape[1] - 1} LLM feature(s) from {LLM_PATH}")
else:
    print(f"No LLM features at {LLM_PATH} — skipping")

y = data["predicted_drug"].astype(int)
class_counts = y.value_counts().sort_index().to_dict()
print(f"Drug patients: {len(data)}   class counts: {class_counts}")

all_features = [c for c in data.columns
                if c not in {"encounter_id", "predicted_drug"}]


# ---- Models + CV ----
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
lr = Pipeline([("scaler", StandardScaler()),
               ("clf", LogisticRegression(max_iter=1000))])

models = [("RF", rf), ("LR", lr)]
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

X = data[all_features].copy()
X["onset_minutes"] = X["onset_minutes"].fillna(X["onset_minutes"].median())
X = X.fillna(X.median(numeric_only=True))


def evaluate(model, X, y, name):
    preds = cross_val_predict(model, X, y, cv=kf)
    probs = cross_val_predict(model, X, y, cv=kf, method="predict_proba")
    classes = sorted(y.unique())
    prauc = np.mean([average_precision_score((y == c).astype(int), probs[:, i])
                     for i, c in enumerate(classes)])
    return {
        "model":              name,
        "accuracy":           round(accuracy_score(y, preds), 3),
        "macro_precision":    round(precision_score(y, preds, average="macro"), 3),
        "macro_recall":       round(recall_score(y, preds, average="macro"), 3),
        "macro_f1":           round(f1_score(y, preds, average="macro"), 3),
        "macro_roc_auc_ovr":  round(roc_auc_score(y, probs, multi_class="ovr",
                                                    average="macro"), 3),
        "macro_pr_auc_ovr":   round(prauc, 3),
    }


print(f"\n=== 10-fold CV ({X.shape[1]} features) ===")
results = [evaluate(m, X, y, name) for name, m in models]
print(pd.DataFrame(results).to_string(index=False))


# ---- Per-model report + confusion matrix ----
for name, model in models:
    preds = cross_val_predict(model, X, y, cv=kf)
    print(f"\n=== {name} per-class report ===")
    print(classification_report(y, preds, target_names=["Kraken", "Triton", "Coral"],
                                 digits=3))
    print(f"=== {name} confusion matrix ===")
    print(pd.DataFrame(confusion_matrix(y, preds),
                       index=["Kraken", "Triton", "Coral"],
                       columns=["Kraken", "Triton", "Coral"]).to_string())


# ---- RF feature importance ----
rf.fit(X, y)
imp = pd.DataFrame({
    "feature":    all_features,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False).head(10)
print("\n=== RF top 10 features ===")
print(imp.to_string(index=False))