"""
Step 2 — Which drug (step2_which_drug.py)
Inputs:

Same two files as step 0
Filters to 157 drug patients (inner join with GT)

Features: same 24 as step 1's base (mode of arrival and LLM features not currently wired into step 2 — could be added)
Target: multiclass predicted_drug from v5 (1=Kraken, 2=Triton, 3=Coral)
Outputs:

Printed only:

10-fold CV metrics: accuracy, macro precision/recall/F1, macro ROC-AUC, macro PR-AUC
Per-class classification report
Confusion matrix (Kraken/Triton/Coral)
Top 10 feature importances
"""

import ast
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                              classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


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
DRUG_NAMES = {1: "Kraken", 2: "Triton", 3: "Coral"}


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

labs = triage["triage.labs"].apply(parse_labs).apply(pd.Series)
labs["encounter_id"] = triage["encounter_id"].values
labs = labs[["encounter_id"] + LAB_COLS]

chief = pd.get_dummies(
    triage[["encounter_id", "triage_chief_complaint"]],
    columns=["triage_chief_complaint"], drop_first=True,
)

# Filter to drug patients via inner join with GT
data = (triage[["encounter_id", "onset_minutes"] + VITALS]
        .merge(labs,  on="encounter_id")
        .merge(chief, on="encounter_id")
        .merge(gt[["encounter_id", "predicted_drug"]], on="encounter_id", how="inner"))

feature_cols = [c for c in data.columns if c not in {"encounter_id", "predicted_drug"}]
X = data[feature_cols].fillna(data[feature_cols].median(numeric_only=True))
y = data["predicted_drug"].astype(int)
print(f"Drug patients: {len(data)}   class counts: {y.value_counts().sort_index().to_dict()}")


# ---- Model + CV ----
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

preds = cross_val_predict(rf, X, y, cv=kf)
probs = cross_val_predict(rf, X, y, cv=kf, method="predict_proba")


# ---- Metrics ----
class_labels = sorted(y.unique())
class_names  = [DRUG_NAMES[c] for c in class_labels]
y_onehot = pd.get_dummies(y)[class_labels].values

print(f"\n=== Random forest, 10-fold CV ({len(feature_cols)} features) ===")
print(f"accuracy:           {accuracy_score(y, preds):.3f}")
print(f"macro precision:    {precision_score(y, preds, average='macro', zero_division=0):.3f}")
print(f"macro recall:       {recall_score(y, preds, average='macro', zero_division=0):.3f}")
print(f"macro f1:           {f1_score(y, preds, average='macro', zero_division=0):.3f}")
print(f"macro ROC-AUC (ovr):{roc_auc_score(y, probs, multi_class='ovr', average='macro'):.3f}")
print(f"macro PR-AUC (ovr): {average_precision_score(y_onehot, probs, average='macro'):.3f}")

print("\n=== Per-class report ===")
print(classification_report(y, preds, target_names=class_names, digits=3))

print("=== Confusion matrix ===")
print(pd.DataFrame(confusion_matrix(y, preds),
                   index=class_names, columns=class_names).to_string())


# ---- Feature importance (full-data fit) ----
rf.fit(X, y)
imp = pd.DataFrame({
    "feature":    feature_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False).head(10)
print("\n=== Top 10 features ===")
print(imp.to_string(index=False))