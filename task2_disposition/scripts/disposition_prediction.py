"""
SAEM26 Hackathon — Task 2 disposition prediction.
Tuned gradient boosting on triage + 4-hour features for festival patients.
Final model uses feature pruning at importance >= 0.005
(see feature_pruning.py for the threshold comparison that justifies this).
"""

import ast
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, make_scorer)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                      cross_val_predict)
from sklearn.preprocessing import StandardScaler


def fpr_fnr_table(y_true, y_pred, class_names):
    """Per-class false positive rate and false negative rate."""
    cm = confusion_matrix(y_true, y_pred)
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        rows.append({"class": name, "FPR": round(fpr, 3), "FNR": round(fnr, 3)})
    return pd.DataFrame(rows)


# ---- Paths ----
DATA_PATH = "data/Hackathon_Data_Release_1_SHARE.xlsx"
TASK1_OUT = "task1_drug_identifier/out"
OUT_DIR   = "task2_disposition/out"
PRUNE_THRESHOLD = 0.005   # drop features with importance below this in the final model


# ---- Load data, keep festival patients only ----
triage      = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
four_hour   = pd.read_excel(DATA_PATH, sheet_name="Four_Hour_Data")
disposition = pd.read_excel(DATA_PATH, sheet_name="Disposition")
clusters    = pd.read_csv(f"{TASK1_OUT}/cluster_labels.csv")

festival_ids = clusters["encounter_id"]
triage      = triage[triage["encounter_id"].isin(festival_ids)]
four_hour   = four_hour[four_hour["encounter_id"].isin(festival_ids)]
disposition = disposition[disposition["encounter_id"].isin(festival_ids)]
print("Festival patients:", len(clusters))


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

# Parse triage labs from stringified dict
def parse_labs(cell):
    if pd.isna(cell): return {}
    parsed = ast.literal_eval(cell)
    return parsed[0] if parsed else {}

labs = triage["triage.labs"].apply(parse_labs).apply(pd.Series)
labs["encounter_id"] = triage["encounter_id"].values
labs = labs[["encounter_id"] + lab_cols]

# Triage extras: ESI, supplemental O2, PMH flags, chief complaint (one-hot)
pmh_cols = ["triage_mh_psych", "triage_mh_cardiac", "triage_mh_pulm",
            "triage_mh_renal", "triage_mh_substance_use"]
triage_extras = triage[["encounter_id", "triage_esi",
                        "triage_supplemental_oxygen"] + pmh_cols].copy()
chief_complaint = pd.get_dummies(
    triage[["encounter_id", "triage_chief_complaint"]],
    columns=["triage_chief_complaint"], drop_first=True,
)

# Additional labs drawn flag: 1 if any 4h scalar lab has a value, else 0.
extra_labs = ["ed_course_reassessment_4h.lactate_4h",
              "ed_course_reassessment_4h.cpk_4h",
              "ed_course_reassessment_4h.vbg_ph_4h",
              "ed_course_reassessment_4h.troponin_4h"]
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
X = features[feature_cols].astype(float)

# Target: 0 = Discharge, 1 = Floor, 2 = ICU
disp_map = {"Discharge": 0, "Floor": 1, "ICU": 2}
y = disposition.set_index("encounter_id").loc[features["encounter_id"]]
y = y["encounter_disposition_label"].map(disp_map).values


# ---- 10-fold stratified CV ----
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# ---- Proportional-odds check (justifies choosing GB over ordinal logistic) ----
X_scaled = StandardScaler().fit_transform(X)
m1 = LogisticRegression(max_iter=1000).fit(X_scaled, (y >= 1).astype(int))
m2 = LogisticRegression(max_iter=1000).fit(X_scaled, (y >= 2).astype(int))
po_corr = np.corrcoef(m1.coef_[0], m2.coef_[0])[0, 1]
print(f"Proportional-odds correlation: {po_corr:.3f}  (low = use GB)")

coef_compare = pd.DataFrame({
    "feature": feature_cols,
    "coef_dx_vs_floor+icu": m1.coef_[0].round(3),
    "coef_dx+floor_vs_icu": m2.coef_[0].round(3),
})
coef_compare["abs_diff"] = (
    coef_compare["coef_dx_vs_floor+icu"] - coef_compare["coef_dx+floor_vs_icu"]
).abs().round(3)


# ---- Grid search on gradient boosting (full features), optimizing ICU F1 ----
icu_f1 = make_scorer(f1_score, labels=[2], average="macro")
param_grid = {
    "n_estimators":  [100, 150, 200],
    "max_depth":     [2, 3, 4],
    "learning_rate": [0.05, 0.075, 0.1],
}
grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=param_grid, scoring=icu_f1, cv=cv, n_jobs=-1,
).fit(X, y)
print(f"Best ICU F1 (full features): {grid.best_score_:.3f}  params: {grid.best_params_}")


# ---- Feature importance from the tuned full-feature model ----
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": grid.best_estimator_.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)


# ---- Prune low-importance features and re-fit with same best params ----
kept = importances.loc[importances["importance"] >= PRUNE_THRESHOLD, "feature"].tolist()
dropped = [f for f in feature_cols if f not in kept]
print(f"\nPruning at {PRUNE_THRESHOLD}: keeping {len(kept)} features, dropping {len(dropped)}")

X_pruned = X[kept]
final_model = GradientBoostingClassifier(**grid.best_params_, random_state=42)
final_model.fit(X_pruned, y)


# ---- Final metrics on the pruned model ----
class_names = ["Discharge", "Floor", "ICU"]
preds = cross_val_predict(final_model, X_pruned, y, cv=cv)

print("\n" + classification_report(y, preds, target_names=class_names, digits=3))
print("Confusion matrix:")
print(pd.DataFrame(confusion_matrix(y, preds), index=class_names, columns=class_names))
print("\n" + fpr_fnr_table(y, preds, class_names).to_string(index=False))


# ---- Save ----
coef_compare.to_csv(f"{OUT_DIR}/proportional_odds_check.txt", sep="\t", index=False)
importances.to_csv(f"{OUT_DIR}/feature_importance.txt", sep="\t", index=False)
pd.DataFrame(grid.cv_results_)[
    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
].sort_values("rank_test_score").to_csv(
    f"{OUT_DIR}/gb_grid_search_results.txt", sep="\t", index=False,
)
pd.Series(kept, name="feature").to_csv(f"{OUT_DIR}/final_feature_list.txt", index=False)
print(f"\nSaved to {OUT_DIR}/")