"""
SAEM26 Hackathon — Step 3 starter code.

Trains a triage drug classifier using only triage vitals.
The cluster labels from Step 2 are the prediction target.
We try two models (logistic regression and gradient boosting)
and compare them with 10-fold cross-validation.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---- 1. Load data + cluster labels from Step 2 ----
DATA_PATH = "/data/Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = "/task1_drug_identifier/out"

triage = pd.read_excel(DATA_PATH, sheet_name="Triage_Data")
clusters = pd.read_csv(f"{OUT_DIR}/cluster_labels.csv")

# Keep only festival patients (the ones with cluster labels).
data = triage.merge(clusters, on="encounter_id")
print("Patients available for classification:", len(data))


# ---- 2. Pick features (vitals only) ----
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

X = data[vitals_cols]
y = data["cluster"]   # the cluster from Step 2 is our target


# ---- 3. Set up 10-fold cross-validation ----
# StratifiedKFold keeps each cluster's proportion roughly equal in every fold.
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# ---- 4. Model 1: Logistic Regression ----
# Scaler + logistic regression in a pipeline so scaling happens fold-by-fold.
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

logreg_scores = cross_val_score(logreg, X, y, cv=cv, scoring="accuracy")
print(f"\nLogistic Regression — accuracy: {logreg_scores.mean():.3f} "
      f"(+/- {logreg_scores.std():.3f})")


# ---- 5. Model 2: Gradient Boosting ----
gb = GradientBoostingClassifier(random_state=42)

gb_scores = cross_val_score(gb, X, y, cv=cv, scoring="accuracy")
print(f"Gradient Boosting   — accuracy: {gb_scores.mean():.3f} "
      f"(+/- {gb_scores.std():.3f})")


# ---- 6. Inspect logistic regression coefficients ----
# Coefficients tell us which vitals drive each cluster prediction.
# We fit on the full data (not just one CV fold) for the coefficient summary.
logreg.fit(X, y)
coefs = pd.DataFrame(
    logreg.named_steps["clf"].coef_,
    columns=vitals_cols,
    index=[f"cluster_{c}" for c in logreg.named_steps["clf"].classes_],
).round(3)

print("\nLogistic regression coefficients (standardized features):")
print(coefs.T)

coefs_path = f"{OUT_DIR}/logreg_coefficients.txt"
coefs.T.to_csv(coefs_path, sep="\t")
print(f"\nSaved: {coefs_path}")


# ---- 7. Export model + scaler for the Shiny calculator ----
scaler = logreg.named_steps["scaler"]
clf = logreg.named_steps["clf"]

# One row per cluster (Kraken/Triton/Coral), columns = vitals + intercept.
model_export = pd.DataFrame(clf.coef_, columns=vitals_cols)
model_export.insert(0, "intercept", clf.intercept_)
model_export.insert(0, "cluster", clf.classes_)
model_export.to_csv(f"{OUT_DIR}/model_coefficients.csv", index=False)

# Means and SDs used by the StandardScaler — needed at prediction time.
scaling_export = pd.DataFrame({
    "feature": vitals_cols,
    "mean": scaler.mean_,
    "sd":   scaler.scale_,
})
scaling_export.to_csv(f"{OUT_DIR}/feature_scaling.csv", index=False)

print(f"Saved: {OUT_DIR}/model_coefficients.csv")
print(f"Saved: {OUT_DIR}/feature_scaling.csv")