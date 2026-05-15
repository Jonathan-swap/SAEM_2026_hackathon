# SAEM_2026_hackathon

Public repo for the UVA × Stanford × University of Washington team in the SAEM 2026 hackathon.

## The challenge

A fictional ED is overwhelmed during the "Soaking Man" music festival. Patients arrive having taken one of three unknown drugs (Kraken Candy, Triton Tabs, Coral Dust) that standard tox screens don't detect. We build three things:

1. **Identify festival drug overdoses and which drug they took** — from triage data only
2. **Predict disposition and severity** — using triage + 4-hour data
3. **A Shiny triage calculator** — for clinicians at the bedside

Task 1 (drug identification) is broken into three sequential steps:
- **Step 0** — Data prep: build the feature matrix
- **Step 1** — Drug vs. no-drug (binary classifier)
- **Step 2** — Which drug: Kraken / Triton / Coral (multiclass classifier)

## Repo structure

```
SAEM_2026_hackathon/
├── data/                              # dataset goes here (gitignored)
├── task1_drug_identifier/
│   └── scripts/
│       ├── step0_data_prep.py
│       ├── step1_drug_nodrug.py
│       └── step2_which_drug.py
├── task2_disposition/
│   └── scripts/
│       ├── disposition_prediction.py
│       └── feature_pruning.py
├── task3_triage calculator/
│   └── App.R
├── requirements.txt
├── setup.sh
└── README.md
```

## Setup

One-time, after cloning:

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

For the R app:

```bash
R -e "install.packages(c('shiny','here'), repos='https://cloud.r-project.org')"
```

Place the hackathon data in `data/` (see `data/README.md`).

## Running the pipeline

Run from the repo root. Steps within Task 1 build on each other.

```bash
# Task 1 — drug identification
python task1_drug_identifier/scripts/step0_data_prep.py
python task1_drug_identifier/scripts/step1_drug_nodrug.py
python task1_drug_identifier/scripts/step2_which_drug.py

# Task 2 — disposition + severity
python task2_disposition/scripts/disposition_prediction.py

# Task 3 — Shiny calculator
R -e "shiny::runApp('task3_triage calculator/App.R')"
```

## What each file does

### Task 1 — Drug identification

**`step0_data_prep.py`** — Loads the triage data and Yohan's ground-truth labels file. Builds a triage-only feature matrix: 8 vital signs, 6 lab values (split out from the dict column), an `onset_minutes` feature extracted from the brief note, and one-hot chief complaint. Validates Yohan's hypothesis that drug patients have an onset time recorded. Also runs exploratory K-means on the drug patients (just for inspection — clustering isn't part of the prediction pipeline anymore).

**`step1_drug_nodrug.py`** — Predicts drug vs. no-drug. Trains random forest, gradient boosting, and logistic regression side by side. 10-fold stratified cross-validation. Reports accuracy, precision, recall, F1, ROC-AUC, and PR-AUC. Yohan's benchmark is ~90% accuracy.

**`step2_which_drug.py`** — Predicts which drug (1=Kraken / 2=Triton / 3=Coral) on drug patients only. Random forest, 10-fold CV. Reports per-class metrics, the confusion matrix, and top features. Triton vs. Coral is the hardest pair to separate.

### Task 2 — Disposition + severity

**`disposition_prediction.py`** — Trains two parallel gradient boosting models on the same feature matrix (drug patients only): one predicts disposition (Discharge / Floor / ICU), the other predicts severity (1=low / 2=moderate / 3=high). Grid search for hyperparameters, importance-based feature pruning at threshold 0.005, 10-fold CV. Each model optimizes F1 on its most critical class (ICU for disposition, High for severity).

**`feature_pruning.py`** — Companion experiment that justified the 0.005 pruning threshold for the disposition model. Compares model performance after dropping features below thresholds 0.001, 0.005, and 0.01.

### Task 3 — Triage calculator

**`App.R`** — Shiny app where a clinician enters triage vitals and gets predicted drug probabilities. Designed to embed the trained model coefficients directly so no Python backend is needed at runtime.

## Methodology notes

- **Ground-truth labels** come from Yohan's `ground_truth_labels_v4.csv`, derived from the 4-hour MDM flag. Used for training only — the deployed model only uses triage-time features.
- **Why these models:** Random forest for the supervised classifiers (Yohan's recommendation; handles mixed feature types well on small N). Gradient boosting for disposition + severity (won out over ordinal logistic regression after the proportional-odds assumption failed). Logistic regression compared throughout for interpretability.
- **Why supervised, not clustering, for Task 1:** Earlier work showed unsupervised clusters don't track drug identity (ARI ≈ 0 against ground truth). With labels available, supervised models are the cleaner approach. Clustering remains in `step0_data_prep.py` only for exploratory inspection.
- **Onset time** is used as a regular feature, not as a cohort rule. 72% of drug patients and 84% of no-drug patients have it recorded, so it's not a label proxy.

## Day-of adaptation plan

1. Drop the new dataset into `data/`
2. Re-run Task 1 (step 0 → step 1 → step 2)
3. Re-run Task 2 (disposition + severity)
4. Shiny app picks up new coefficients
5. Update slides with new performance numbers