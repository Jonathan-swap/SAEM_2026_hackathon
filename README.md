# SAEM_2026_hackathon
This is the public repository for the UVA x Stanford x University of Washington team in the SAEM 2026 hackathon. It is broken into the 3 tasks: 1) identifying festival-goers and clustering drug types 2) prediction of disposition using data available during triage (first 4 hours) 3) a calculator to predict the probability of each drug.


# SAEM26 Hackathon — Festival Drug Triage

Pipeline for the SAEM26 Hackathon challenge: identify which festival drug a patient took from triage data, predict their disposition (Discharge / Floor / ICU), and provide a clinician-facing triage calculator.

## Overview

Three connected tasks:

1. **Drug identification at triage** — Cluster festival patients by physiology, then train a classifier on triage vitals to predict the drug.
2. **Disposition prediction** — Predict Discharge / Floor / ICU for festival patients using triage + 4-hour data.
3. **Rapid triage calculator** — A Shiny app where nurses enter vitals once and see both the predicted drug and a copy-paste vitals summary for the EMR.

## Repo Structure

```
hackathon/
├── data/                          # Dataset (not committed — see data/README.md)
├── task1_drug_identifier/
│   ├── scripts/
│   │   ├── festival_flag.py       # Step 1 — flag festival vs. non-festival patients
│   │   ├── clustering.py          # Step 2 — cluster festival patients into 3 drug groups
│   │   └── drug_classifier.py          # Step 3 — train logistic regression on triage vitals
│   └── out/                       # Generated outputs (not committed)
├── task2_disposition/
│   ├── scripts/
│   │   └── disposition.py         # Tuned gradient boosting for disposition prediction
│   └── out/                       # Generated outputs (not committed)
├── task3_triage_calculator/
│   └── app.R                      # Shiny calculator app
├── requirements.txt               # Python dependencies
└── README.md
```

## Setup

Run the setup.sh script once after cloning. It creates a Python virtual environment, installs dependencies, and creates the output folders the pipeline writes to.

```bash 
chmod +x setup.sh   # one-time, makes the script executable
./setup.sh
```

Then activate the virtual environment in your shell:
```bash
source .venv/bin/activate
```

Run this command in the terminal to install shiny for the R app:
```bash
R -e "install.packages('shiny', repos='https://cloud.r-project.org')"
```

Place hackathon data in the `data/` folder. See `data/README.md`.

## Running the Pipeline

Run in order and from the main repo folder space as the pathing for all files are relative from the main repo space. Each step depends on the previous one's output:

```bash
# Task 1
python task1_drug_identifier/scripts/festival_flag.py
python task1_drug_identifier/scripts/clustering.py
python task1_drug_identifier/scripts/drug_classifier.py

# Task 2
python task2_disposition/scripts/disposition.py

# Task 3 — Shiny calculator
R -e "shiny::runApp('task3_triage_calculator/app.R')"
```

The Shiny app reads `model_coefficients.csv`, `feature_scaling.csv`, and `feature_bounds.csv` from `task1_drug_identifier/out/`, so Task 1's classifier must be run first.

## Methodology Highlights

- **Task 1** uses unsupervised clustering because the dataset has no drug labels. Clusters are mapped to Kraken / Triton / Coral by physiologic signature.
- **Task 2** uses gradient boosting after ordinal logistic regression's proportional-odds assumption failed (cutpoint correlation 0.59). Intervention flags are excluded to avoid target leakage.
- **Task 3** embeds the Task 1 logistic regression coefficients directly into Shiny, so the app needs no backend.

## Day-of Adaptation

The pipeline is built to handle the in-person twist:
- Class counts and hyperparameters are not hard-coded
- Each task lives in its own folder with separate train/predict logic
- The Shiny app reads model bounds from a CSV that regenerates with each retrain

To retrain on new data: drop the new file in `data/` and re-run the pipeline above.