# Data

This folder holds the SAEM26 Hackathon dataset.

The dataset is not committed to this repository. It was provided by the SAEM26 Hackathon organizers and is synthetic data modeled after real ED patient records.

## Setup

Place the following file in this folder:

- `Hackathon_Data_Release_1_SHARE.xlsx`

The file has three sheets:

- **Triage_Data** — vitals, labs, and triage notes collected on arrival
- **Four_Hour_Data** — repeat vitals, labs, imaging, and clinical narratives at the 4-hour mark
- **Disposition** — final disposition (Discharge / Floor / ICU) for each encounter

## Day-of Hackathon

A second dataset will be released at the in-person hackathon event. Drop it in this folder and re-run the pipeline (see the main README) to retrain.