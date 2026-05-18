"""Build a single canonical outcomes file containing both supervised
targets:

  - ``ground_truth_drug``         int code  (0=None, 1=Kraken, 2=Triton, 3=Coral)
  - ``ground_truth_drug_name``    string    (per Task1_Two_Tier_Input_Data.csv)
  - ``encounter_disposition_label`` string  (Discharge / Floor / ICU,
                                              from the xlsx 'Disposition' sheet)

Output: ``derived/outcomes.csv`` — the canonical source for BOTH
Task-1 (drug ID at triage) and Task-2 (4h deterioration). The two
trainers + the temporal-holdout evaluator + the cluster script all
read this file instead of pulling each label from a different
location.

Depends on:
  - derived/ground_truth.csv          (written by load_ground_truth.py)
  - data/Hackathon_Data_Release_1_SHARE.xlsx :: Disposition sheet

Privacy: writes only encounter_id + two label strings. No row data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
XLSX = ROOT / "data" / "Hackathon_Data_Release_1_SHARE.xlsx"


def main() -> None:
    # Drug labels — already on disk in canonical form
    gt = pd.read_csv(DERIVED / "ground_truth.csv")
    expected_gt_cols = {"encounter_id", "ground_truth_drug",
                         "ground_truth_drug_name"}
    if not expected_gt_cols.issubset(gt.columns):
        missing = expected_gt_cols - set(gt.columns)
        raise SystemExit(
            f"derived/ground_truth.csv missing columns: {missing}. "
            f"Run src/labels/load_ground_truth.py first.")

    # Disposition labels — from the Disposition sheet of the xlsx
    dispo = pd.read_excel(XLSX, sheet_name="Disposition",
                            engine="openpyxl")
    expected_dispo_cols = {"encounter_id", "encounter_disposition_label"}
    if not expected_dispo_cols.issubset(dispo.columns):
        missing = expected_dispo_cols - set(dispo.columns)
        raise SystemExit(
            f"xlsx Disposition sheet missing columns: {missing}.")

    outcomes = gt.merge(
        dispo[["encounter_id", "encounter_disposition_label"]],
        on="encounter_id", how="outer", indicator=True)

    only_left = (outcomes["_merge"] == "left_only").sum()
    only_right = (outcomes["_merge"] == "right_only").sum()
    both = (outcomes["_merge"] == "both").sum()
    if only_left or only_right:
        print(f"WARN: merge mismatch — only-in-ground_truth={only_left}, "
              f"only-in-Disposition={only_right}, both={both}")
    else:
        print(f"OK   ground_truth.csv and Disposition sheet align on "
              f"all {both} encounters.")

    outcomes = outcomes.drop(columns=["_merge"])
    # Stable column order
    cols = ["encounter_id", "ground_truth_drug", "ground_truth_drug_name",
            "encounter_disposition_label"]
    outcomes = outcomes[cols]

    out_path = DERIVED / "outcomes.csv"
    outcomes.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")
    print(f"  rows: {len(outcomes)}")
    print(f"  drug distribution:")
    print("  " + outcomes["ground_truth_drug_name"].fillna("None")
            .value_counts().to_string().replace("\n", "\n  "))
    print(f"  disposition distribution:")
    print("  " + outcomes["encounter_disposition_label"]
            .value_counts().to_string().replace("\n", "\n  "))


if __name__ == "__main__":
    main()
