"""Export consolidated probability CSVs for the three Task-1 binary
classifiers.

For each of:
  - Tier-1 drug-vs-no-drug         (cohort: all 261)
  - Kraken-vs-rest                 (cohort: 157 drug-positive)
  - Triton-vs-Coral                (cohort: 99 non-Kraken drug-positive)

merges the existing 5-fold-OOF predictions and the temporal-holdout
predictions (last-day test set) into one CSV per binary task, with
both splits side-by-side. The OOF probability is the model's
out-of-sample estimate for that encounter when it sat in the test
fold; the temporal probability is populated only for last-day
encounters (model trained on prior days only).

Inputs (read-only):
  derived/task1_binary_oof_predictions.csv
  derived/task1_binary_temporal_predictions.csv
  derived/task1_kraken_binary_oof_predictions.csv
  derived/task1_kraken_binary_temporal_predictions.csv
  derived/task1_triton_coral_oof_predictions.csv
  derived/task1_triton_coral_temporal_predictions.csv

Outputs:
  derived/task1_tier1_probabilities.csv         (n=261 rows)
  derived/task1_kraken_vs_rest_probabilities.csv (n=157)
  derived/task1_triton_vs_coral_probabilities.csv (n=99)

Output schema (per row, one row per encounter in the cohort):
  encounter_id, true_label,
  prob_<model>_cv_oof          (5-fold out-of-fold probability)
  prob_<model>_temporal        (last-day holdout probability; NaN if
                                encounter is not in the last day's
                                test set)
  is_last_day                  (1 if encounter is in the temporal
                                test set, else 0)

Where <model> ∈ {logreg, rforest, hgb} and the probability target is
the positive class for that binary: P(drug-positive),
P(Kraken Candy), P(Triton Tabs) respectively.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

MODELS = ["logreg", "rforest", "hgb"]

# (positive_class label, oof_prob_col_prefix, oof_csv, temporal_csv, out_name)
TASKS = [
    (
        "drug_vs_nodrug",
        "p_drug",
        "task1_binary_oof_predictions.csv",
        "task1_binary_temporal_predictions.csv",
        "task1_tier1_probabilities.csv",
    ),
    (
        "kraken_vs_rest",
        "p_kraken",
        "task1_kraken_binary_oof_predictions.csv",
        "task1_kraken_binary_temporal_predictions.csv",
        "task1_kraken_vs_rest_probabilities.csv",
    ),
    (
        "triton_vs_coral",
        "p_triton",
        "task1_triton_coral_oof_predictions.csv",
        "task1_triton_coral_temporal_predictions.csv",
        "task1_triton_vs_coral_probabilities.csv",
    ),
]


def consolidate(prefix: str, oof_csv: str, temp_csv: str,
                out_name: str) -> pd.DataFrame:
    oof = pd.read_csv(DERIVED / oof_csv)
    temp = pd.read_csv(DERIVED / temp_csv)

    # OOF is wide: encounter_id, true_label, p_<class>_<model> (3 cols)
    out = oof[["encounter_id", "true_label"]].copy()
    for m in MODELS:
        out[f"prob_{m}_cv_oof"] = oof[f"{prefix}_{m}"]

    # Temporal is long: model, encounter_id, true_label, p_<class>, pred_label.
    # Pivot to wide, then left-merge onto the cohort.
    temp_wide = temp.pivot_table(
        index="encounter_id",
        columns="model",
        values=prefix,
        aggfunc="first",
    ).rename_axis(columns=None).reset_index()
    rename = {m: f"prob_{m}_temporal" for m in MODELS}
    temp_wide = temp_wide.rename(columns=rename)

    merged = out.merge(temp_wide, on="encounter_id", how="left")
    merged["is_last_day"] = merged["prob_logreg_temporal"].notna().astype(int)

    # Sanity: true_label consistency between OOF and temporal for last-day rows
    if not temp.empty:
        true_lookup = (temp[["encounter_id", "true_label"]]
                       .drop_duplicates("encounter_id")
                       .set_index("encounter_id")["true_label"])
        mask = merged["is_last_day"] == 1
        ours = merged.loc[mask].set_index("encounter_id")["true_label"]
        theirs = true_lookup.loc[ours.index]
        mismatches = (ours != theirs).sum()
        if mismatches:
            print(f"  WARN: {mismatches} encounters with mismatched true_label "
                  "between OOF and temporal sources")

    # Stable column order
    cols = (
        ["encounter_id", "true_label"]
        + [f"prob_{m}_cv_oof" for m in MODELS]
        + [f"prob_{m}_temporal" for m in MODELS]
        + ["is_last_day"]
    )
    return merged[cols]


def main() -> None:
    for name, prefix, oof_csv, temp_csv, out_name in TASKS:
        df = consolidate(prefix, oof_csv, temp_csv, out_name)
        out_path = DERIVED / out_name
        df.to_csv(out_path, index=False)
        n = len(df)
        n_last = int(df["is_last_day"].sum())
        n_pos = int((df["true_label"]
                     == df["true_label"].mode().iloc[0]).sum())
        # Report positive-class prevalence for whichever class is positive
        pos = (df["true_label"].value_counts(normalize=True)
               .sort_values(ascending=False).iloc[0])
        print(f"Wrote {out_path.name}: n={n}, last-day n={n_last}, "
              f"majority-class prevalence={pos:.2f}")


if __name__ == "__main__":
    main()
