"""Load manually-annotated ground-truth labels from
data/Task1_Two_Tier_Input_Data.csv and emit derived/ground_truth.csv
with a stable encounter_id, ground_truth_drug (int 0-3), and
ground_truth_drug_name (str) schema.

Label mapping (taken from the source file's column order, where the
final_p_* probabilities are written as p_no_drug, p_kraken, p_triton,
p_coral — making the integer encoding canonical):

    0  ->  None            (no festival drug)
    1  ->  Kraken Candy    (sympathomimetic)
    2  ->  Triton Tabs     (sedative-hypnotic)
    3  ->  Coral Dust      (hallucinogenic)

Per the hackathon brief these labels are the manually-validated
gold standard and supersede our 10-agent LLM consensus for any
supervised-learning step.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "Task1_Two_Tier_Input_Data.csv"
OUT = ROOT / "derived" / "ground_truth.csv"

LABEL_NAMES = {0: "None", 1: "Kraken Candy", 2: "Triton Tabs",
                3: "Coral Dust"}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing: {SRC}")

    df = pd.read_csv(SRC)
    required = {"encounter_id", "ground_truth_drug"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Source is missing columns: {missing}")

    out = df[["encounter_id", "ground_truth_drug"]].copy()
    out["ground_truth_drug_name"] = out["ground_truth_drug"].map(LABEL_NAMES)
    unknown = out[out["ground_truth_drug_name"].isna()]
    if len(unknown):
        raise SystemExit(f"Unknown label codes in source: "
                          f"{sorted(unknown['ground_truth_drug'].unique())}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Source: {SRC}  ({len(df)} rows)")
    print(f"Wrote:  {OUT}  ({len(out)} rows)\n")

    print("Class distribution (ground_truth_drug):")
    counts = out["ground_truth_drug_name"].value_counts()
    for name in ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]:
        n = int(counts.get(name, 0))
        pct = n / len(out) * 100
        print(f"  {name:14s}  {n:>3d}  ({pct:5.1f}%)")

    # Quick agreement diagnostic vs the LLM-consensus argmax
    probs_path = ROOT / "derived" / "probs_avg.csv"
    if probs_path.exists():
        probs = pd.read_csv(probs_path, keep_default_na=False,
                             na_values=[""])
        merged = out.merge(
            probs[["encounter_id", "argmax_class"]],
            on="encounter_id", how="left",
        )
        merged["match"] = (merged["ground_truth_drug_name"]
                            == merged["argmax_class"])
        agree = merged["match"].mean() * 100
        print(f"\nAgreement with 10-agent LLM consensus argmax: "
              f"{agree:.1f}% ({int(merged['match'].sum())}/{len(merged)})")
        print("\nConfusion (rows = ground truth, cols = LLM argmax):")
        ct = pd.crosstab(merged["ground_truth_drug_name"],
                          merged["argmax_class"], margins=True)
        print(ct.to_string())


if __name__ == "__main__":
    main()
