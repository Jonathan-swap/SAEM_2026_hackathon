"""Phase C — merge the three independent agent label CSVs.

Reads:
  derived/labels_A.csv  (subjective / historical view)
  derived/labels_B.csv  (objective / treatment view)
  derived/labels_C.csv  (clinical reasoning view)

Computes pairwise Cohen's kappa, 3-way agreement rate, majority
vote per encounter, and writes derived/derived_labels.csv.

Print: marginal distributions, agreement rates, confusion matrices,
count of split (3-way disagreement) records — these are the rows a
human reviewer would inspect first.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

VALID_LABELS = {"Kraken Candy", "Triton Tabs", "Coral Dust", "None"}


def load_agent_csv(name: str) -> pd.DataFrame:
    path = DERIVED / f"labels_{name}.csv"
    # keep_default_na=False so the literal string "None" survives
    # (otherwise pandas reads "None" as NaN).
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    expected = {"encounter_id", "drug_label", "confidence", "evidence_phrase"}
    missing = expected - set(df.columns)
    assert not missing, f"{path}: missing columns {missing}"
    # Normalize whitespace + handle truly empty labels
    df["drug_label"] = df["drug_label"].astype(str).str.strip()
    df.loc[df["drug_label"].isin(["", "nan", "NaN"]), "drug_label"] = "Unlabeled"
    bad = set(df["drug_label"].unique()) - VALID_LABELS - {"Unlabeled"}
    if bad:
        print(f"  WARNING {path.name}: non-canonical labels seen: {bad}")
    n_unlabeled = (df["drug_label"] == "Unlabeled").sum()
    if n_unlabeled:
        print(f"  WARNING {path.name}: {n_unlabeled} rows with empty label "
              f"-> set to 'Unlabeled'")
    return df.rename(columns={
        "drug_label": f"label_{name}",
        "confidence": f"conf_{name}",
        "evidence_phrase": f"evidence_{name}",
    })


def majority_with_confidence_tiebreak(row: pd.Series) -> tuple[str, str]:
    """Return (majority_label, agreement_tier)."""
    labels = [row["label_A"], row["label_B"], row["label_C"]]
    confs = [row["conf_A"], row["conf_B"], row["conf_C"]]
    counts = Counter(labels)

    if len(counts) == 1:
        return labels[0], "unanimous"

    most_common = counts.most_common()
    if most_common[0][1] == 2:
        return most_common[0][0], "majority"

    # 3-way split: tie-break by highest confidence
    best_idx = max(range(3), key=lambda i: confs[i])
    return labels[best_idx], "split"


def main() -> None:
    print("Loading agent outputs...")
    a = load_agent_csv("A")
    b = load_agent_csv("B")
    c = load_agent_csv("C")

    for name, df in [("A", a), ("B", b), ("C", c)]:
        print(f"  Agent {name}: {len(df)} rows, "
              f"label dist = {df[f'label_{name}'].value_counts().to_dict()}")

    print("\nMerging...")
    merged = a.merge(b, on="encounter_id").merge(c, on="encounter_id")
    assert len(merged) == 261, f"After merge got {len(merged)} rows, expected 261"

    print(f"  Merged: {len(merged)} rows")

    print("\nPairwise Cohen's kappa:")
    for x, y in [("A", "B"), ("B", "C"), ("A", "C")]:
        k = cohen_kappa_score(merged[f"label_{x}"], merged[f"label_{y}"])
        agree = (merged[f"label_{x}"] == merged[f"label_{y}"]).mean()
        print(f"  {x} vs {y}: kappa = {k:.3f}   raw_agreement = {agree:.3f}")

    print("\nPairwise confusion matrices (rows = first agent, cols = second):")
    for x, y in [("A", "B"), ("B", "C"), ("A", "C")]:
        ct = pd.crosstab(merged[f"label_{x}"], merged[f"label_{y}"],
                         margins=True)
        print(f"\n  {x} (rows) vs {y} (cols):")
        print(ct.to_string())

    print("\nApplying majority vote...")
    merged[["majority_label", "agreement_tier"]] = merged.apply(
        majority_with_confidence_tiebreak, axis=1, result_type="expand")
    merged["confidence_mean"] = merged[["conf_A", "conf_B", "conf_C"]].mean(axis=1)

    print(f"\nAgreement tier distribution:")
    print(merged["agreement_tier"].value_counts().to_string())

    print(f"\nMajority label distribution:")
    print(merged["majority_label"].value_counts().to_string())

    n_unanimous = (merged["agreement_tier"] == "unanimous").sum()
    n_majority = (merged["agreement_tier"] == "majority").sum()
    n_split = (merged["agreement_tier"] == "split").sum()
    print(f"\nReviewer triage:")
    print(f"  unanimous (3/3 agree):   {n_unanimous}  -- trust as-is")
    print(f"  majority  (2/3 agree):   {n_majority}  -- spot-check")
    print(f"  split     (3-way disag): {n_split}  -- manual review")
    print(f"  3-way agreement rate:     "
          f"{n_unanimous / len(merged) * 100:.1f}%")

    # Final column order
    cols = (
        ["encounter_id",
         "label_A", "label_B", "label_C",
         "conf_A", "conf_B", "conf_C",
         "majority_label", "agreement_tier", "confidence_mean",
         "evidence_A", "evidence_B", "evidence_C"]
    )
    final = merged[cols]

    out_path = DERIVED / "derived_labels.csv"
    final.to_csv(out_path, index=False)
    print(f"\nWrote {len(final)} rows x {len(cols)} cols to {out_path}")


if __name__ == "__main__":
    main()
