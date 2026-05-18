"""EDA for the v6-derived features.

Ranks every new feature by:
  - Mutual information vs Task-1 target (ground_truth_drug, 4-class)
  - Mutual information vs Task-2 target (encounter_disposition_label,
    3-class, drug-positive cohort)
  - Per-drug-class fraction-positive (for binary features) — directly
    auditable against the v6 discriminator hierarchy

Prints rankings and writes derived/v6_feature_audit.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
REPORT = DERIVED / "v6_feature_audit.md"

V6_FEATURE_PREFIXES = (
    "pe_", "peak_lactate_", "peak_cpk_", "peak_troponin_",
    "peak_hr_", "peak_temp_", "kraken_severity_anchor",
    "all_peak_labs_normal",
    "triage_chief_", "note_arousal_density", "note_inward_density",
    "note_perceptual_density", "triage_ag_above_", "triage_ph_above_",
    "triage_hr_above_", "triage_temp_above_", "triage_glucose_above_",
    "triage_sympathomimetic_combo",
)

DRUG_CLASSES = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]
DISPO_CLASSES = ["Discharge", "Floor", "ICU"]


def is_v6_feature(col: str) -> bool:
    return any(col.startswith(p) or col == p for p in V6_FEATURE_PREFIXES)


def rank_mi(X: pd.DataFrame, y: np.ndarray, label: str) -> pd.DataFrame:
    """Per-column MI against y. Median-imputes NaN before MI."""
    X = X.copy()
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    mi = mutual_info_classif(X.to_numpy(), y, discrete_features="auto",
                              random_state=42)
    return (pd.DataFrame({"feature": X.columns, f"mi_{label}": mi})
              .sort_values(f"mi_{label}", ascending=False)
              .reset_index(drop=True))


def class_fraction_positive(X: pd.DataFrame,
                             truth: np.ndarray,
                             classes: list[str]) -> pd.DataFrame:
    """For each (binary) feature, fraction positive within each class."""
    rows = []
    for col in X.columns:
        if not set(pd.Series(X[col]).dropna().unique()).issubset({0, 1}):
            continue
        rec = {"feature": col}
        for c in classes:
            mask = truth == c
            if mask.sum() == 0:
                rec[c] = float("nan")
            else:
                rec[c] = float(X.loc[mask, col].fillna(0).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in fmt:
                cells.append(fmt[c].format(v) if pd.notna(v) else "—")
            elif isinstance(v, float):
                cells.append("—" if pd.isna(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    print("Loading features + targets...")
    triage = pd.read_csv(DERIVED / "features_triage.csv")
    fourh = pd.read_csv(DERIVED / "features_fourh.csv")
    gt = pd.read_csv(DERIVED / "ground_truth.csv")
    gt["truth_name"] = gt["ground_truth_drug_name"].fillna("None")

    # --- TASK 1 ---
    print("\n=== TASK 1 (triage horizon) ===")
    t1 = triage.merge(gt[["encounter_id", "ground_truth_drug",
                           "truth_name"]], on="encounter_id")
    new_cols_t1 = [c for c in t1.columns if is_v6_feature(c)]
    print(f"v6 features present in features_triage.csv: {len(new_cols_t1)}")
    X_t1 = t1[new_cols_t1].apply(pd.to_numeric, errors="coerce")
    y_t1 = t1["ground_truth_drug"].astype(int).to_numpy()
    mi_t1 = rank_mi(X_t1, y_t1, "task1")
    print("\nTop-15 by MI vs Task-1 target:")
    print(mi_t1.head(15).to_string(index=False))

    frac_t1 = class_fraction_positive(X_t1, t1["truth_name"].to_numpy(),
                                        DRUG_CLASSES)
    if not frac_t1.empty:
        # Show fractions sorted by Kraken signal strength first
        frac_t1 = frac_t1.merge(mi_t1, on="feature").sort_values(
            "mi_task1", ascending=False)
        print("\nBinary features — fraction-positive by drug class:")
        print(frac_t1.head(15).to_string(index=False))

    # --- TASK 2 ---
    print("\n\n=== TASK 2 (4h horizon, drug-positive cohort) ===")
    t2 = (fourh.merge(gt[["encounter_id", "ground_truth_drug",
                            "truth_name"]], on="encounter_id"))
    t2 = t2[t2["ground_truth_drug"] != 0].reset_index(drop=True)
    new_cols_t2 = [c for c in t2.columns if is_v6_feature(c)]
    print(f"v6 features present in features_fourh.csv: {len(new_cols_t2)}")
    X_t2_drug = t2[new_cols_t2].apply(pd.to_numeric, errors="coerce")
    # Task-2 target = disposition (Discharge/Floor/ICU)
    y_dispo = (t2["encounter_disposition_label"]
                 .map({c: i for i, c in enumerate(DISPO_CLASSES)})
                 .astype(int).to_numpy())
    mi_dispo = rank_mi(X_t2_drug, y_dispo, "task2_dispo")
    print("\nTop-15 by MI vs disposition (drug-positive cohort):")
    print(mi_dispo.head(15).to_string(index=False))

    # Also: MI vs DRUG class within Task 2 features (validates that
    # the new features actually discriminate the toxidromes when the
    # 4h horizon is available — this is what the agent labelling
    # pipeline needs to align with manual ground truth)
    y_drug = t2["ground_truth_drug"].astype(int).to_numpy()
    mi_t2_drug = rank_mi(X_t2_drug, y_drug, "task2_drug")
    print("\nTop-15 by MI vs DRUG-class (drug-positive cohort):")
    print(mi_t2_drug.head(15).to_string(index=False))

    frac_t2 = class_fraction_positive(X_t2_drug, t2["truth_name"].to_numpy(),
                                        ["Kraken Candy", "Triton Tabs",
                                         "Coral Dust"])
    frac_t2_audit = frac_t2.merge(mi_t2_drug, on="feature").sort_values(
        "mi_task2_drug", ascending=False)
    print("\nBinary v6 features — fraction-positive by drug class (drug-positive cohort):")
    print(frac_t2_audit.head(20).to_string(index=False))

    # --- Write markdown report ---
    lines = []
    lines.append("# v6 feature audit\n")
    lines.append(f"Generated by `src/eda/eda_v6_features.py`. "
                 f"Inputs: `features_triage.csv` ({len(t1)} rows), "
                 f"`features_fourh.csv` drug-positive cohort "
                 f"({len(t2)} rows).\n")

    lines.append("## Task 1 — top features by MI vs `ground_truth_drug`")
    lines.append("")
    lines.append(md_table(mi_t1.head(20),
                          fmt={"mi_task1": "{:.4f}"}))
    lines.append("")

    if not frac_t1.empty:
        lines.append("### Task 1 — class-conditional fraction-positive (top 15 by MI)\n")
        cols = ["feature", "None", "Kraken Candy", "Triton Tabs",
                "Coral Dust", "mi_task1"]
        lines.append(md_table(frac_t1[cols].head(15),
                              fmt={"None": "{:.2f}",
                                   "Kraken Candy": "{:.2f}",
                                   "Triton Tabs": "{:.2f}",
                                   "Coral Dust": "{:.2f}",
                                   "mi_task1": "{:.4f}"}))
        lines.append("")

    lines.append("## Task 2 — top features by MI vs `encounter_disposition_label`")
    lines.append("(drug-positive cohort, 3 classes: Discharge/Floor/ICU)\n")
    lines.append(md_table(mi_dispo.head(20),
                          fmt={"mi_task2_dispo": "{:.4f}"}))
    lines.append("")

    lines.append("## Task 2 (sanity check) — top features by MI vs `ground_truth_drug`")
    lines.append("(drug-positive cohort, 3 classes: Kraken/Triton/Coral)\n")
    lines.append(md_table(mi_t2_drug.head(20),
                          fmt={"mi_task2_drug": "{:.4f}"}))
    lines.append("")

    lines.append("### Task 2 — class-conditional fraction-positive (drug-positive cohort)\n")
    cols_t2 = ["feature", "Kraken Candy", "Triton Tabs",
                "Coral Dust", "mi_task2_drug"]
    lines.append(md_table(frac_t2_audit[cols_t2].head(20),
                          fmt={"Kraken Candy": "{:.2f}",
                               "Triton Tabs": "{:.2f}",
                               "Coral Dust": "{:.2f}",
                               "mi_task2_drug": "{:.4f}"}))
    lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {REPORT}")


if __name__ == "__main__":
    main()
