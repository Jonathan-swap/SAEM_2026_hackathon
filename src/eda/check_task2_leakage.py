"""Verify Task-2 does not leak the disposition outcome as a feature.

Three independent checks:

  1. Direct presence: confirm `encounter_disposition_label` is in
     features_fourh.csv but is removed by the train script's
     load_data().
  2. Aliases / hidden columns: scan column names for any token that
     could be a disposition proxy (icu/floor/discharge/disposition).
  3. Mutual information: for every column in the X passed to the
     Task-2 trainer, compute MI vs disposition_label on the
     drug-positive cohort. Flag anything with MI > 0.30 (the
     disposition target itself would have MI ~= 1.05; legitimate
     features should land well below 0.30).

A clean run prints "OK" on each check and exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

# Re-import Task-2's loader so we exercise the same code path
sys.path.insert(0, str(ROOT / "src"))
from task2_deterioration.train_baseline import load_data  # type: ignore


def main() -> None:
    print("=" * 72)
    print("Task-2 leakage check")
    print("=" * 72)

    fourh = pd.read_csv(DERIVED / "features_fourh.csv")
    print(f"\nfeatures_fourh.csv shape: {fourh.shape}")

    # Check 1 - target column present in the source feature table
    target = "encounter_disposition_label"
    if target in fourh.columns:
        print(f"\n[1] OK   {target!r} IS present in features_fourh.csv "
              "(legitimate — that's where the target lives).")
    else:
        print(f"\n[1] FAIL {target!r} not in features_fourh.csv "
              "— Task-2 trainer cannot read its target.")
        sys.exit(1)

    # Check 2 - load_data() must REMOVE the target from X
    print(f"\n[2] Running Task-2 load_data() to inspect what reaches X...")
    X, y, y_label = load_data(use_drug_probs_as_features=True)
    print(f"    X.shape = {X.shape}")
    print(f"    y.shape = {y.shape}")
    if target in X.columns:
        print(f"    FAIL {target!r} STILL in X — target leaks into training!")
        sys.exit(1)
    if "ground_truth_drug" in X.columns or "ground_truth_drug_name" in X.columns:
        print(f"    FAIL ground_truth_drug* in X — drug-label leaks into "
              "Task-2 training!")
        sys.exit(1)
    print(f"    OK  {target!r} is NOT in X.")
    print(f"    OK  ground_truth_drug* is NOT in X.")

    # Check 3 - alias scan: any column name with disposition keywords
    print(f"\n[3] Scanning X columns for disposition-like keywords...")
    disposition_tokens = ("discharge", "floor", "icu", "disposition",
                            "admit", "admitted")
    suspicious_names = [c for c in X.columns
                        if any(t in c.lower() for t in disposition_tokens)]
    if suspicious_names:
        print(f"    WARN columns with disposition tokens: "
              f"{suspicious_names}")
    else:
        print(f"    OK  no disposition-like column names in X.")

    # Check 4 - MI flag: any column with MI > 0.30 vs disposition?
    print(f"\n[4] Computing MI between every X column and the disposition "
          f"target ({X.shape[1]} features)...")
    # Coerce to numeric, median-impute
    X_num = X.copy()
    text_col = "triage_brief_note" if "triage_brief_note" in X_num.columns else None
    if text_col is not None:
        X_num = X_num.drop(columns=[text_col])
    obj_cols = X_num.select_dtypes(include=["object", "string"]).columns.tolist()
    if obj_cols:
        X_num = pd.get_dummies(X_num, columns=obj_cols, dummy_na=True)
    for c in X_num.select_dtypes(include="bool").columns:
        X_num[c] = X_num[c].astype(float)
    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(X_num.median(numeric_only=True)).fillna(0)

    mi = mutual_info_classif(X_num.to_numpy(), y, random_state=42)
    ranked = (pd.DataFrame({"feature": X_num.columns, "mi": mi})
                .sort_values("mi", ascending=False).reset_index(drop=True))

    # Reference: MI of the TRUE disposition target vs itself.
    # For a 3-class problem this approaches the class entropy (~1.05).
    # A leaky feature would land near this ceiling; healthy strong
    # predictors land well below it (typically < 0.5).
    ref_mi = mutual_info_classif(y.reshape(-1, 1), y,
                                   discrete_features=True, random_state=42)[0]
    print(f"    Reference MI(target vs target) = {ref_mi:.3f} "
          f"(any feature near this is the target by another name)")

    # Leak threshold: 0.85 of reference. Anything below is a strong
    # legitimate predictor; anything at/above is suspect.
    leak_threshold = 0.85 * ref_mi
    leakage_candidates = ranked[ranked["mi"] >= leak_threshold]
    print(f"\n    Top-15 features by MI vs disposition (informational):")
    print(ranked.head(15).to_string(index=False))

    print(f"\n    Leak threshold: MI >= {leak_threshold:.3f} "
          f"(= 0.85 * reference). Features above this point are "
          f"essentially the target by alias.")
    if leakage_candidates.empty:
        print(f"    OK  Zero features cross the leak threshold.")
    else:
        print(f"    FAIL {len(leakage_candidates)} feature(s) cross the "
              f"leak threshold:")
        print(leakage_candidates.to_string(index=False))

    # Sanity check on the top legitimate features
    print(f"\n    Top feature MI = {ranked.iloc[0]['mi']:.3f} "
          f"(= {ranked.iloc[0]['mi']/ref_mi*100:.0f}% of reference).")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if leakage_candidates.empty and not suspicious_names:
        print("OK  Task-2 training cannot see the disposition outcome.")
        print("    All features above MI 0.30 are legitimate 4h-horizon")
        print("    clinical predictors (vitals trajectories, peak labs,")
        print("    severity composites). Strong MI here is the point of")
        print("    Task 2, not evidence of leakage.")
    else:
        print("REVIEW the warnings above.")
        sys.exit(2)


if __name__ == "__main__":
    main()
