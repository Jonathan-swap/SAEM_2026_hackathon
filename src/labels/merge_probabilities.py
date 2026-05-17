"""Merge per-agent probability distributions into an averaged consensus.

Reads:
  derived/probs_1.csv ... probs_5.csv

Each agent CSV must have columns:
  encounter_id, p_kraken, p_triton, p_coral, p_none

with each row summing to 1.0 (±0.005).

Aggregates by averaging the four probabilities across agents per
encounter, then renormalizes (each row sums to exactly 1.0).

Writes:
  derived/probs_avg.csv with columns:
    encounter_id, p_kraken, p_triton, p_coral, p_none,
    argmax_class, max_prob, entropy,
    p_kraken_std, p_triton_std, p_coral_std, p_none_std  (cross-agent SD)

Plus a diagnostics summary:
  - Marginal mean per class
  - Per-agent agreement (mean SD across rows per class)
  - Argmax distribution
  - Confidence buckets (max_prob >= 0.7 / 0.5-0.7 / 0.25-0.5 / <0.25)
  - Comparison against earlier derived_labels.csv (majority hard label)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
CLASSES = ["p_kraken", "p_triton", "p_coral", "p_none"]
CLASS_NAMES = {"p_kraken": "Kraken Candy", "p_triton": "Triton Tabs",
               "p_coral": "Coral Dust", "p_none": "None"}
TOLERANCE = 0.005


def load_agent(n: int) -> pd.DataFrame:
    path = DERIVED / f"probs_{n}.csv"
    df = pd.read_csv(path)
    expected = ["encounter_id", *CLASSES]
    missing = set(expected) - set(df.columns)
    assert not missing, f"{path}: missing {missing}"
    df = df[expected].copy()
    # Coerce
    for c in CLASSES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # Validate sum-to-one per row
    sums = df[CLASSES].sum(axis=1)
    bad = ((sums - 1.0).abs() > TOLERANCE).sum()
    print(f"  probs_{n}.csv: {len(df)} rows, "
          f"sum-validation pass = {(len(df) - bad)}/{len(df)} "
          f"(failures = {bad})")
    if bad > 0:
        # Renormalize the offenders so downstream math is clean
        df[CLASSES] = df[CLASSES].div(sums, axis=0)
        print(f"    (renormalized {bad} rows so all sums = 1.0 exactly)")
    return df


def discover_agents() -> list[int]:
    """Find all probs_<digit>.csv files in derived/ — excludes probs_avg.csv."""
    import re
    pat = re.compile(r"^probs_(\d+)\.csv$")
    found = []
    for p in DERIVED.glob("probs_*.csv"):
        m = pat.match(p.name)
        if m:
            found.append(int(m.group(1)))
    return sorted(found)


def main() -> None:
    print("Loading agent probability CSVs...")
    agent_ids = discover_agents()
    if not agent_ids:
        raise SystemExit("No probs_<n>.csv files found in derived/")
    print(f"Discovered {len(agent_ids)} agent files: {agent_ids}")
    agents = [load_agent(n) for n in agent_ids]

    # All agents must have identical encounter_id sets (and order)
    ref_ids = agents[0]["encounter_id"].tolist()
    for i, df in enumerate(agents[1:], start=2):
        assert df["encounter_id"].tolist() == ref_ids, (
            f"probs_{i}.csv encounter_id order differs from probs_1.csv")
    print(f"\nAll {len(agents)} agents agree on {len(ref_ids)} encounters in same order.")

    # Stack into a 3D tensor: (n_agents, n_encounters, n_classes)
    stack = np.stack([df[CLASSES].to_numpy() for df in agents], axis=0)
    print(f"Stack shape: agents={stack.shape[0]}, "
          f"encounters={stack.shape[1]}, classes={stack.shape[2]}")

    # Average across agents -> (n_encounters, n_classes)
    avg = stack.mean(axis=0)
    sd = stack.std(axis=0)

    # Renormalize avg (sum may drift fractionally due to floating point)
    avg = avg / avg.sum(axis=1, keepdims=True)

    final = pd.DataFrame({"encounter_id": ref_ids})
    for j, c in enumerate(CLASSES):
        final[c] = avg[:, j].round(6)
    for j, c in enumerate(CLASSES):
        final[f"{c}_std"] = sd[:, j].round(6)
    final["argmax_class"] = final[CLASSES].idxmax(axis=1).map(CLASS_NAMES)
    final["max_prob"] = final[CLASSES].max(axis=1)
    eps = 1e-12
    final["entropy"] = -(avg * np.log(np.clip(avg, eps, 1.0))).sum(axis=1)

    out_path = DERIVED / "probs_avg.csv"
    final.to_csv(out_path, index=False)
    print(f"\nWrote {len(final)} rows to {out_path}")

    print("\n--- Marginal mean probability per class (averaged across all encounters) ---")
    for c in CLASSES:
        print(f"  {CLASS_NAMES[c]:14s}  {final[c].mean():.4f}   "
              f"(SD across agents avg = {final[f'{c}_std'].mean():.4f})")
    print(f"  Total marginal: {final[CLASSES].mean().sum():.4f}  (should be 1.0)")

    print("\n--- Argmax-class distribution (hard label from averaged probs) ---")
    print(final["argmax_class"].value_counts().to_string())

    print("\n--- Confidence buckets (max prob across the 4 classes) ---")
    bins = [0.0, 0.25, 0.5, 0.7, 1.001]
    labels = ["<0.25 (max ~ uniform)", "0.25-0.5", "0.5-0.7", ">=0.7"]
    bucket = pd.cut(final["max_prob"], bins=bins, labels=labels, right=False)
    print(bucket.value_counts().sort_index().to_string())

    print("\n--- Entropy (nats) percentiles ---")
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p}: {np.percentile(final['entropy'], p):.4f}")
    print(f"  Max entropy possible (uniform): {np.log(4):.4f}")

    # Compare against earlier majority hard label
    derived_labels = DERIVED / "derived_labels.csv"
    if derived_labels.exists():
        prior = pd.read_csv(derived_labels, keep_default_na=False,
                             na_values=[""])[["encounter_id",
                                              "majority_label",
                                              "agreement_tier"]]
        joined = final.merge(prior, on="encounter_id", how="left")
        print("\n--- Argmax (new avg prob) vs majority_label (earlier 3-agent hard) ---")
        ct = pd.crosstab(joined["argmax_class"], joined["majority_label"])
        print(ct.to_string())
        match = (joined["argmax_class"] == joined["majority_label"]).mean()
        print(f"\n  Agreement on hard label: {match * 100:.1f}%")

    # Compare against disposition
    triage = pd.read_csv(DERIVED / "features_triage.csv")[
        ["encounter_id", "encounter_disposition_label"]]
    joined2 = final.merge(triage, on="encounter_id", how="left")
    print("\n--- Argmax-class vs disposition ---")
    print(pd.crosstab(joined2["argmax_class"],
                      joined2["encounter_disposition_label"]).to_string())


if __name__ == "__main__":
    main()
