"""Merge the 10-agent 5-class probability CSVs for Phase-2 into a
single consensus file.

Reads `derived/phase2/probs_1.csv` .. `probs_10.csv` (each with
columns [encounter_id, p_none, p_kraken, p_triton, p_coral,
p_siren_spark]), validates schema + row-sum to 1.0, averages across
agents, and writes `derived/phase2/probs_avg.csv` plus a per-agent
agreement diagnostic.

Outputs:
  derived/phase2/probs_avg.csv
      columns: encounter_id, p_none, p_kraken, p_triton, p_coral,
               p_siren_spark, argmax_class, max_prob, entropy_nats
  derived/phase2/probs_agent_agreement.csv
      per-row standard deviation across the 10 agents for each class
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PHASE2 = ROOT / "derived" / "phase2"

CLASSES = ["p_none", "p_kraken", "p_triton", "p_coral", "p_siren_spark"]
NAMES = {"p_none": "None", "p_kraken": "Kraken Candy",
         "p_triton": "Triton Tabs", "p_coral": "Coral Dust",
         "p_siren_spark": "Siren Spark"}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ("encounter_id", *CLASSES) if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    sums = df[CLASSES].sum(axis=1)
    bad = (sums - 1.0).abs() > 0.005
    if bad.any():
        n_bad = int(bad.sum())
        print(f"  WARN {path.name}: {n_bad} rows don't sum to 1 — re-normalising")
        df[CLASSES] = df[CLASSES].div(sums, axis=0)
    return df


def main() -> None:
    files = sorted(PHASE2.glob("probs_*.csv"))
    # Filter to numeric N (exclude probs_avg.csv etc.)
    keep = []
    for f in files:
        stem = f.stem  # "probs_1"
        if stem.startswith("probs_") and stem.split("_", 1)[1].isdigit():
            keep.append(f)
    files = keep
    if not files:
        raise SystemExit(f"No probs_<N>.csv files found in {PHASE2}")
    print(f"Discovered {len(files)} agent files: "
          f"{[int(f.stem.split('_', 1)[1]) for f in files]}")

    agents = []
    for f in files:
        df = _load(f)
        agents.append(df)
        print(f"  {f.name}: {len(df)} rows")

    # Align on encounter_id (every agent must have the same rows)
    ids0 = agents[0]["encounter_id"].tolist()
    for f, df in zip(files, agents):
        assert df["encounter_id"].tolist() == ids0, \
            f"{f.name} has different encounter ordering"

    stack = np.stack([df[CLASSES].to_numpy(dtype=float) for df in agents],
                      axis=0)  # (n_agents, n_enc, 5)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=0)

    out = pd.DataFrame({"encounter_id": ids0})
    for j, c in enumerate(CLASSES):
        out[c] = mean[:, j]
    out["argmax_class"] = out[CLASSES].idxmax(axis=1).map(NAMES)
    out["max_prob"] = out[CLASSES].max(axis=1)
    eps = 1e-12
    p = mean.copy()
    p[p < eps] = eps
    out["entropy_nats"] = -(p * np.log(p)).sum(axis=1)

    avg_path = PHASE2 / "probs_avg.csv"
    out.to_csv(avg_path, index=False)
    print(f"\nWrote consensus: {avg_path}  ({len(out)} rows)")

    # Per-row agent agreement diagnostic
    agree = pd.DataFrame({"encounter_id": ids0})
    for j, c in enumerate(CLASSES):
        agree[f"{c}_std"] = std[:, j]
    agree.to_csv(PHASE2 / "probs_agent_agreement.csv", index=False)
    print(f"Wrote agreement: {PHASE2 / 'probs_agent_agreement.csv'}")

    # Headline distribution
    print(f"\n=== Consensus argmax distribution ===")
    dist = out["argmax_class"].value_counts().sort_index().to_dict()
    for k, v in dist.items():
        print(f"  {k:<14s} {v:>4d} ({v / len(out) * 100:5.1f}%)")

    print(f"\n=== Mean class probability (across all Phase-2 encounters) ===")
    for c in CLASSES:
        print(f"  {NAMES[c]:<14s} {mean[:, CLASSES.index(c)].mean():.4f}  "
              f"(mean cross-agent SD = "
              f"{std[:, CLASSES.index(c)].mean():.4f})")

    print(f"\n=== Confidence buckets (max_prob across all 5 classes) ===")
    buckets = pd.cut(out["max_prob"],
                      bins=[0, 0.25, 0.4, 0.6, 0.8, 1.0],
                      labels=["<0.25", "0.25-0.40", "0.40-0.60",
                              "0.60-0.80", ">=0.80"])
    print(buckets.value_counts().sort_index().to_string())

    print(f"\n=== Mean entropy: {out['entropy_nats'].mean():.4f} nats  "
          f"(uniform 5-class = {np.log(5):.4f})")


if __name__ == "__main__":
    main()
