"""Feature cleanup + candidate merge.

Final step in the feature pipeline. Idempotent; safe to re-run.

Operations applied in order to features_triage.csv and
features_fourh.csv:

1. Drop columns with zero variance (constant for every row).
2. Drop near-constant columns (>=99% same value) — see THRESHOLD.
   Stricter than EDA reports — keeping a column with up to 1% novelty
   still helps tree models on rare patterns.
3. Resolve any remaining merge-suffix duplicates (`*_x`/`*_y`) by
   keeping the `_x` copy and dropping `_y` (they are identical when
   they came from the bug we just fixed in extract_time_features).
4. Merge derived/exploratory_features.csv (committed candidate
   features from eda_descriptive.py) into BOTH feature tables —
   the candidates are derived from triage-time inputs only, so
   they're safe at the triage horizon.
5. Re-run the leakage sentinel.

Whitelist: never drop encounter_id, encounter_arrival_date,
encounter_disposition_label.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

NEAR_CONST_THRESHOLD = 0.99  # drop if >=99% of values are identical
WHITELIST = {"encounter_id", "encounter_arrival_date",
              "encounter_disposition_label"}


def find_constants(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if col in WHITELIST:
            continue
        s = df[col].dropna()
        if s.empty:
            out.append(col)  # all NaN -> drop
            continue
        if s.nunique() == 1:
            out.append(col)
    return out


def find_near_constants(df: pd.DataFrame, thresh: float) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for col in df.columns:
        if col in WHITELIST:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        top_share = s.value_counts(normalize=True).iloc[0]
        if top_share >= thresh:
            out.append((col, float(top_share)))
    return out


def find_xy_duplicates(df: pd.DataFrame) -> list[str]:
    """Find `<name>_y` columns when `<name>_x` also exists."""
    cols = set(df.columns)
    drop: list[str] = []
    for col in df.columns:
        if col.endswith("_y"):
            base = col[:-2]
            if f"{base}_x" in cols:
                drop.append(col)
    return drop


def rename_x_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """After dropping the _y twins, rename the surviving _x to base name."""
    rename: dict[str, str] = {}
    for col in df.columns:
        if col.endswith("_x"):
            base = col[:-2]
            if base not in df.columns and f"{base}_y" not in df.columns:
                rename[col] = base
    if rename:
        df = df.rename(columns=rename)
    return df


def cleanup_one(path: Path) -> None:
    df = pd.read_csv(path)
    n0 = df.shape[1]
    print(f"\n=== {path.name} ===")
    print(f"  start: {df.shape}")

    consts = find_constants(df)
    if consts:
        print(f"  dropping {len(consts)} constant columns "
              f"(showing first 5): {consts[:5]}")
        df = df.drop(columns=consts)

    nc = find_near_constants(df, NEAR_CONST_THRESHOLD)
    if nc:
        names = [n for n, _ in nc]
        print(f"  dropping {len(nc)} near-constant columns "
              f"(>= {NEAR_CONST_THRESHOLD*100:.0f}% same value, "
              f"showing first 5): {[(n, f'{s*100:.1f}%') for n, s in nc[:5]]}")
        df = df.drop(columns=names)

    xy = find_xy_duplicates(df)
    if xy:
        print(f"  dropping {len(xy)} merge-suffix _y duplicates "
              f"(showing first 5): {xy[:5]}")
        df = df.drop(columns=xy)
        df = rename_x_to_canonical(df)

    df.to_csv(path, index=False)
    print(f"  end:   {df.shape}  (removed {n0 - df.shape[1]} columns)")


def merge_candidates() -> None:
    src = DERIVED / "exploratory_features.csv"
    if not src.exists():
        print(f"\n(no {src.name} to merge — skipping)")
        return
    cand = pd.read_csv(src)
    new_cols = [c for c in cand.columns if c != "encounter_id"]
    print(f"\nMerging {len(new_cols)} candidate features from "
          f"{src.name} into both feature tables")
    for target in ("features_triage.csv", "features_fourh.csv"):
        path = DERIVED / target
        df = pd.read_csv(path)
        prior = [c for c in df.columns if c in new_cols]
        if prior:
            df = df.drop(columns=prior)
        merged = df.merge(cand, on="encounter_id", how="left")
        merged.to_csv(path, index=False)
        print(f"  {target}: {merged.shape}")


def leakage_sentinel() -> None:
    path = DERIVED / "features_triage.csv"
    df = pd.read_csv(path)
    forbidden_prefixes = ("vts_", "lts_", "itv_", "xmod_",
                           "stab_", "arc_")
    leaked = [c for c in df.columns
              if c.startswith(forbidden_prefixes) or "_4h" in c
              or "delta_" in c or c.startswith("diff_")
              or c.startswith("abs_diff_") or c.startswith("pct_change_")
              or c.startswith("direction_")]
    leaked = [c for c in leaked if c not in WHITELIST]
    if leaked:
        raise AssertionError(
            f"Leakage in features_triage.csv: {leaked}")
    print("\nOK: leakage sentinel passed on features_triage.csv "
          f"(no post-triage columns leaked)")


def main() -> None:
    for target in ("features_triage.csv", "features_fourh.csv"):
        cleanup_one(DERIVED / target)
    merge_candidates()
    leakage_sentinel()


if __name__ == "__main__":
    main()
