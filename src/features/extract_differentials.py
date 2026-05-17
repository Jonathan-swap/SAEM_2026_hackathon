"""Construct triage <-> 4h differential features.

For every vital and POC lab measured BOTH at triage (minute 0) and
during the 0-4h ED course (or at the 4h reassessment), emit:
  - diff_<x>        = v_4h - v_triage
  - abs_diff_<x>    = |v_4h - v_triage|
  - pct_change_<x>  = diff_<x> / max(|v_triage|, eps)
  - direction_<x>   = sign(diff_<x>) in {-1, 0, +1}

Plus a few composite stability flags:
  - n_vitals_worsening   = count of monitored vitals trending wrong way
  - any_vital_crit_at_4h = >=1 vital crossed into critical band at 4h
                           that was normal at triage
  - supp_o2_escalated    = supplemental_oxygen went from 0 -> 1

Only emits Task-2 features. Triage CSV is untouched. Then merges
into features_fourh.csv (overwriting any prior diff_*/pct_change_*
columns so reruns are idempotent).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

# Vitals paired at triage <-> 4h reassessment
VITAL_PAIRS = [
    # (short_name, triage_col, fourh_col)
    ("hr",   "triage_heart_rate",                       "ed_course_reassessment_4h.heart_rate_4h"),
    ("rr",   "triage_respiratory_rate",                 "ed_course_reassessment_4h.respiratory_rate_4h"),
    ("sbp",  "triage_snapshot.systolic_bp",             "ed_course_reassessment_4h.systolic_bp_4h"),
    ("dbp",  "triage_snapshot.diastolic_bp",            "ed_course_reassessment_4h.diastolic_bp_4h"),
    ("spo2", "triage_snapshot.oxygen_saturation",       "ed_course_reassessment_4h.oxygen_saturation_4h"),
    ("temp", "triage_temperature_c",                    "ed_course_reassessment_4h.temperature_c_4h"),
    ("gcs",  "triage_gcs",                              "ed_course_reassessment_4h.gcs_4h"),
    ("supp_o2", "triage_supplemental_oxygen",
                "ed_course_reassessment_4h.supplemental_oxygen_4h"),
]

# Lab pairs: triage POC iStat (minute 0) <-> 4h labs_timeseries
# We rely on the lts_*_last value that extract_time_features.py already wrote.
LAB_PAIRS = [
    # (short, triage_col, fourh_col)
    ("glucose",   "triage_lab_glucose",   "lts_poct_glucose_last"),
    ("ph",        "triage_lab_ph",        "lts_vbg_ph_last"),
    ("sodium",    "triage_lab_sodium",    "lts_bmp_sodium_last"),
    ("potassium", "triage_lab_potassium", "lts_bmp_potassium_last"),
]

# Critical bands (same as time-features script)
CRITICAL = {
    "hr": (40, 130),
    "rr": (8, 24),
    "sbp": (90, 220),
    "dbp": (50, 110),
    "spo2": (91, 101),
    "temp": (35.0, 39.0),
    "gcs": (13, 16),
}

EPS = 1e-6


def add_pair_diff(df: pd.DataFrame, short: str,
                   col_triage: str, col_4h: str) -> list[str]:
    """Append diff/abs_diff/pct_change/direction columns. Returns names."""
    added: list[str] = []
    if col_triage not in df.columns or col_4h not in df.columns:
        return added
    v_t = pd.to_numeric(df[col_triage], errors="coerce")
    v_4 = pd.to_numeric(df[col_4h], errors="coerce")
    diff = v_4 - v_t
    df[f"diff_{short}"] = diff
    df[f"abs_diff_{short}"] = diff.abs()
    df[f"pct_change_{short}"] = diff / v_t.abs().clip(lower=EPS)
    df[f"direction_{short}"] = np.sign(diff).fillna(0).astype(int)
    added.extend([f"diff_{short}", f"abs_diff_{short}",
                   f"pct_change_{short}", f"direction_{short}"])
    return added


def add_composites(df: pd.DataFrame) -> list[str]:
    """Stability composites that summarize the differential pattern."""
    added: list[str] = []

    # n_vitals_worsening — count of monitored vitals trending toward critical
    # Define "wrong direction" per vital:
    #   hr: increased >0  -> worsening (tachycardia trend)
    #   rr: any large change -> instability
    #   sbp: decreased -> worsening (hypotension trend)
    #   spo2: decreased -> worsening
    #   gcs: decreased -> worsening (mental status decline)
    #   temp: any large change > 0.5C -> worsening
    n_worse = pd.Series(0, index=df.index)
    if "direction_hr" in df.columns:
        n_worse += (df["direction_hr"] == 1).astype(int)
    if "direction_sbp" in df.columns:
        n_worse += (df["direction_sbp"] == -1).astype(int)
    if "direction_spo2" in df.columns:
        n_worse += (df["direction_spo2"] == -1).astype(int)
    if "direction_gcs" in df.columns:
        n_worse += (df["direction_gcs"] == -1).astype(int)
    if "abs_diff_temp" in df.columns:
        n_worse += (df["abs_diff_temp"] > 0.5).astype(int)
    df["n_vitals_worsening"] = n_worse
    added.append("n_vitals_worsening")

    # any_vital_crit_at_4h_new — crossed into critical at 4h, normal at triage
    def crit(value: float, lo: float, hi: float) -> bool:
        if pd.isna(value):
            return False
        return value < lo or value > hi

    new_crit = pd.Series(0, index=df.index, dtype=int)
    for short, (lo, hi) in CRITICAL.items():
        col_t = dict((s, t) for s, t, _ in VITAL_PAIRS).get(short)
        col_4 = dict((s, f) for s, _, f in VITAL_PAIRS).get(short)
        if not col_t or not col_4 or col_t not in df.columns:
            continue
        triage_crit = df[col_t].apply(lambda v: crit(v, lo, hi))
        fourh_crit = df[col_4].apply(lambda v: crit(v, lo, hi))
        new_crit += (~triage_crit & fourh_crit).astype(int)
    df["any_vital_crit_at_4h_new"] = (new_crit > 0).astype(int)
    df["n_vitals_crit_at_4h_new"] = new_crit
    added.extend(["any_vital_crit_at_4h_new", "n_vitals_crit_at_4h_new"])

    # supp_o2_escalated
    if "triage_supplemental_oxygen" in df.columns and \
       "ed_course_reassessment_4h.supplemental_oxygen_4h" in df.columns:
        t = df["triage_supplemental_oxygen"].fillna(0).astype(int)
        f = df["ed_course_reassessment_4h.supplemental_oxygen_4h"].fillna(0).astype(int)
        df["supp_o2_escalated"] = ((t == 0) & (f == 1)).astype(int)
        df["supp_o2_weaned"] = ((t == 1) & (f == 0)).astype(int)
        added.extend(["supp_o2_escalated", "supp_o2_weaned"])

    return added


def main() -> None:
    src = DERIVED / "features_fourh.csv"
    df = pd.read_csv(src)
    print(f"Loaded {src.name}: {df.shape}")

    # Drop any prior diff_*/abs_diff_*/pct_change_*/direction_* columns
    prior = [c for c in df.columns
             if c.startswith(("diff_", "abs_diff_", "pct_change_",
                               "direction_"))
             or c in {"n_vitals_worsening", "any_vital_crit_at_4h_new",
                       "n_vitals_crit_at_4h_new",
                       "supp_o2_escalated", "supp_o2_weaned"}]
    if prior:
        print(f"Dropping {len(prior)} prior differential columns "
              f"(idempotent rerun)")
        df = df.drop(columns=prior)

    added_total: list[str] = []
    for short, ct, cf in VITAL_PAIRS:
        added = add_pair_diff(df, short, ct, cf)
        if added:
            print(f"  vital pair {short:8s}: +{len(added)} cols")
        added_total.extend(added)
    for short, ct, cf in LAB_PAIRS:
        added = add_pair_diff(df, short, ct, cf)
        if added:
            print(f"  lab   pair {short:8s}: +{len(added)} cols")
        added_total.extend(added)

    composites = add_composites(df)
    print(f"  composites:                +{len(composites)} cols")
    added_total.extend(composites)

    df.to_csv(src, index=False)
    print(f"\nWrote {df.shape[0]} rows x {df.shape[1]} cols -> {src}")
    print(f"Added {len(added_total)} differential features.")

    # Quick coverage / sanity
    print("\nNon-null counts for new differentials (out of "
          f"{len(df)} rows):")
    for col in added_total[:8]:
        nn = int(df[col].notna().sum())
        print(f"  {col:25s} non_null={nn}")
    print("  ...")


if __name__ == "__main__":
    main()
