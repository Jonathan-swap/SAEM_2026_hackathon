"""Audit script — check the actual minute range in the 4h timeseries.

If the source `Four_Hour_Data` sheet contains any record with
minute > 240, the time-feature extractor silently leaks post-4h
signal into Task-2 features.

Outputs a scalar summary — no row data printed. Safe per project
privacy rules.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "data" / "Hackathon_Data_Release_1_SHARE.xlsx"


def safe_parse(s: object) -> list[dict]:
    if not isinstance(s, str) or not s.strip() or s.strip() == "[]":
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def minute_stats(series: pd.Series, label: str) -> dict:
    all_minutes = []
    n_records_per_encounter = []
    n_over_240 = 0
    encounters_with_over_240 = 0
    for cell in series:
        rows = safe_parse(cell)
        ms = [r["minute"] for r in rows
              if isinstance(r, dict) and "minute" in r
              and isinstance(r["minute"], (int, float))]
        n_records_per_encounter.append(len(ms))
        all_minutes.extend(ms)
        over = sum(1 for m in ms if m > 240)
        n_over_240 += over
        if over > 0:
            encounters_with_over_240 += 1
    s = pd.Series(all_minutes, dtype=float)
    return {
        "label": label,
        "n_encounters": len(series),
        "n_records_total": len(s),
        "n_records_over_240": n_over_240,
        "n_encounters_with_post_4h": encounters_with_over_240,
        "min_minute": float(s.min()) if not s.empty else float("nan"),
        "max_minute": float(s.max()) if not s.empty else float("nan"),
        "p99_minute": float(s.quantile(0.99)) if not s.empty else float("nan"),
        "p95_minute": float(s.quantile(0.95)) if not s.empty else float("nan"),
        "median_records_per_encounter": float(
            pd.Series(n_records_per_encounter).median()),
    }


def main() -> None:
    fourh = pd.read_excel(XLSX, sheet_name="Four_Hour_Data",
                          engine="openpyxl")
    print(f"Loaded Four_Hour_Data: {fourh.shape[0]} rows")
    print()

    results = []
    for col, label in [
        ("ed_course.vitals_timeseries", "vitals_timeseries"),
        ("ed_course.labs_timeseries",   "labs_timeseries"),
        ("ed_course.interventions",     "interventions"),
    ]:
        results.append(minute_stats(fourh[col], label))

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    if df["n_records_over_240"].sum() > 0:
        print("*** LEAKAGE RISK: at least one timeseries record has "
              "minute > 240 ***")
    else:
        print("OK: every timeseries record has minute <= 240.")


if __name__ == "__main__":
    main()
