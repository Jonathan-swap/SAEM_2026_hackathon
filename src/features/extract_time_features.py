"""Phase A.3 — Time-relevant feature engineering.

Implements Groups A-G from .claude/plans/05_time_features_plan.md.

Group A — arrival-time features          (Task 1 + Task 2)
Group B — vital-sign trajectory features (Task 2)
Group C — lab trajectory features        (Task 2)
Group D — intervention sequencing        (Task 2)
Group E — cross-modal timing             (Task 2)
Group F — stability / volatility         (Task 2)
Group G — recovery / deterioration arc   (Task 2)

Emits two CSVs and merges them into the existing feature tables:
- derived/time_features_triage.csv  (Group A only)
- derived/time_features_fourh.csv   (Groups A-G)
- derived/features_triage.csv       (re-merged)
- derived/features_fourh.csv        (re-merged)

Runs a leakage sentinel: no _4h / vts_ / lts_ / itv_ / xmod_ /
stab_ / arc_ columns may appear in the triage CSV.
"""
from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "data" / "Hackathon_Data_Release_1_SHARE.xlsx"
DERIVED = ROOT / "derived"

FESTIVAL_START = datetime(2025, 5, 18)


# ---- Parsing ----------------------------------------------------------

def safe_parse(s: object) -> list[dict]:
    if not isinstance(s, str) or not s.strip() or s.strip() == "[]":
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


# ---- Group A — arrival-time features ----------------------------------

def arrival_features(arrival_date: object,
                     same_day_volume: int) -> dict[str, float]:
    """Triage-available time context (Task 1 + Task 2)."""
    if not isinstance(arrival_date, (pd.Timestamp, datetime)):
        try:
            arrival_date = pd.to_datetime(arrival_date)
        except (TypeError, ValueError):
            return {}
    delta_days = (arrival_date.to_pydatetime() - FESTIVAL_START).days \
        if hasattr(arrival_date, "to_pydatetime") else \
        (arrival_date - FESTIVAL_START).days
    dow = arrival_date.weekday()  # 0=Mon
    return {
        "arrival_day_of_festival": delta_days,
        "arrival_dow": dow,
        "arrival_is_weekend": int(dow in (5, 6)),
        "arrival_is_peak_festival_day": int(delta_days in (1, 2, 3)),
        "arrival_same_day_volume": same_day_volume,
    }


# ---- Group B — vital-sign trajectory ---------------------------------

VITAL_COLS = ["heart_rate", "respiratory_rate", "systolic_bp",
              "diastolic_bp", "oxygen_saturation", "temperature_c",
              "gcs", "end_tidal_co2"]

# NEWS-2-inspired critical bounds (rough, for synthetic-data signal).
CRITICAL = {
    "heart_rate":        (40, 130),
    "respiratory_rate":  (8, 24),
    "systolic_bp":       (90, 220),
    "diastolic_bp":      (50, 110),
    "oxygen_saturation": (91, 101),
    "temperature_c":     (35.0, 39.0),
    "gcs":               (13, 16),
    "end_tidal_co2":     (30, 55),
}


def _slope(minutes: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    if np.allclose(minutes, minutes[0]):
        return 0.0
    return float(np.polyfit(minutes, values, 1)[0])


def vital_trajectory(ts_rows: list[dict]) -> dict[str, float]:
    """Group B: per-vital trajectory features."""
    out: dict[str, float] = {"vts_n_records": float(len(ts_rows))}
    if not ts_rows:
        return out

    df = pd.DataFrame(ts_rows)
    if "minute" not in df.columns:
        return out
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    df = df.dropna(subset=["minute"]).sort_values("minute")

    for col in VITAL_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        m = df.loc[s.index, "minute"].to_numpy()
        v = s.to_numpy()

        out[f"vts_{col}_min"] = float(v.min())
        out[f"vts_{col}_max"] = float(v.max())
        out[f"vts_{col}_mean"] = float(v.mean())
        out[f"vts_{col}_last"] = float(v[-1])
        out[f"vts_{col}_range"] = float(v.max() - v.min())
        out[f"vts_{col}_cv"] = float(v.std() / max(abs(v.mean()), 1e-9))
        out[f"vts_{col}_slope"] = _slope(m, v)
        out[f"vts_{col}_slope_first60"] = _slope(m[m <= 60], v[m <= 60])
        out[f"vts_{col}_slope_60_240"] = _slope(m[(m > 60) & (m <= 240)],
                                                v[(m > 60) & (m <= 240)])
        if len(v) >= 3:
            d2 = np.diff(np.diff(v))
            out[f"vts_{col}_accel_sign"] = float(np.sign(d2.mean()))
        # Peak / nadir timing
        out[f"vts_{col}_peak_value"] = float(v.max())
        out[f"vts_{col}_peak_minute"] = float(m[np.argmax(v)])
        out[f"vts_{col}_nadir_value"] = float(v.min())
        out[f"vts_{col}_nadir_minute"] = float(m[np.argmin(v)])

        # Critical-zone crossings
        lo, hi = CRITICAL[col]
        in_crit = (v < lo) | (v > hi)
        out[f"vts_{col}_n_critical"] = float(in_crit.sum())
        if in_crit.any():
            out[f"vts_{col}_time_to_first_critical"] = float(m[in_crit][0])
        else:
            out[f"vts_{col}_time_to_first_critical"] = -1.0  # sentinel

        # Last-30-min tail mean
        tail_mask = m >= max(m.max() - 30, 0)
        out[f"vts_{col}_last30_mean"] = float(v[tail_mask].mean())

        # Recovery half-time (from peak toward baseline=first)
        if len(v) >= 3:
            baseline = v[0]
            peak_idx = int(np.argmax(np.abs(v - baseline)))
            peak_minute = m[peak_idx]
            peak_value = v[peak_idx]
            target = (baseline + peak_value) / 2.0
            after = v[peak_idx:]
            after_min = m[peak_idx:]
            if peak_value > baseline:
                hit = np.where(after <= target)[0]
            else:
                hit = np.where(after >= target)[0]
            if hit.size:
                out[f"vts_{col}_recovery_halftime"] = \
                    float(after_min[hit[0]] - peak_minute)
            else:
                out[f"vts_{col}_recovery_halftime"] = -1.0
    return out


# ---- Group C — lab trajectory ----------------------------------------

LAB_COLS = ["cbc_wbc", "bmp_sodium", "bmp_potassium", "bmp_bicarb",
            "lft_ast", "vbg_ph", "troponin", "lactate", "cpk",
            "esr", "crp", "poct_glucose", "serum_osm"]


def lab_trajectory(ts_rows: list[dict]) -> dict[str, float]:
    """Group C: per-analyte trajectory + informative-missingness."""
    out: dict[str, float] = {
        "lts_n_records": float(len(ts_rows)),
        "lts_any_drawn": float(len(ts_rows) > 0),
    }
    if not ts_rows:
        # Informative absence — flag each analyte as not-drawn
        for col in LAB_COLS:
            out[f"lts_{col}_was_drawn"] = 0.0
        return out

    df = pd.DataFrame(ts_rows)
    if "minute" not in df.columns:
        return out
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    df = df.dropna(subset=["minute"]).sort_values("minute")

    for col in LAB_COLS:
        if col not in df.columns:
            out[f"lts_{col}_was_drawn"] = 0.0
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            out[f"lts_{col}_was_drawn"] = 0.0
            continue
        m = df.loc[s.index, "minute"].to_numpy()
        v = s.to_numpy()

        out[f"lts_{col}_was_drawn"] = 1.0
        out[f"lts_{col}_first_minute"] = float(m[0])
        out[f"lts_{col}_last_minute"] = float(m[-1])
        out[f"lts_{col}_n_draws"] = float(len(v))
        out[f"lts_{col}_first_value"] = float(v[0])
        out[f"lts_{col}_last_value"] = float(v[-1])
        out[f"lts_{col}_max_value"] = float(v.max())
        out[f"lts_{col}_max_minute"] = float(m[np.argmax(v)])
        out[f"lts_{col}_delta"] = float(v[-1] - v[0])
        out[f"lts_{col}_pct_change"] = float(
            (v[-1] - v[0]) / max(abs(v[0]), 1e-9))

    # Panel completeness
    drawn_keys = [f"lts_{c}_was_drawn" for c in LAB_COLS]
    out["lts_panel_completeness"] = float(
        sum(out[k] for k in drawn_keys) / len(LAB_COLS))
    return out


# ---- Group D — intervention sequencing ------------------------------

INTERVENTION_KW = {
    "benzo":       ("benzo", "lorazepam", "midazolam", "diazepam"),
    "fluid":       ("fluid", "saline", "crystalloid", "ivf"),
    "intubation":  ("intubat", "ett ", "rsi"),
    "reversal":    ("naloxone", "flumazenil", "reversal", "narcan"),
    "antipyretic": ("antipyretic", "acetaminophen", "ibuprofen"),
    "vasopressor": ("vasopressor", "norepinephrine", "epinephrine",
                    "phenylephrine", "vasopressin"),
    "nippv":       ("nippv", "bipap", "cpap", "positive pressure"),
    "cooling":     ("cool", "ice", "evaporative"),
    "monitor":     ("monitor", "telemetry", "serial"),
    "physostig":   ("physostigmine",),
}

# Severity ladder for sequential_escalation_score
LADDER = ["monitor", "fluid", "antipyretic", "benzo",
          "reversal", "nippv", "intubation", "vasopressor"]


def _matches(name: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in name for kw in keywords)


def intervention_features(ts_rows: list[dict]) -> dict[str, float]:
    """Group D: intervention timing and sequencing."""
    out: dict[str, float] = {
        "itv_n_total": float(len(ts_rows)),
        "itv_any": float(len(ts_rows) > 0),
    }
    if not ts_rows:
        for kw in INTERVENTION_KW:
            out[f"itv_{kw}_count"] = 0.0
            out[f"itv_time_to_first_{kw}"] = -1.0
        out["itv_first_minute"] = -1.0
        out["itv_last_minute"] = -1.0
        out["itv_density_first_hour"] = 0.0
        out["itv_escalation_max_rung"] = 0.0
        out["itv_intubation_after_benzo"] = 0.0
        return out

    df = pd.DataFrame(ts_rows)
    if "minute" not in df.columns:
        df["minute"] = np.nan
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    df = df.dropna(subset=["minute"]).sort_values("minute")
    if df.empty:
        return out

    names = df.get("event_name", pd.Series([""] * len(df))) \
              .astype(str).str.lower()
    minutes = df["minute"].to_numpy()

    out["itv_first_minute"] = float(minutes.min())
    out["itv_last_minute"] = float(minutes.max())
    out["itv_density_first_hour"] = float((minutes <= 60).sum())

    first_minute_kw: dict[str, float] = {}
    for kw, kws in INTERVENTION_KW.items():
        mask = names.apply(lambda n: _matches(n, kws))
        out[f"itv_{kw}_count"] = float(mask.sum())
        if mask.any():
            first_minute_kw[kw] = float(minutes[mask][0])
            out[f"itv_time_to_first_{kw}"] = first_minute_kw[kw]
        else:
            out[f"itv_time_to_first_{kw}"] = -1.0

    # Sequential escalation ladder (max rung reached, 1-indexed)
    max_rung = 0
    for rung_idx, kw in enumerate(LADDER, start=1):
        if first_minute_kw.get(kw, -1.0) >= 0.0:
            max_rung = max(max_rung, rung_idx)
    out["itv_escalation_max_rung"] = float(max_rung)

    # Intubation after failed benzo
    benzo_min = first_minute_kw.get("benzo", -1.0)
    intub_min = first_minute_kw.get("intubation", -1.0)
    out["itv_intubation_after_benzo"] = float(
        benzo_min >= 0 and intub_min >= 0 and intub_min > benzo_min)
    return out


# ---- Group E — cross-modal timing -----------------------------------

def cross_modal_features(vital_rows: list[dict],
                         lab_rows: list[dict],
                         intervention_rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}

    # First-lab to first-intervention latency
    if lab_rows and intervention_rows:
        first_lab_min = min(int(r["minute"]) for r in lab_rows
                            if "minute" in r)
        first_itv_min = min(int(r["minute"]) for r in intervention_rows
                            if "minute" in r)
        out["xmod_first_lab_to_first_itv_min"] = float(
            first_itv_min - first_lab_min)
    else:
        out["xmod_first_lab_to_first_itv_min"] = np.nan

    # First HR-critical to first benzo
    if vital_rows and intervention_rows:
        try:
            vdf = pd.DataFrame(vital_rows)
            vdf["minute"] = pd.to_numeric(vdf.get("minute"), errors="coerce")
            vdf["heart_rate"] = pd.to_numeric(vdf.get("heart_rate"),
                                              errors="coerce")
            vdf = vdf.dropna(subset=["minute", "heart_rate"])
            crit = vdf[(vdf["heart_rate"] < 40) | (vdf["heart_rate"] > 130)]
            if not crit.empty:
                first_hr_crit = float(crit["minute"].min())
                itv_df = pd.DataFrame(intervention_rows)
                itv_df["minute"] = pd.to_numeric(itv_df.get("minute"),
                                                 errors="coerce")
                names = itv_df.get("event_name", pd.Series(""))\
                    .astype(str).str.lower()
                benzo_mask = names.apply(
                    lambda n: _matches(n, INTERVENTION_KW["benzo"]))
                if benzo_mask.any():
                    out["xmod_hr_crit_to_benzo_min"] = float(
                        itv_df.loc[benzo_mask, "minute"].min()
                        - first_hr_crit)
                else:
                    out["xmod_hr_crit_to_benzo_min"] = np.nan
            else:
                out["xmod_hr_crit_to_benzo_min"] = np.nan
        except (KeyError, ValueError):
            out["xmod_hr_crit_to_benzo_min"] = np.nan
    else:
        out["xmod_hr_crit_to_benzo_min"] = np.nan
    return out


# ---- Group F — stability / volatility -------------------------------

def stability_features(vital_rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not vital_rows:
        out["stab_total_critical_breaches"] = 0.0
        out["stab_max_consecutive_critical_min"] = 0.0
        out["stab_hr_oscillations"] = 0.0
        return out

    df = pd.DataFrame(vital_rows)
    df["minute"] = pd.to_numeric(df.get("minute"), errors="coerce")
    df = df.dropna(subset=["minute"]).sort_values("minute")

    total_breaches = 0
    max_consec_minutes = 0
    for col, (lo, hi) in CRITICAL.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        breach = (s < lo) | (s > hi)
        total_breaches += int(breach.sum())
        # max consecutive critical minutes for this vital
        if breach.any():
            minutes = df["minute"].to_numpy()
            in_run = False
            run_start = 0
            for i, b in enumerate(breach.fillna(False).to_numpy()):
                if b and not in_run:
                    in_run = True
                    run_start = minutes[i]
                elif not b and in_run:
                    in_run = False
                    max_consec_minutes = max(max_consec_minutes,
                                             int(minutes[i] - run_start))
            if in_run:
                max_consec_minutes = max(max_consec_minutes,
                                         int(minutes[-1] - run_start))

    out["stab_total_critical_breaches"] = float(total_breaches)
    out["stab_max_consecutive_critical_min"] = float(max_consec_minutes)

    # HR oscillation count
    if "heart_rate" in df.columns:
        hr = pd.to_numeric(df["heart_rate"], errors="coerce").dropna()
        if len(hr) >= 3:
            d = np.diff(hr.to_numpy())
            sign_changes = int(np.sum(np.diff(np.sign(d)) != 0))
            out["stab_hr_oscillations"] = float(sign_changes)
        else:
            out["stab_hr_oscillations"] = 0.0
    else:
        out["stab_hr_oscillations"] = 0.0

    return out


# ---- Group G — recovery / deterioration arc -------------------------

def arc_features(vital_rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not vital_rows:
        out["arc_trajectory_class"] = 0.0  # unknown
        out["arc_time_to_min_gcs_min"] = -1.0
        out["arc_time_to_gcs_recovery_min"] = -1.0
        out["arc_steady_state_last60"] = 0.0
        return out

    df = pd.DataFrame(vital_rows)
    df["minute"] = pd.to_numeric(df.get("minute"), errors="coerce")
    df = df.dropna(subset=["minute"]).sort_values("minute")

    # GCS arc
    if "gcs" in df.columns:
        gcs = pd.to_numeric(df["gcs"], errors="coerce")
        valid = gcs.notna()
        if valid.any():
            gcs_v = gcs[valid].to_numpy()
            gcs_m = df.loc[valid, "minute"].to_numpy()
            out["arc_time_to_min_gcs_min"] = float(gcs_m[np.argmin(gcs_v)])
            # Recovery: first minute in last 60 where GCS >= 13
            tail = gcs_m >= max(gcs_m.max() - 60, 0)
            recover_mask = tail & (gcs_v >= 13)
            if recover_mask.any():
                out["arc_time_to_gcs_recovery_min"] = float(
                    gcs_m[recover_mask][0])
            else:
                out["arc_time_to_gcs_recovery_min"] = -1.0
        else:
            out["arc_time_to_min_gcs_min"] = -1.0
            out["arc_time_to_gcs_recovery_min"] = -1.0
    else:
        out["arc_time_to_min_gcs_min"] = -1.0
        out["arc_time_to_gcs_recovery_min"] = -1.0

    # Steady-state: low CV across HR, RR, GCS in last 60 min
    tail = df["minute"] >= max(df["minute"].max() - 60, 0)
    steady_count = 0
    for col, threshold in [("heart_rate", 0.15), ("respiratory_rate", 0.20),
                            ("gcs", 0.10)]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df.loc[tail, col], errors="coerce").dropna()
        if len(s) >= 2 and (s.std() / max(abs(s.mean()), 1e-9)) < threshold:
            steady_count += 1
    out["arc_steady_state_last60"] = float(steady_count >= 2)

    # Trajectory class: -1 worsening, 0 stable, +1 improving (HR or GCS)
    if "heart_rate" in df.columns and "gcs" in df.columns:
        hr_slope = _slope(df["minute"].to_numpy(),
                          pd.to_numeric(df["heart_rate"], errors="coerce")
                          .ffill().bfill()
                          .to_numpy())
        gcs_slope = _slope(df["minute"].to_numpy(),
                           pd.to_numeric(df["gcs"], errors="coerce")
                           .ffill().bfill()
                           .to_numpy())
        # Improving: HR slope <0 and GCS slope >=0; Worsening: opposite
        if hr_slope < -0.1 and gcs_slope >= 0:
            out["arc_trajectory_class"] = 1.0  # improving
        elif hr_slope > 0.1 and gcs_slope < 0:
            out["arc_trajectory_class"] = -1.0  # worsening
        else:
            out["arc_trajectory_class"] = 0.0  # stable / mixed
    else:
        out["arc_trajectory_class"] = 0.0
    return out


# ---- Orchestration --------------------------------------------------

def main() -> None:
    print("Reading xlsx...")
    triage = pd.read_excel(XLSX, sheet_name="Triage_Data", engine="openpyxl")
    fourh = pd.read_excel(XLSX, sheet_name="Four_Hour_Data", engine="openpyxl")

    # Same-day arrival volume (used in Group A)
    sd_volume = (triage.groupby("encounter_arrival_date")["encounter_id"]
                       .transform("count")).rename("arrival_same_day_volume_raw")
    triage = pd.concat([triage, sd_volume], axis=1)

    # ---- TRIAGE side: Group A only ----
    rows_t: list[dict] = []
    for _, row in triage.iterrows():
        feats = arrival_features(row["encounter_arrival_date"],
                                  int(row["arrival_same_day_volume_raw"]))
        feats["encounter_id"] = row["encounter_id"]
        rows_t.append(feats)
    time_t = pd.DataFrame(rows_t)
    triage_path = DERIVED / "time_features_triage.csv"
    time_t.to_csv(triage_path, index=False)
    print(f"Wrote triage time-features: {time_t.shape} -> {triage_path}")

    # ---- 4-HOUR side: Groups A-G ----
    print("Computing 4-hour time features (Groups A-G) — 261 encounters...")
    rows_f: list[dict] = []
    for _, row in fourh.merge(triage[["encounter_id",
                                       "encounter_arrival_date",
                                       "arrival_same_day_volume_raw"]],
                              on="encounter_id").iterrows():
        vitals = safe_parse(row["ed_course.vitals_timeseries"])
        labs = safe_parse(row["ed_course.labs_timeseries"])
        intvs = safe_parse(row["ed_course.interventions"])

        feats: dict[str, float | str] = {}
        feats.update(arrival_features(row["encounter_arrival_date"],
                                       int(row["arrival_same_day_volume_raw"])))
        feats.update(vital_trajectory(vitals))
        feats.update(lab_trajectory(labs))
        feats.update(intervention_features(intvs))
        feats.update(cross_modal_features(vitals, labs, intvs))
        feats.update(stability_features(vitals))
        feats.update(arc_features(vitals))
        feats["encounter_id"] = row["encounter_id"]
        rows_f.append(feats)

    time_f = pd.DataFrame(rows_f)
    # Move encounter_id to first column
    cols = ["encounter_id"] + [c for c in time_f.columns if c != "encounter_id"]
    time_f = time_f[cols]
    fourh_path = DERIVED / "time_features_fourh.csv"
    time_f.to_csv(fourh_path, index=False)
    print(f"Wrote 4-hour time-features: {time_f.shape} -> {fourh_path}")

    # ---- Merge into existing feature tables ----
    features_triage = pd.read_csv(DERIVED / "features_triage.csv")
    features_fourh = pd.read_csv(DERIVED / "features_fourh.csv")

    # Drop any prior Group-A columns to avoid duplicate-merge collisions
    drop_priorA = [c for c in features_triage.columns
                   if c.startswith("arrival_")]
    if drop_priorA:
        features_triage = features_triage.drop(columns=drop_priorA)
    drop_priorA_f = [c for c in features_fourh.columns
                     if c.startswith("arrival_")]
    if drop_priorA_f:
        features_fourh = features_fourh.drop(columns=drop_priorA_f)

    # Fix _x/_y duplicate bug: extract_structured.py also writes vts_/lts_/itv_
    # columns from a simpler aggregator; this script is the canonical source.
    # Drop any pre-existing time-aggregated columns from the 4h table so the
    # merge doesn't collide with the richer Group B-G outputs.
    canonical_prefixes = ("vts_", "lts_", "itv_", "xmod_", "stab_", "arc_")
    drop_collisions = [c for c in features_fourh.columns
                        if c.startswith(canonical_prefixes)]
    if drop_collisions:
        print(f"Dropping {len(drop_collisions)} pre-existing "
              f"vts_/lts_/itv_/xmod_/stab_/arc_ columns "
              f"(extract_time_features is canonical source)")
        features_fourh = features_fourh.drop(columns=drop_collisions)

    features_triage = features_triage.merge(time_t, on="encounter_id",
                                             how="left")
    features_fourh = features_fourh.merge(time_f, on="encounter_id",
                                           how="left")

    features_triage.to_csv(DERIVED / "features_triage.csv", index=False)
    features_fourh.to_csv(DERIVED / "features_fourh.csv", index=False)

    print(f"Merged feature tables:")
    print(f"  features_triage.csv: {features_triage.shape}")
    print(f"  features_fourh.csv:  {features_fourh.shape}")

    # ---- Leakage sentinel ----
    forbidden_prefixes = ("vts_", "lts_", "itv_", "xmod_",
                          "stab_", "arc_")
    leaked = [c for c in features_triage.columns
              if c.startswith(forbidden_prefixes) or "_4h" in c
              or "delta_" in c]
    assert not leaked, (f"Leakage in triage features: {leaked}")
    print("\nOK: no leakage; triage features contain only arrival-time + "
          "minute-0 signals.")


if __name__ == "__main__":
    main()
