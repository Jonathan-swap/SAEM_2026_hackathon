"""Phase-2 EDA + integrity tests.

Reads the Phase-2 feature tables (derived/phase2/features_triage.csv +
features_fourh.csv produced by predict_phase2.py against
data2/Hackathon_Data_Release_2_SHARE.xlsx) and writes:

  derived/phase2/feature_summary_stats.csv   per-feature summary
  derived/phase2/missingness.csv             per-column NA counts/fractions
  derived/phase2/integrity_results.csv       pass/fail per test
  derived/phase2/phase1_vs_phase2_shift.csv  mean shift for shared numeric
                                              features
  derived/phase2/eda_plots/*.png             distribution plots
  derived/phase2/eda_report.md               consolidated markdown

Integrity tests check:
  - encounter_id is unique
  - features_triage and features_fourh cover the same cohort
  - vital signs lie in plausible physiologic ranges
  - encounter_arrival_date parses + is within 2024-01-01..2027-01-01
  - no Task-1 / Task-2 target columns leaked into features
  - no all-NaN columns
  - no duplicate column names
  - one-hot dummy columns sum to 0 or 1 row-wise (no >1)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "derived"
PHASE2 = DERIVED / "phase2"
PLOTS = PHASE2 / "eda_plots"

PHASE1_TRIAGE = DERIVED / "features_triage.csv"
PHASE1_FOURH = DERIVED / "features_fourh.csv"
PHASE2_TRIAGE = PHASE2 / "features_triage.csv"
PHASE2_FOURH = PHASE2 / "features_fourh.csv"

# Physiologic ranges (loose — flag values outside as integrity-suspect).
VITAL_BOUNDS = {
    "triage_heart_rate":                 (20, 250),
    "triage_respiratory_rate":           (4, 60),
    "triage_snapshot.systolic_bp":       (50, 260),
    "triage_snapshot.diastolic_bp":      (20, 160),
    "triage_snapshot.oxygen_saturation": (40, 100),
    "triage_temperature_c":              (32.0, 42.5),
    "triage_gcs":                        (3, 15),
    "triage_age":                        (0, 120),
    "triage_lab_glucose":                (20, 1200),
    "triage_lab_ph":                     (6.7, 7.8),
    "triage_lab_sodium":                 (110, 175),
    "triage_lab_potassium":              (1.5, 8.0),
    "triage_lab_anion_gap":              (0, 50),
    "triage_lab_hemoglobin":             (4, 22),
}

FORBIDDEN_IN_FEATURES = (
    "encounter_disposition_label",
    "ground_truth_drug",
    "ground_truth_drug_name",
)


# ---------- Summary stats ----------------------------------------------

def summary_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per-column summary. Numeric cols get count/mean/median/std/min/
    max/q25/q75; bool gets fraction-true; object gets nunique + top."""
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.notna().sum()
        n_missing = s.isna().sum()
        rec = {
            "table": label,
            "column": col,
            "dtype": str(s.dtype),
            "n": int(non_null),
            "n_missing": int(n_missing),
            "pct_missing": float(n_missing / len(df)) if len(df) else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_bool_dtype(s):
            rec["mean"] = float(s.fillna(False).astype(float).mean())
        elif pd.api.types.is_numeric_dtype(s):
            sn = pd.to_numeric(s, errors="coerce").dropna()
            if len(sn):
                rec["mean"] = float(sn.mean())
                rec["std"] = float(sn.std())
                rec["min"] = float(sn.min())
                rec["q25"] = float(sn.quantile(0.25))
                rec["median"] = float(sn.median())
                rec["q75"] = float(sn.quantile(0.75))
                rec["max"] = float(sn.max())
        else:
            vc = s.dropna().astype(str).value_counts()
            rec["top_value"] = vc.index[0] if len(vc) else None
            rec["top_value_count"] = int(vc.iloc[0]) if len(vc) else 0
        rows.append(rec)
    return pd.DataFrame(rows)


def missingness_report(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        rows.append({
            "table": label,
            "column": col,
            "n_missing": n_miss,
            "pct_missing": float(n_miss / len(df)) if len(df) else 0.0,
        })
    return pd.DataFrame(rows).sort_values(
        ["pct_missing", "column"], ascending=[False, True],
    ).reset_index(drop=True)


# ---------- Integrity tests --------------------------------------------

def integrity_tests(triage: pd.DataFrame, fourh: pd.DataFrame) -> pd.DataFrame:
    results = []

    def check(name, ok, detail=""):
        results.append({"test": name, "result": "PASS" if ok else "FAIL",
                        "detail": detail})

    # 1. encounter_id uniqueness
    for name, df in (("triage", triage), ("fourh", fourh)):
        dup = df["encounter_id"].duplicated().sum()
        check(f"encounter_id_unique__{name}", dup == 0,
              f"{dup} duplicate(s)")

    # 2. cohort match
    t_ids = set(triage["encounter_id"])
    f_ids = set(fourh["encounter_id"])
    only_t = t_ids - f_ids
    only_f = f_ids - t_ids
    check("cohort_match_triage_vs_fourh", not only_t and not only_f,
          f"only_in_triage={len(only_t)}  only_in_fourh={len(only_f)}")

    # 3. arrival date sane + parseable
    try:
        dates = pd.to_datetime(triage["encounter_arrival_date"])
        in_range = ((dates >= "2024-01-01") & (dates < "2027-01-01")).all()
        check("arrival_date_in_2024_2027",
              bool(in_range),
              f"range = [{dates.min()}, {dates.max()}]")
    except Exception as e:
        check("arrival_date_in_2024_2027", False, f"parse error: {e}")

    # 4. vitals in physiologic range
    for col, (lo, hi) in VITAL_BOUNDS.items():
        if col not in triage.columns:
            continue
        s = pd.to_numeric(triage[col], errors="coerce")
        bad = ((s < lo) | (s > hi)).sum()
        check(f"vital_in_range__{col}", int(bad) == 0,
              f"{int(bad)} value(s) outside [{lo}, {hi}]")

    # 5. no leaked outcome columns
    for name, df in (("triage", triage), ("fourh", fourh)):
        leaked = [c for c in df.columns if c in FORBIDDEN_IN_FEATURES]
        check(f"no_outcome_columns__{name}", not leaked,
              f"leaked={leaked}")

    # 6. no all-NaN columns (informational; some may legitimately be
    # all-NaN if their derived signals don't fire on this cohort)
    for name, df in (("triage", triage), ("fourh", fourh)):
        all_nan = [c for c in df.columns if df[c].isna().all()]
        check(f"no_all_nan_columns__{name}", not all_nan,
              f"all-NaN count = {len(all_nan)}: "
              f"{all_nan[:3]}{'...' if len(all_nan) > 3 else ''}")

    # 7. no duplicate column names
    for name, df in (("triage", triage), ("fourh", fourh)):
        dups = df.columns[df.columns.duplicated()].tolist()
        check(f"no_duplicate_columns__{name}", not dups, f"dups={dups}")

    # 8. one-hot dummy columns (festival_*, mode_of_arrival_*, etc.)
    # should sum to 0 or 1 row-wise. Check the obvious one-hot groups.
    one_hot_prefixes = [
        ("triage_mode_of_arrival_", triage),
    ]
    for prefix, df in one_hot_prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        sums = df[cols].sum(axis=1)
        bad = (sums > 1).sum()
        check(f"one_hot_consistent__{prefix.rstrip('_')}",
              int(bad) == 0,
              f"{int(bad)} rows with multiple hot values")

    return pd.DataFrame(results)


# ---------- Phase-1 vs Phase-2 distribution shift ----------------------

def distribution_shift(p1: pd.DataFrame, p2: pd.DataFrame) -> pd.DataFrame:
    """For shared numeric columns, compare Phase-1 vs Phase-2 means,
    medians, and Cohen's-d effect size."""
    shared = sorted(set(p1.columns) & set(p2.columns))
    rows = []
    for col in shared:
        s1 = pd.to_numeric(p1[col], errors="coerce").dropna()
        s2 = pd.to_numeric(p2[col], errors="coerce").dropna()
        if len(s1) < 5 or len(s2) < 5:
            continue
        m1, m2 = s1.mean(), s2.mean()
        sd1, sd2 = s1.std(ddof=1), s2.std(ddof=1)
        pooled = float(np.sqrt(((len(s1) - 1) * sd1**2
                                + (len(s2) - 1) * sd2**2)
                                / max(len(s1) + len(s2) - 2, 1)))
        d = (m2 - m1) / pooled if pooled > 0 else float("nan")
        rows.append({
            "column": col,
            "phase1_n": int(len(s1)),
            "phase2_n": int(len(s2)),
            "phase1_mean": float(m1),
            "phase2_mean": float(m2),
            "delta_mean": float(m2 - m1),
            "phase1_median": float(s1.median()),
            "phase2_median": float(s2.median()),
            "cohen_d": float(d),
        })
    return pd.DataFrame(rows).sort_values(
        "cohen_d", key=lambda s: s.abs(), ascending=False,
    ).reset_index(drop=True)


# ---------- Distribution plots ----------------------------------------

KEY_VITALS = [
    "triage_heart_rate",
    "triage_respiratory_rate",
    "triage_snapshot.systolic_bp",
    "triage_snapshot.diastolic_bp",
    "triage_snapshot.oxygen_saturation",
    "triage_temperature_c",
    "triage_gcs",
    "triage_age",
]
KEY_LABS = [
    "triage_lab_glucose",
    "triage_lab_ph",
    "triage_lab_sodium",
    "triage_lab_potassium",
    "triage_lab_anion_gap",
    "triage_lab_hemoglobin",
]
PMH_FLAGS = [
    "triage_mh_psych",
    "triage_mh_cardiac",
    "triage_mh_pulm",
    "triage_mh_renal",
    "triage_mh_substance_use",
]
DEMOGRAPHICS_NUMERIC = [
    "triage_age",
    "triage_esi",
    "triage_pain_scale",
]
FOURH_REASSESS_VITALS = [
    "heart_rate_4h",
    "respiratory_rate_4h",
    "systolic_bp_4h",
    "diastolic_bp_4h",
    "oxygen_saturation_4h",
    "temperature_c_4h",
    "gcs_4h",
]
FOURH_DELTAS = [
    "delta_hr",
    "delta_temp",
    "delta_gcs",
]
PEAK_LABS = [
    "peak_lactate",
    "peak_cpk",
    "peak_vbg_ph",
    "peak_troponin",
    "peak_hr",
    "peak_temp",
]
INTERVENTIONS = [
    "ivf_count_0_4h",
    "benzodiazepine_count_0_4h",
    "reversal_count_0_4h",
    "intubated_0_4h",
    "positive_pressure_0_4h",
    "cvc_0_4h",
]
CAND_COMPOSITES = [
    "cand_news_total",
    "cand_shock_index",
    "cand_sympathetic_score",
    "cand_cns_depression_score",
    "cand_pmh_count",
    "cand_poc_abn_count",
]


def density_panel(ax, s1: pd.Series, s2: pd.Series, title: str,
                   xlabel: str = "") -> None:
    """Density-normalised comparison. Auto-switches to side-by-side
    proportion bars for low-cardinality discrete data (so both phases
    are visually comparable regardless of cohort size)."""
    s1 = pd.to_numeric(s1, errors="coerce").dropna()
    s2 = pd.to_numeric(s2, errors="coerce").dropna()
    if s1.empty and s2.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        return
    union_unique = (set(s1.unique()) if not s1.empty else set()) | \
                   (set(s2.unique()) if not s2.empty else set())
    if len(union_unique) <= 12:
        vals = sorted(union_unique)
        x = np.arange(len(vals))
        w = 0.4
        p1 = [float((s1 == v).mean()) if not s1.empty else 0 for v in vals]
        p2 = [float((s2 == v).mean()) if not s2.empty else 0 for v in vals]
        ax.bar(x - w / 2, p1, w, color="#1f77b4",
               label=f"P1 (n={len(s1)})", alpha=0.85)
        ax.bar(x + w / 2, p2, w, color="#d62728",
               label=f"P2 (n={len(s2)})", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in vals], fontsize=7)
        ax.set_ylabel("proportion", fontsize=7)
    else:
        lo = float(min(s1.min() if not s1.empty else np.inf,
                       s2.min() if not s2.empty else np.inf))
        hi = float(max(s1.max() if not s1.empty else -np.inf,
                       s2.max() if not s2.empty else -np.inf))
        bins = np.linspace(lo, hi, 25)
        if not s1.empty:
            ax.hist(s1, bins=bins, density=True, alpha=0.45,
                    color="#1f77b4", label=f"P1 (n={len(s1)})")
        if not s2.empty:
            ax.hist(s2, bins=bins, density=True, alpha=0.45,
                    color="#d62728", label=f"P2 (n={len(s2)})")
        ax.set_ylabel("density", fontsize=7)
    ax.set_title(title, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6)


def make_distribution_plot(p1: pd.DataFrame, p2: pd.DataFrame,
                            cols: list[str], title: str,
                            out_path: Path,
                            label_strip: str = "triage_") -> None:
    """Density-normalised multi-panel comparison. `cols` are matched
    against both DataFrames; missing-in-one columns are skipped."""
    cols = [c for c in cols if c in p1.columns and c in p2.columns]
    if not cols:
        return
    ncol = 4
    nrow = (len(cols) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 2.8))
    axes = np.atleast_2d(axes).ravel()
    for i, col in enumerate(cols):
        label = col.replace(label_strip, "").replace(".", " ")
        density_panel(axes[i], p1[col], p2[col], label, col)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_categorical_plot(p1: pd.DataFrame, p2: pd.DataFrame,
                            cols: list[str], title: str,
                            out_path: Path) -> None:
    """Side-by-side proportion bar charts for categorical/string cols."""
    cols = [c for c in cols if c in p1.columns and c in p2.columns]
    if not cols:
        return
    ncol = 2
    nrow = (len(cols) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.5, nrow * 3.2))
    axes = np.atleast_2d(axes).ravel()
    for i, col in enumerate(cols):
        ax = axes[i]
        v1 = p1[col].astype(str).fillna("nan").value_counts(normalize=True)
        v2 = p2[col].astype(str).fillna("nan").value_counts(normalize=True)
        vals = sorted(set(v1.index) | set(v2.index),
                       key=lambda x: -max(v1.get(x, 0), v2.get(x, 0)))
        # cap to top-8 values to keep readable
        if len(vals) > 8:
            vals = vals[:8]
        x = np.arange(len(vals))
        w = 0.4
        ax.bar(x - w / 2, [v1.get(v, 0) for v in vals], w,
               color="#1f77b4", label=f"P1 (n={len(p1)})", alpha=0.85)
        ax.bar(x + w / 2, [v2.get(v, 0) for v in vals], w,
               color="#d62728", label=f"P2 (n={len(p2)})", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([v[:22] for v in vals], rotation=30,
                              ha="right", fontsize=7)
        ax.set_ylabel("proportion", fontsize=7)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_missingness_plot(triage_stats: pd.DataFrame,
                            fourh_stats: pd.DataFrame,
                            out_path: Path,
                            top_n: int = 30) -> None:
    combined = pd.concat([
        triage_stats.assign(table="triage"),
        fourh_stats.assign(table="fourh"),
    ], ignore_index=True)
    top = combined[combined["pct_missing"] > 0].sort_values(
        "pct_missing", ascending=False).head(top_n)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(top))))
    colors = top["table"].map({"triage": "#1f77b4", "fourh": "#2ca02c"})
    ax.barh(range(len(top)), top["pct_missing"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{r.column} [{r.table}]"
                          for r in top.itertuples()], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction missing")
    ax.set_title(f"Top {len(top)} feature columns by missingness "
                  f"(Phase-2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------- Orchestration ---------------------------------------------

def main() -> None:
    PHASE2.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase-2 EDA + integrity")
    print("=" * 72)
    p2_triage = pd.read_csv(PHASE2_TRIAGE)
    p2_fourh = pd.read_csv(PHASE2_FOURH)
    p1_triage = pd.read_csv(PHASE1_TRIAGE)
    p1_fourh = pd.read_csv(PHASE1_FOURH)
    print(f"  triage Phase-1: {p1_triage.shape}   Phase-2: "
          f"{p2_triage.shape}")
    print(f"  fourh  Phase-1: {p1_fourh.shape}    Phase-2: "
          f"{p2_fourh.shape}")

    # Summary stats
    stats_triage = summary_stats(p2_triage, "triage")
    stats_fourh = summary_stats(p2_fourh, "fourh")
    stats = pd.concat([stats_triage, stats_fourh], ignore_index=True)
    stats.to_csv(PHASE2 / "feature_summary_stats.csv", index=False)
    print(f"  wrote feature_summary_stats.csv "
          f"({len(stats)} feature-rows)")

    # Missingness
    miss_triage = missingness_report(p2_triage, "triage")
    miss_fourh = missingness_report(p2_fourh, "fourh")
    miss = pd.concat([miss_triage, miss_fourh], ignore_index=True)
    miss.to_csv(PHASE2 / "missingness.csv", index=False)
    nz_miss = miss[miss["pct_missing"] > 0]
    print(f"  wrote missingness.csv: {len(nz_miss)} cols with any "
          f"missing")

    # Integrity tests
    integ = integrity_tests(p2_triage, p2_fourh)
    integ.to_csv(PHASE2 / "integrity_results.csv", index=False)
    n_fail = int((integ["result"] == "FAIL").sum())
    print(f"  wrote integrity_results.csv: "
          f"{n_fail} fail / {len(integ) - n_fail} pass / {len(integ)} total")

    # Phase-1 vs Phase-2 shift
    shift_triage = distribution_shift(p1_triage, p2_triage)
    shift_fourh = distribution_shift(p1_fourh, p2_fourh)
    shift_triage["table"] = "triage"
    shift_fourh["table"] = "fourh"
    shift = pd.concat([shift_triage, shift_fourh], ignore_index=True)
    shift.to_csv(PHASE2 / "phase1_vs_phase2_shift.csv", index=False)
    big_shift = shift[shift["cohen_d"].abs() > 0.5]
    print(f"  wrote phase1_vs_phase2_shift.csv: "
          f"{len(big_shift)} features with |Cohen d| > 0.5")

    # Plots — density-normalised throughout. Each multi-panel image
    # groups conceptually-related features so the user can scan one
    # axis at a time.
    plot_groups = [
        ("vitals_distribution.png",
         "Triage vitals — Phase 1 vs Phase 2 (density)",
         KEY_VITALS, p1_triage, p2_triage, "triage_"),
        ("labs_distribution.png",
         "Triage POC labs — Phase 1 vs Phase 2 (density)",
         KEY_LABS, p1_triage, p2_triage, "triage_lab_"),
        ("pmh_flags.png",
         "PMH flags — Phase 1 vs Phase 2",
         PMH_FLAGS, p1_triage, p2_triage, "triage_mh_"),
        ("demographics.png",
         "Demographics + acuity — Phase 1 vs Phase 2",
         DEMOGRAPHICS_NUMERIC, p1_triage, p2_triage, "triage_"),
        ("fourh_reassessment_vitals.png",
         "4-hour reassessment vitals — Phase 1 vs Phase 2 (density)",
         FOURH_REASSESS_VITALS, p1_fourh, p2_fourh, ""),
        ("fourh_deltas.png",
         "Triage → 4h vital deltas — Phase 1 vs Phase 2 (density)",
         FOURH_DELTAS, p1_fourh, p2_fourh, ""),
        ("peak_labs.png",
         "Peak labs (0–4h) — Phase 1 vs Phase 2 (density)",
         PEAK_LABS, p1_fourh, p2_fourh, "peak_"),
        ("interventions.png",
         "ED interventions (0–4h counts/flags) — Phase 1 vs Phase 2",
         INTERVENTIONS, p1_fourh, p2_fourh, ""),
        ("cand_composites.png",
         "Composite candidate features — Phase 1 vs Phase 2 (density)",
         CAND_COMPOSITES, p1_triage, p2_triage, "cand_"),
    ]
    for name, title, cols, p1, p2, strip in plot_groups:
        make_distribution_plot(p1, p2, cols, title,
                                PLOTS / name, label_strip=strip)
        print(f"  -> {name}")

    # Categorical / string columns (mode_of_arrival, sex, race/ethn.)
    make_categorical_plot(
        p1_triage, p2_triage,
        ["triage_mode_of_arrival",
         "triage_sex_gender",
         "triage_race_ethnicity",
         "triage_chief_complaint"],
        "Categorical triage fields — Phase 1 vs Phase 2",
        PLOTS / "categorical_triage.png",
    )
    print(f"  -> categorical_triage.png")

    make_missingness_plot(miss_triage, miss_fourh,
                          PLOTS / "missingness_top30.png")
    print(f"  wrote plots to {PLOTS.relative_to(ROOT)}")

    # Markdown report
    md = [
        "# Phase-2 EDA + Integrity Report",
        "",
        f"Source: `data2/Hackathon_Data_Release_2_SHARE.xlsx`",
        "",
        "## Cohort",
        "",
        f"| Table | Phase-1 rows × cols | Phase-2 rows × cols |",
        f"|---|---:|---:|",
        f"| features_triage | {p1_triage.shape} | {p2_triage.shape} |",
        f"| features_fourh  | {p1_fourh.shape}  | {p2_fourh.shape} |",
        "",
        f"Phase-2 arrival-date range: "
        f"`{p2_triage['encounter_arrival_date'].min()}` → "
        f"`{p2_triage['encounter_arrival_date'].max()}`",
        f"(Phase-1: `{p1_triage['encounter_arrival_date'].min()}` → "
        f"`{p1_triage['encounter_arrival_date'].max()}`)",
        "",
        "## Integrity test results",
        "",
        f"**{n_fail} FAIL / {len(integ) - n_fail} PASS** "
        f"out of {len(integ)} tests.",
        "",
        "| Test | Result | Detail |",
        "|---|---|---|",
    ]
    for _, r in integ.iterrows():
        md.append(f"| `{r['test']}` | {r['result']} | {r['detail']} |")

    md += [
        "",
        "## Top-30 most-missing features",
        "",
        "| Table | Column | n_missing | pct_missing |",
        "|---|---|---:|---:|",
    ]
    for _, r in miss[miss["pct_missing"] > 0].head(30).iterrows():
        md.append(
            f"| {r['table']} | `{r['column']}` | {r['n_missing']} | "
            f"{r['pct_missing']:.2%} |"
        )
    if (miss["pct_missing"] > 0).sum() == 0:
        md.append("| — | (no missing values) | 0 | 0.00% |")
    md.append("")
    md.append(f"![missingness](eda_plots/missingness_top30.png)")
    md.append("")

    md += [
        "## Density-normalised feature comparisons",
        "",
        "All histograms below use density on the y-axis (or "
        "proportions for discrete data) so the 261-vs-139 sample-size "
        "difference doesn't bias the visual comparison.",
        "",
        "### Triage vitals",
        "",
        "![vitals](eda_plots/vitals_distribution.png)",
        "",
        "### Triage POC labs",
        "",
        "![labs](eda_plots/labs_distribution.png)",
        "",
        "### Past medical history flags",
        "",
        "![pmh](eda_plots/pmh_flags.png)",
        "",
        "### Demographics + acuity (age, ESI, pain)",
        "",
        "![demographics](eda_plots/demographics.png)",
        "",
        "### Categorical triage fields (mode of arrival, sex, race, complaint)",
        "",
        "![categorical](eda_plots/categorical_triage.png)",
        "",
        "### 4-hour reassessment vitals",
        "",
        "![fourh-vitals](eda_plots/fourh_reassessment_vitals.png)",
        "",
        "### Triage → 4h vital deltas",
        "",
        "![deltas](eda_plots/fourh_deltas.png)",
        "",
        "### Peak labs (0–4h)",
        "",
        "![peak-labs](eda_plots/peak_labs.png)",
        "",
        "### ED interventions (0–4h counts/flags)",
        "",
        "![interventions](eda_plots/interventions.png)",
        "",
        "### Composite candidate features (NEWS, shock index, etc.)",
        "",
        "![composites](eda_plots/cand_composites.png)",
        "",
        "## Largest Phase-1 → Phase-2 distribution shifts",
        "",
        "Cohen's d ≥ 0.5 indicates a non-trivial shift. d > 0 → Phase-2 "
        "mean is higher than Phase-1.",
        "",
        "| Table | Feature | Phase-1 mean | Phase-2 mean | Δ | Cohen d |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in big_shift.head(30).iterrows():
        md.append(
            f"| {r['table']} | `{r['column']}` | "
            f"{r['phase1_mean']:.2f} | {r['phase2_mean']:.2f} | "
            f"{r['delta_mean']:+.2f} | {r['cohen_d']:+.2f} |"
        )
    if big_shift.empty:
        md.append("| — | (no features with |d| > 0.5) | — | — | — | — |")
    md.append("")

    # Build a Phase-1 vs Phase-2 stats side-by-side table for triage.
    # Numeric columns shared by both phases get P1 mean/median + P2
    # mean/median + delta-mean + Cohen's d.
    p1_stats = summary_stats(p1_triage, "triage").set_index("column")
    p2_stats = stats_triage.set_index("column")
    shift_lookup = shift_triage.set_index("column")

    md += [
        "## Numeric-feature summary statistics (triage) — Phase-1 vs Phase-2",
        "",
        "Side-by-side change per numeric triage feature. Cohen's d > 0 "
        "means Phase-2 mean is higher than Phase-1 mean. Full per-phase "
        "summary in `feature_summary_stats.csv`; full shift table in "
        "`phase1_vs_phase2_shift.csv`.",
        "",
        "| Column | P1 n | P2 n | P1 mean | P2 mean | Δ mean | P1 median | P2 median | Cohen d |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    numeric_t = p2_stats[
        p2_stats["dtype"].isin(["float64", "int64"])
    ]
    numeric_t = numeric_t[numeric_t["mean"].notna()]
    # Sort by absolute Cohen-d so the biggest changes come first.
    ordered_cols = []
    for col in numeric_t.index:
        if col in shift_lookup.index:
            ordered_cols.append((col, abs(shift_lookup.at[col, "cohen_d"])
                                 if pd.notna(shift_lookup.at[col, "cohen_d"])
                                 else 0.0))
        else:
            ordered_cols.append((col, 0.0))
    ordered_cols.sort(key=lambda kv: -kv[1])

    for col, _ in ordered_cols[:30]:
        if col not in p1_stats.index or col not in p2_stats.index:
            continue
        p1_row = p1_stats.loc[col]
        p2_row = p2_stats.loc[col]
        m1 = p1_row.get("mean", np.nan)
        m2 = p2_row.get("mean", np.nan)
        med1 = p1_row.get("median", np.nan)
        med2 = p2_row.get("median", np.nan)
        d = shift_lookup.at[col, "cohen_d"] \
            if col in shift_lookup.index else np.nan
        delta = (m2 - m1) if pd.notna(m1) and pd.notna(m2) else np.nan
        md.append(
            f"| `{col}` | {int(p1_row['n'])} | {int(p2_row['n'])} | "
            f"{m1:.3f} | {m2:.3f} | "
            f"{delta:+.3f} | "
            f"{med1:.2f} | {med2:.2f} | "
            f"{(f'{d:+.2f}' if pd.notna(d) else '—')} |"
        )

    md += [
        "",
        "## Files",
        "",
        "- `derived/phase2/feature_summary_stats.csv` — per-column stats",
        "- `derived/phase2/missingness.csv` — per-column NA counts",
        "- `derived/phase2/integrity_results.csv` — pass/fail per test",
        "- `derived/phase2/phase1_vs_phase2_shift.csv` — distribution shift",
        "- `derived/phase2/eda_plots/` — histograms + missingness bar chart",
        "",
    ]

    (PHASE2 / "eda_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote eda_report.md")

    print()
    print("Integrity test summary:")
    for _, r in integ.iterrows():
        marker = "OK" if r["result"] == "PASS" else "FAIL"
        print(f"  [{marker}] {r['test']:<48s} {r['detail']}")


if __name__ == "__main__":
    main()
