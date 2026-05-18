"""Descriptive EDA + new-feature exploration.

Companion to:
  - eda_initial.py   (raw xlsx + codebook inspection)
  - eda_advanced.py  (engineered-feature integrity + importance)

Sections:
  A. Dataset distributional profile  (numerics, categoricals, dates)
  B. Engineered-feature family overview + descriptives per family
  C. Outlier flagging (Tukey 1.5x IQR, aggregate only)
  D. Correlation structure (Pearson + Spearman on triage features)
  E. Candidate new-feature library (build, describe, quantify lift)
  F. Recommendation: which candidates to commit

Reproducibility:
  - Seed: 42
  - Aggregate-only output: no row-level data is logged or persisted
  - Synthetic dataset per organizers (no PHI), but workspace
    privacy conventions are preserved

Writes:
  derived/eda_descriptive_report.md
  derived/exploratory_features.csv  (committed candidates)
  derived/eda_plots/                 (PNGs)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
PLOTS = DERIVED / "eda_plots"
PLOTS.mkdir(parents=True, exist_ok=True)
REPORT = DERIVED / "eda_descriptive_report.md"
CANDIDATES_OUT = DERIVED / "exploratory_features.csv"

DRUG_CLASSES = ["Kraken Candy", "Triton Tabs", "Coral Dust", "None"]
DISPO_CLASSES = ["Discharge", "Floor", "ICU"]


# ---------- shared utils ----------------------------------------------

def fmt(x: float, k: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.{k}f}"


def mi_against(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Mutual information of every column in X against y. Imputes median."""
    keep = [c for c in X.columns if X[c].notna().any()]
    Xs = X[keep].copy()
    Xs = pd.get_dummies(Xs, dummy_na=True)
    Xs = SimpleImputer(strategy="median").fit_transform(Xs.values)
    mi = mutual_info_classif(Xs, y, random_state=42)
    cols = [c for c in pd.get_dummies(X[keep], dummy_na=True).columns]
    return pd.Series(mi, index=cols).sort_values(ascending=False)


# ---------- A. Distributional profile --------------------------------

def section_a_profile(rep: list[str], triage: pd.DataFrame,
                       fourh: pd.DataFrame) -> None:
    rep.append("## A. Dataset distributional profile\n")

    # Headline counts
    rep.append(f"- Encounters: **{len(triage)}**")
    rep.append(f"- Date range: "
               f"`{triage['encounter_arrival_date'].min()}` to "
               f"`{triage['encounter_arrival_date'].max()}` "
               f"({triage['encounter_arrival_date'].nunique()} unique days)")
    rep.append(f"- Triage features: **{triage.shape[1]} cols**")
    rep.append(f"- 4h features:     **{fourh.shape[1]} cols**\n")

    # Numerics: skew, kurtosis, range
    num = triage.select_dtypes(include="number")
    skip = {"encounter_id"}
    cols = [c for c in num.columns if c not in skip]
    rows = []
    for c in cols[:25]:  # core vitals + lab block
        s = pd.to_numeric(num[c], errors="coerce").dropna()
        if len(s) < 10:
            continue
        rows.append({
            "feature": c,
            "n": len(s),
            "mean": s.mean(),
            "sd": s.std(),
            "median": s.median(),
            "p25": s.quantile(0.25),
            "p75": s.quantile(0.75),
            "min": s.min(),
            "max": s.max(),
            "skew": stats.skew(s),
            "kurt": stats.kurtosis(s),
        })
    tbl = pd.DataFrame(rows)
    rep.append("### Numeric vitals + labs (first 25 cols)\n")
    rep.append("| Feature | n | mean | sd | median | p25 | p75 | min | max | skew | kurt |")
    rep.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        rep.append("| " + r["feature"] + " | " + " | ".join([
            f"{r['n']}",
            fmt(r["mean"]), fmt(r["sd"]),
            fmt(r["median"]), fmt(r["p25"]), fmt(r["p75"]),
            fmt(r["min"]), fmt(r["max"]),
            fmt(r["skew"]), fmt(r["kurt"]),
        ]) + " |")
    rep.append("")

    # Categoricals: top-k counts
    rep.append("### Categorical fields (top values)\n")
    obj = triage.select_dtypes(include=["object", "string"])
    for c in obj.columns:
        if c in {"encounter_id", "triage_brief_note"}:
            continue
        vc = triage[c].value_counts().head(6)
        rep.append(f"\n**{c}** ({triage[c].nunique()} unique)")
        for v, n in vc.items():
            rep.append(f"  - `{v}`: {n}")
    rep.append("")


# ---------- B. Engineered-feature family overview ---------------------

def section_b_families(rep: list[str], triage: pd.DataFrame,
                        fourh: pd.DataFrame) -> None:
    rep.append("## B. Engineered-feature family overview\n")

    families = [
        ("triage vitals",        r"^triage_(?:heart_rate|respiratory|snapshot|temperature|gcs|supplemental)"),
        ("triage demographics",  r"^triage_(?:age|sex|race)"),
        ("triage POC labs",      r"^triage_lab_"),
        ("PMH flags",            r"^triage_mh_"),
        ("arrival/context",      r"^(arrival_|festival_|is_festival|triage_mode_)"),
        ("note features",        r"^note_"),
        ("4h reassessment",      r"^ed_course_reassessment_4h\."),
        ("vital time-series agg",r"^vts_"),
        ("lab time-series agg",  r"^lts_"),
        ("intervention seq",     r"^itv_"),
        ("cross-modal",          r"^xmod_"),
        ("stability",            r"^stab_"),
        ("recovery arc",         r"^arc_"),
        ("differentials",        r"^(diff_|abs_diff_|pct_change_|direction_|n_vitals|any_vital_crit|supp_o2_)"),
        ("imaging abn flags",    r"_abnormal$"),
    ]
    import re as _re
    rep.append("| Family | Triage cols | 4h cols | Description |")
    rep.append("|---|---:|---:|---|")
    desc = {
        "triage vitals":        "Vitals captured at triage (minute 0)",
        "triage demographics":  "Age, sex, race",
        "triage POC labs":      "iStat panel at triage (glucose, pH, Na, K, Hgb, anion gap)",
        "PMH flags":            "Past medical history (psych/cardiac/pulm/renal/substance)",
        "arrival/context":      "Festival exposure markers + day of festival",
        "note features":        "Onset minutes + festival location parsed from triage_brief_note",
        "4h reassessment":      "Vitals + labs + intervention flags at 4h mark",
        "vital time-series agg":"Slopes / peaks / recovery half-time from minute-level vitals",
        "lab time-series agg":  "Per-analyte trajectory (first/last/n_draws/delta)",
        "intervention seq":     "Time-to-first-X, escalation ladder, intubation-after-benzo",
        "cross-modal":          "Latency between labs and interventions, HR-crit to benzo",
        "stability":            "Critical-band breach count, oscillation count",
        "recovery arc":         "Trajectory class, time-to-min-GCS, steady-state flag",
        "differentials":        "Triage <-> 4h paired deltas, pct-change, direction signs",
        "imaging abn flags":    "EKG/CXR/CT abnormal binary flags (at 4h)",
    }
    for label, pat in families:
        n_t = sum(1 for c in triage.columns if _re.search(pat, c))
        n_f = sum(1 for c in fourh.columns if _re.search(pat, c))
        rep.append(f"| {label} | {n_t} | {n_f} | {desc[label]} |")
    rep.append("")


# ---------- C. Outliers ----------------------------------------------

def section_c_outliers(rep: list[str], triage: pd.DataFrame) -> None:
    rep.append("## C. Outlier flagging (Tukey 1.5 x IQR)\n")
    rep.append("Flags — does not delete. Outliers in synthetic data may "
               "be intentional severity cases.\n")
    rep.append("| Feature | n_low | n_high | low fence | high fence |")
    rep.append("|---|---:|---:|---:|---:|")
    candidates = [
        "triage_heart_rate", "triage_respiratory_rate",
        "triage_snapshot.systolic_bp", "triage_snapshot.diastolic_bp",
        "triage_snapshot.oxygen_saturation", "triage_temperature_c",
        "triage_gcs", "triage_age", "triage_lab_glucose",
        "triage_lab_ph", "triage_lab_anion_gap",
        "triage_lab_potassium",
    ]
    for col in candidates:
        if col not in triage.columns:
            continue
        s = pd.to_numeric(triage[col], errors="coerce").dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        n_lo = int((s < lo).sum())
        n_hi = int((s > hi).sum())
        rep.append(f"| {col} | {n_lo} | {n_hi} | "
                   f"{fmt(lo)} | {fmt(hi)} |")
    rep.append("")


# ---------- D. Correlation structure ---------------------------------

def section_d_correlation(rep: list[str], triage: pd.DataFrame) -> None:
    rep.append("## D. Correlation among triage numeric features\n")
    cols = [c for c in triage.columns
            if c.startswith(("triage_heart", "triage_resp",
                              "triage_snapshot", "triage_temperature",
                              "triage_gcs", "triage_age",
                              "triage_lab_", "triage_esi", "triage_pain"))]
    num = triage[cols].select_dtypes(include="number")
    if num.shape[1] < 2:
        rep.append("(insufficient numeric columns)\n")
        return
    pear = num.corr(method="pearson")
    spear = num.corr(method="spearman")

    # Top absolute Pearson off-diagonal pairs
    u = pear.where(np.triu(np.ones(pear.shape, dtype=bool), k=1))
    top = u.abs().stack().sort_values(ascending=False).head(15)
    rep.append("### Top-15 Pearson correlation pairs (triage features)\n")
    rep.append("| Feature A | Feature B | Pearson r | Spearman r |")
    rep.append("|---|---|---:|---:|")
    for (a, b), _ in top.items():
        rep.append(f"| {a} | {b} | "
                   f"{pear.loc[a, b]:.3f} | {spear.loc[a, b]:.3f} |")
    rep.append("")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pear, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pear.columns)))
    ax.set_yticks(range(len(pear.columns)))
    ax.set_xticklabels(pear.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(pear.columns, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Triage numerics — Pearson correlation")
    fig.tight_layout()
    path = PLOTS / "triage_corr_heatmap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    rep.append(f"![triage corr]({path.relative_to(DERIVED).as_posix()})\n")


# ---------- E. Candidate new features -------------------------------

def build_candidates(triage: pd.DataFrame,
                       fourh: pd.DataFrame) -> pd.DataFrame:
    """Construct candidate features for evaluation."""
    cand = pd.DataFrame({"encounter_id": triage["encounter_id"]})

    # Convenience
    hr_t = pd.to_numeric(triage["triage_heart_rate"], errors="coerce")
    rr_t = pd.to_numeric(triage["triage_respiratory_rate"], errors="coerce")
    sbp_t = pd.to_numeric(triage["triage_snapshot.systolic_bp"],
                            errors="coerce")
    dbp_t = pd.to_numeric(triage["triage_snapshot.diastolic_bp"],
                            errors="coerce")
    spo2_t = pd.to_numeric(triage["triage_snapshot.oxygen_saturation"],
                             errors="coerce")
    temp_t = pd.to_numeric(triage["triage_temperature_c"], errors="coerce")
    gcs_t = pd.to_numeric(triage["triage_gcs"], errors="coerce")
    age_t = pd.to_numeric(triage["triage_age"], errors="coerce")
    onset = pd.to_numeric(triage.get("note_onset_minutes",
                                       pd.Series([np.nan]*len(triage))),
                            errors="coerce")
    gluc_t = pd.to_numeric(triage["triage_lab_glucose"], errors="coerce")
    ph_t = pd.to_numeric(triage["triage_lab_ph"], errors="coerce")
    k_t = pd.to_numeric(triage["triage_lab_potassium"], errors="coerce")
    na_t = pd.to_numeric(triage["triage_lab_sodium"], errors="coerce")
    ag_t = pd.to_numeric(triage["triage_lab_anion_gap"], errors="coerce")

    # 1. Hemodynamic composites
    cand["cand_shock_index"] = hr_t / sbp_t.clip(lower=1)
    cand["cand_mod_shock_index"] = hr_t / ((2 * dbp_t + sbp_t) / 3).clip(lower=1)
    cand["cand_pulse_pressure"] = sbp_t - dbp_t
    cand["cand_map"] = dbp_t + (sbp_t - dbp_t) / 3.0
    cand["cand_shock_index_age"] = (hr_t / sbp_t.clip(lower=1)) * age_t
    cand["cand_rate_pressure_product"] = hr_t * sbp_t  # myocardial workload proxy
    cand["cand_hr_temp_product"] = hr_t * temp_t       # sympathetic intensity

    # 2. Lab composites
    cand["cand_k_extreme"] = ((k_t < 3.5) | (k_t > 5.0)).astype(int)
    cand["cand_na_extreme"] = ((na_t < 135) | (na_t > 145)).astype(int)
    cand["cand_glucose_extreme"] = ((gluc_t < 70) | (gluc_t > 200)).astype(int)
    cand["cand_high_anion_gap"] = (ag_t > 12).astype(int)
    cand["cand_acidosis"] = (ph_t < 7.35).astype(int)
    cand["cand_alkalosis"] = (ph_t > 7.45).astype(int)
    cand["cand_poc_abn_count"] = (
        cand["cand_k_extreme"] + cand["cand_na_extreme"]
        + cand["cand_glucose_extreme"] + cand["cand_high_anion_gap"]
        + cand["cand_acidosis"] + cand["cand_alkalosis"]
    )

    # 3. Vital risk-points (NEWS2-inspired, simplified)
    def points_hr(v):
        if pd.isna(v): return np.nan
        if v <= 40 or v >= 131: return 3
        if v <= 50 or v >= 111: return 2
        if v <= 60 or v >= 91: return 1
        return 0
    def points_rr(v):
        if pd.isna(v): return np.nan
        if v <= 8 or v >= 25: return 3
        if v >= 21: return 2
        if v >= 12 and v <= 20: return 0
        return 1
    def points_sbp(v):
        if pd.isna(v): return np.nan
        if v <= 90 or v >= 220: return 3
        if v <= 100: return 2
        if v <= 110: return 1
        return 0
    def points_spo2(v):
        if pd.isna(v): return np.nan
        if v <= 91: return 3
        if v <= 93: return 2
        if v <= 95: return 1
        return 0
    def points_temp(v):
        if pd.isna(v): return np.nan
        if v <= 35.0 or v >= 39.1: return 3
        if v >= 38.1: return 2
        if v <= 36.0: return 1
        return 0
    def points_gcs(v):
        if pd.isna(v): return np.nan
        if v <= 11: return 3
        if v <= 13: return 2
        if v <= 14: return 1
        return 0

    cand["cand_news_hr"] = hr_t.apply(points_hr)
    cand["cand_news_rr"] = rr_t.apply(points_rr)
    cand["cand_news_sbp"] = sbp_t.apply(points_sbp)
    cand["cand_news_spo2"] = spo2_t.apply(points_spo2)
    cand["cand_news_temp"] = temp_t.apply(points_temp)
    cand["cand_news_gcs"] = gcs_t.apply(points_gcs)
    cand["cand_news_total"] = cand[[c for c in cand.columns
                                       if c.startswith("cand_news_")]].sum(axis=1)
    cand["cand_news_high_risk"] = (cand["cand_news_total"] >= 5).astype(int)

    # 4. Onset-adjusted severity (only meaningful when onset parsed)
    # Idea: rapid onset + severe presentation -> drug-class-discriminative
    cand["cand_onset_x_news"] = cand["cand_news_total"] / np.maximum(onset, 1)
    cand["cand_onset_x_hr"] = hr_t / np.maximum(onset, 1)
    cand["cand_log_onset"] = np.log1p(onset)
    cand["cand_is_fast_onset"] = (onset < 60).astype(int)
    cand["cand_is_slow_onset"] = (onset > 180).astype(int)

    # 5. PMH composite
    cand["cand_pmh_count"] = (
        pd.to_numeric(triage.get("triage_mh_psych", 0), errors="coerce").fillna(0)
        + pd.to_numeric(triage.get("triage_mh_cardiac", 0), errors="coerce").fillna(0)
        + pd.to_numeric(triage.get("triage_mh_pulm", 0), errors="coerce").fillna(0)
        + pd.to_numeric(triage.get("triage_mh_renal", 0), errors="coerce").fillna(0)
        + pd.to_numeric(triage.get("triage_mh_substance_use", 0), errors="coerce").fillna(0)
    )

    # 6. Sympathetic intensity (vital cluster)
    cand["cand_sympathetic_score"] = (
        (hr_t > 100).astype(int) + (sbp_t > 140).astype(int)
        + (temp_t > 38.0).astype(int) + (rr_t > 22).astype(int)
    )

    # 7. CNS-depression score
    cand["cand_cns_depression_score"] = (
        (gcs_t < 14).astype(int) + (rr_t < 12).astype(int)
        + (spo2_t < 93).astype(int)
    )

    return cand


def section_e_candidates(rep: list[str], triage: pd.DataFrame,
                          fourh: pd.DataFrame, y_drug: np.ndarray,
                          y_dispo: np.ndarray,
                          dispo_cohort_ids: list[str]) -> pd.DataFrame:
    rep.append("## E. Candidate new-feature exploration\n")
    cand = build_candidates(triage, fourh)

    # Coverage
    rep.append("### Coverage (non-null counts)\n")
    rep.append("| Candidate | n_non_null | mean | median | min | max |")
    rep.append("|---|---:|---:|---:|---:|---:|")
    feat_cols = [c for c in cand.columns if c.startswith("cand_")]
    for c in feat_cols:
        s = pd.to_numeric(cand[c], errors="coerce").dropna()
        if s.empty:
            rep.append(f"| {c} | 0 | — | — | — | — |")
            continue
        rep.append(f"| {c} | {len(s)} | "
                   f"{fmt(s.mean())} | {fmt(s.median())} | "
                   f"{fmt(s.min())} | {fmt(s.max())} |")
    rep.append("")

    # MI lift vs Task-1 (drug)
    X_drug = cand[feat_cols].copy()
    mi_drug = mi_against(X_drug, y_drug)

    # MI lift vs Task-2 (dispo) — drug-positive cohort only
    cand_cohort = cand[cand["encounter_id"].isin(dispo_cohort_ids)].reset_index(drop=True)
    X_dispo = cand_cohort[feat_cols].copy()
    mi_dispo = mi_against(X_dispo, y_dispo)

    rep.append("### Candidate mutual information vs targets\n")
    rep.append("Top MI for each task (higher = more informative):\n")
    rep.append("| Candidate | MI(drug) | MI(disposition) |")
    rep.append("|---|---:|---:|")
    combined = pd.DataFrame({"drug": mi_drug, "dispo": mi_dispo}).fillna(0.0)
    combined = combined.sort_values("dispo", ascending=False)
    # Only show candidate features (skip dummies that get introduced if any)
    for c in feat_cols:
        if c in combined.index:
            rep.append(f"| {c} | {combined.loc[c, 'drug']:.4f} | "
                       f"{combined.loc[c, 'dispo']:.4f} |")
    rep.append("")

    # Side-by-side plot: candidates ranked by max(MI)
    combined["max_mi"] = combined.max(axis=1)
    top = combined.sort_values("max_mi", ascending=False).head(15).index
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.4
    idx = np.arange(len(top))
    ax.barh(idx - width/2, combined.loc[top, "drug"], width, label="vs drug")
    ax.barh(idx + width/2, combined.loc[top, "dispo"], width, label="vs dispo")
    ax.set_yticks(idx)
    ax.set_yticklabels(top, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual information (nats)")
    ax.set_title("Candidate features — MI vs Task 1 (drug) and Task 2 (dispo)")
    ax.legend()
    fig.tight_layout()
    path = PLOTS / "candidate_features_mi.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    rep.append(f"![candidates]({path.relative_to(DERIVED).as_posix()})\n")

    return cand


# ---------- F. Recommendations + commit ------------------------------

def section_f_commit(rep: list[str], cand: pd.DataFrame,
                       y_drug: np.ndarray, y_dispo: np.ndarray,
                       dispo_cohort_ids: list[str]) -> None:
    rep.append("## F. Commit recommendation\n")

    feat_cols = [c for c in cand.columns if c.startswith("cand_")]
    X_drug = cand[feat_cols]
    mi_drug = mi_against(X_drug, y_drug)
    cand_cohort = cand[cand["encounter_id"].isin(dispo_cohort_ids)].reset_index(drop=True)
    X_dispo = cand_cohort[feat_cols]
    mi_dispo = mi_against(X_dispo, y_dispo)
    combined = pd.DataFrame({"drug": mi_drug, "dispo": mi_dispo}).fillna(0.0)
    combined["max_mi"] = combined.max(axis=1)

    # Commit candidates whose max(MI) >= 0.03 (rough heuristic — slightly
    # above the noise floor we see for unrelated features)
    THRESH = 0.03
    keep = combined[combined["max_mi"] >= THRESH].index.tolist()
    keep = [c for c in keep if c in feat_cols]

    rep.append(f"Commit threshold: max(MI) >= {THRESH}\n")
    rep.append(f"Kept: **{len(keep)} candidates**\n")
    rep.append("\n| Committed feature | MI(drug) | MI(dispo) |")
    rep.append("|---|---:|---:|")
    for c in keep:
        rep.append(f"| {c} | {combined.loc[c, 'drug']:.4f} | "
                   f"{combined.loc[c, 'dispo']:.4f} |")
    rep.append("")

    out = cand[["encounter_id"] + keep]
    out.to_csv(CANDIDATES_OUT, index=False)
    rep.append(f"Wrote `{CANDIDATES_OUT.name}` "
               f"({out.shape[0]} rows x {out.shape[1]} cols). "
               f"Merge into features_triage.csv / features_fourh.csv "
               f"as the next pipeline step before re-training.\n")


# ---------- main ------------------------------------------------------

def main() -> None:
    import sklearn, scipy
    rep: list[str] = []
    rep.append("# Analysis: Descriptive EDA + Candidate-Feature Exploration\n")
    rep.append("## Question\n")
    rep.append("What does the dataset look like, what have we built, "
               "and what new features might lift Task-1 or Task-2 performance?\n")

    triage = pd.read_csv(DERIVED / "features_triage.csv")
    fourh = pd.read_csv(DERIVED / "features_fourh.csv")
    probs = pd.read_csv(DERIVED / "probs_avg.csv",
                         keep_default_na=False, na_values=[""])
    # Outcomes (canonical label source — feature tables no longer
    # carry encounter_disposition_label).
    outcomes = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "encounter_disposition_label"]]

    # Labels
    drug_map = {"Kraken Candy": 0, "Triton Tabs": 1, "Coral Dust": 2,
                 "None": 3}
    dispo_map = {"Discharge": 0, "Floor": 1, "ICU": 2}
    argmax = probs.set_index("encounter_id")["argmax_class"]
    y_drug = argmax.reindex(triage["encounter_id"]).map(drug_map).to_numpy()
    cohort_ids = probs[probs["argmax_class"] != "None"]["encounter_id"].tolist()
    cohort = (fourh[fourh["encounter_id"].isin(cohort_ids)]
                .merge(outcomes, on="encounter_id", how="left")
                .reset_index(drop=True))
    y_dispo = cohort["encounter_disposition_label"].map(dispo_map).to_numpy()

    rep.append("## Data\n")
    rep.append(f"- features_triage.csv: **{triage.shape}**")
    rep.append(f"- features_fourh.csv:  **{fourh.shape}**")
    rep.append(f"- probs_avg.csv:       **{probs.shape}**  "
               f"(soft Task-1 labels, argmax distribution shown below)")
    rep.append(f"- Drug-positive cohort (Task 2): **{len(cohort)} patients**\n")

    section_a_profile(rep, triage, fourh)
    section_b_families(rep, triage, fourh)
    section_c_outliers(rep, triage)
    section_d_correlation(rep, triage)
    cand = section_e_candidates(rep, triage, fourh, y_drug, y_dispo,
                                  cohort_ids)
    section_f_commit(rep, cand, y_drug, y_dispo, cohort_ids)

    rep.append("## Reproducibility\n")
    rep.append(f"- Code: `src/eda/eda_descriptive.py`")
    rep.append(f"- Seed: 42")
    rep.append(f"- Libraries: pandas={pd.__version__}, numpy={np.__version__}, "
               f"scikit-learn={sklearn.__version__}, scipy={scipy.__version__}")
    rep.append(f"- Outputs: `{REPORT.name}`, "
               f"`exploratory_features.csv`, "
               f"plots in `eda_plots/`\n")

    REPORT.write_text("\n".join(rep), encoding="utf-8")
    print(f"\nWrote: {REPORT}")
    print(f"Wrote: {CANDIDATES_OUT}")
    print(f"Plots: {PLOTS}/")


if __name__ == "__main__":
    main()
