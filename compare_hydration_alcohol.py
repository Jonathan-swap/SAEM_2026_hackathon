"""Phase-1 vs Phase-2 comparison focused on hydration + alcohol.

There is no explicit `triage_hydration_status` or `triage_alcohol_status`
column in either release; this script builds clinical proxies from
what IS available and compares both phases.

Proxies used:

  Hydration
    - triage_lab_sodium               higher -> volume depletion
    - triage_lab_anion_gap            elevated -> contraction
    - triage_lab_glucose              elevated -> osmotic
    - cand_shock_index                HR / SBP (volume-status surrogate)
    - ivf_count_0_4h (xlsx)           IV-fluid bolus count, 4h horizon
    - narrative keyword density       hydration / dehydration / fluid /
                                       crystalloid / IV bolus mentions

  Alcohol
    - triage_mh_substance_use         PMH flag (binary)
    - narrative keyword density       alcohol / etoh / intox / drunk /
                                       binge / beer / liquor / drinking
                                       mentions

Comparisons use DENSITY (not count) histograms / bar charts so the
Phase-1 (n=261) and Phase-2 (n=139) cohorts can be compared visually
without sample-size confound.

Output:
  derived/phase2/hydration_alcohol_report.md
  derived/phase2/hydration_alcohol_table.csv
  derived/phase2/eda_plots/hydration_density.png
  derived/phase2/eda_plots/alcohol_density.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA1 = ROOT / "data" / "Hackathon_Data_Release_1_SHARE.xlsx"
DATA2 = ROOT / "data2" / "Hackathon_Data_Release_2_SHARE.xlsx"
DERIVED = ROOT / "derived"
PHASE2 = DERIVED / "phase2"
PLOTS = PHASE2 / "eda_plots"

# Proxies sourced from the engineered features.
HYDRATION_NUMERIC = [
    ("triage_lab_sodium", "Sodium (mmol/L)"),
    ("triage_lab_anion_gap", "Anion gap (mmol/L)"),
    ("triage_lab_glucose", "POC glucose (mg/dL)"),
    ("cand_shock_index", "Shock index (HR/SBP)"),
]
HYDRATION_INTERVENTION = "ed_course_reassessment_4h.ivf_count_0_4h"

# Narrative-note text fields where keywords are searched.
NARRATIVE_COLS = [
    "narrative_notes_structured_brief_hpi",
    "narrative_notes_structured_hpi",
    "narrative_notes_structured_mdm",
    "narrative_notes_structured_clinical_course",
    "narrative.notes_structured_ed_meds_procedures",
]

# Word-boundary regex (case-insensitive on lowercased text).
HYDRATION_KWS = re.compile(
    r"\b(dehydrat|hydrat|fluid|crystalloid|iv\s*bolus|ns\s*bolus|"
    r"normal\s*saline|lactated\s*ringer|lr\s*bolus|volume\s*depleted|"
    r"oral\s*rehydrat)\w*",
    re.IGNORECASE,
)
ALCOHOL_KWS = re.compile(
    r"\b(alcohol|etoh|intoxicat|drunk|binge|drinking|beer|liquor|"
    r"wine|hangover|withdrawal)\w*",
    re.IGNORECASE,
)


def _open_xlsx(p: Path) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(p, engine="openpyxl")
    return {s: pd.read_excel(xl, sheet_name=s, engine="openpyxl")
            for s in xl.sheet_names}


def _count_matches(text: object, pat: re.Pattern) -> int:
    if not isinstance(text, str):
        return 0
    return len(pat.findall(text))


def keyword_density(fourh: pd.DataFrame, pat: re.Pattern) -> pd.Series:
    """Total keyword hits across all narrative cols per encounter."""
    totals = pd.Series(0, index=fourh.index, dtype=int)
    for c in NARRATIVE_COLS:
        if c not in fourh.columns:
            continue
        totals = totals + fourh[c].map(lambda t: _count_matches(t, pat))
    return totals


# ---------- Density-plot helper ----------------------------------------

def density_plot(ax, s1: pd.Series, s2: pd.Series, title: str,
                  xlabel: str = "") -> None:
    """Histogram density overlay (auto-bin); falls back to bar plot
    for low-cardinality data."""
    s1 = pd.to_numeric(s1, errors="coerce").dropna()
    s2 = pd.to_numeric(s2, errors="coerce").dropna()
    if s1.empty and s2.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title(title, fontsize=9)
        return
    nunique = (s1.nunique() if not s1.empty else 0) + \
              (s2.nunique() if not s2.empty else 0)
    if nunique <= 12:
        # Discrete — show normalised value proportions side by side.
        vals = sorted(set(s1.unique()) | set(s2.unique()))
        x = np.arange(len(vals))
        w = 0.4
        p1 = [float((s1 == v).mean()) if not s1.empty else 0
              for v in vals]
        p2 = [float((s2 == v).mean()) if not s2.empty else 0
              for v in vals]
        ax.bar(x - w / 2, p1, w, color="#1f77b4",
               label=f"P1 (n={len(s1)})", alpha=0.85)
        ax.bar(x + w / 2, p2, w, color="#d62728",
               label=f"P2 (n={len(s2)})", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in vals], fontsize=7)
        ax.set_ylabel("proportion")
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
        ax.set_ylabel("density")
    ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)


# ---------- Summary stats helper ---------------------------------------

def numeric_compare(s1: pd.Series, s2: pd.Series, label: str
                     ) -> dict:
    s1 = pd.to_numeric(s1, errors="coerce").dropna()
    s2 = pd.to_numeric(s2, errors="coerce").dropna()
    out = {
        "feature": label,
        "phase1_n": len(s1),
        "phase2_n": len(s2),
        "phase1_mean": float(s1.mean()) if len(s1) else np.nan,
        "phase2_mean": float(s2.mean()) if len(s2) else np.nan,
        "phase1_median": float(s1.median()) if len(s1) else np.nan,
        "phase2_median": float(s2.median()) if len(s2) else np.nan,
        "phase1_pct_nonzero": float((s1 > 0).mean()) if len(s1) else np.nan,
        "phase2_pct_nonzero": float((s2 > 0).mean()) if len(s2) else np.nan,
    }
    if len(s1) >= 5 and len(s2) >= 5:
        sd1 = s1.std(ddof=1); sd2 = s2.std(ddof=1)
        pooled = float(np.sqrt(((len(s1) - 1) * sd1**2
                                + (len(s2) - 1) * sd2**2)
                               / max(len(s1) + len(s2) - 2, 1)))
        out["cohen_d"] = float((s2.mean() - s1.mean()) / pooled) \
            if pooled > 0 else np.nan
    else:
        out["cohen_d"] = np.nan
    return out


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    print("Loading raw xlsx (Phase-1 + Phase-2)…")
    s1 = _open_xlsx(DATA1)
    s2 = _open_xlsx(DATA2)
    p1_triage = s1["Triage_Data"]
    p2_triage = s2["Triage_Data"]
    p1_fourh = s1["Four_Hour_Data"]
    p2_fourh = s2["Four_Hour_Data"]
    print(f"  Phase-1 triage: {p1_triage.shape}, "
          f"Phase-2 triage: {p2_triage.shape}")

    # Engineered features for the lab-derived proxies + cand_shock_index.
    p1_feat = pd.read_csv(DERIVED / "features_triage.csv")
    p2_feat = pd.read_csv(PHASE2 / "features_triage.csv")

    rows: list[dict] = []

    # ---------- HYDRATION ----------
    print("\n=== Hydration proxies ===")
    for col, _ in HYDRATION_NUMERIC:
        if col not in p1_feat.columns or col not in p2_feat.columns:
            print(f"  skip (missing in one phase): {col}")
            continue
        r = numeric_compare(p1_feat[col], p2_feat[col], f"hydration:{col}")
        rows.append({"category": "hydration", **r})
        print(f"  {col:30s}  P1={r['phase1_mean']:.2f}  "
              f"P2={r['phase2_mean']:.2f}  d={r['cohen_d']:+.2f}")

    # IV fluid count (from raw xlsx)
    if HYDRATION_INTERVENTION in p1_fourh.columns:
        r = numeric_compare(p1_fourh[HYDRATION_INTERVENTION],
                              p2_fourh[HYDRATION_INTERVENTION],
                              "hydration:ivf_count_0_4h")
        rows.append({"category": "hydration", **r})
        print(f"  {'ivf_count_0_4h':30s}  P1={r['phase1_mean']:.2f}  "
              f"P2={r['phase2_mean']:.2f}  d={r['cohen_d']:+.2f}")

    # Narrative keyword density
    hyd_p1 = keyword_density(p1_fourh, HYDRATION_KWS)
    hyd_p2 = keyword_density(p2_fourh, HYDRATION_KWS)
    r = numeric_compare(hyd_p1, hyd_p2, "hydration:narrative_keyword_count")
    rows.append({"category": "hydration", **r})
    print(f"  {'narrative kw hits/note':30s}  "
          f"P1_mean={r['phase1_mean']:.2f}  P2_mean={r['phase2_mean']:.2f}  "
          f"P1_any={r['phase1_pct_nonzero']:.1%}  "
          f"P2_any={r['phase2_pct_nonzero']:.1%}")

    # ---------- ALCOHOL ----------
    print("\n=== Alcohol proxies ===")
    if "triage_mh_substance_use" in p1_triage.columns:
        r = numeric_compare(p1_triage["triage_mh_substance_use"],
                              p2_triage["triage_mh_substance_use"],
                              "alcohol:triage_mh_substance_use")
        rows.append({"category": "alcohol", **r})
        print(f"  {'mh_substance_use (any)':30s}  "
              f"P1={r['phase1_pct_nonzero']:.1%}  "
              f"P2={r['phase2_pct_nonzero']:.1%}  d={r['cohen_d']:+.2f}")

    alc_p1 = keyword_density(p1_fourh, ALCOHOL_KWS)
    alc_p2 = keyword_density(p2_fourh, ALCOHOL_KWS)
    r = numeric_compare(alc_p1, alc_p2, "alcohol:narrative_keyword_count")
    rows.append({"category": "alcohol", **r})
    print(f"  {'narrative kw hits/note':30s}  "
          f"P1_mean={r['phase1_mean']:.2f}  P2_mean={r['phase2_mean']:.2f}  "
          f"P1_any={r['phase1_pct_nonzero']:.1%}  "
          f"P2_any={r['phase2_pct_nonzero']:.1%}")

    # Per-keyword breakdown for alcohol — useful because the bucket
    # is narrow.
    print("\n  Per-keyword breakdown (% of notes mentioning each):")
    individual_kws = ["alcohol", "etoh", "intoxicat", "drunk", "binge",
                       "drinking", "beer", "liquor", "wine", "hangover",
                       "withdrawal"]
    alcohol_kw_rows = []
    for kw in individual_kws:
        pat = re.compile(rf"\b{kw}\w*", re.IGNORECASE)
        p1_any = (keyword_density(p1_fourh, pat) > 0).mean()
        p2_any = (keyword_density(p2_fourh, pat) > 0).mean()
        alcohol_kw_rows.append({
            "keyword": kw,
            "phase1_pct_notes": float(p1_any),
            "phase2_pct_notes": float(p2_any),
            "delta": float(p2_any - p1_any),
        })
        print(f"    {kw:14s}  P1={p1_any:6.2%}  P2={p2_any:6.2%}  "
              f"delta={(p2_any-p1_any)*100:+.1f}pp")

    # ---------- Persist table ----------
    out = pd.DataFrame(rows)
    out.to_csv(PHASE2 / "hydration_alcohol_table.csv", index=False)
    kw_out = pd.DataFrame(alcohol_kw_rows)
    kw_out.to_csv(PHASE2 / "alcohol_keyword_breakdown.csv", index=False)

    # ---------- Density plots ----------
    print("\nWriting density plots…")
    # Hydration grid: 4 numeric proxies + ivf_count + kw_count = 6 panels
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    for ax, (col, xl) in zip(axes, HYDRATION_NUMERIC):
        if col in p1_feat.columns and col in p2_feat.columns:
            density_plot(ax, p1_feat[col], p2_feat[col], xl, col)
    if HYDRATION_INTERVENTION in p1_fourh.columns:
        density_plot(axes[4], p1_fourh[HYDRATION_INTERVENTION],
                      p2_fourh[HYDRATION_INTERVENTION],
                      "IV fluid bolus count (0–4h)", "ivf_count_0_4h")
    density_plot(axes[5], hyd_p1, hyd_p2,
                  "Narrative hydration-keyword hits per encounter",
                  "kw_count")
    fig.suptitle("Hydration proxies — Phase-1 vs Phase-2 "
                  "(density-normalised)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "hydration_density.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  hydration_density.png")

    # Alcohol grid: mh_substance_use + narrative kw count + per-keyword bar
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    if "triage_mh_substance_use" in p1_triage.columns:
        density_plot(axes[0], p1_triage["triage_mh_substance_use"],
                      p2_triage["triage_mh_substance_use"],
                      "PMH substance-use flag",
                      "triage_mh_substance_use (0/1)")
    density_plot(axes[1], alc_p1, alc_p2,
                  "Narrative alcohol-keyword hits per encounter",
                  "kw_count")
    # Per-keyword "any-mention rate" bar chart (already proportions)
    kw_df = pd.DataFrame(alcohol_kw_rows)
    x = np.arange(len(kw_df))
    w = 0.4
    axes[2].bar(x - w / 2, kw_df["phase1_pct_notes"], w,
                 color="#1f77b4", label=f"P1 (n={len(p1_fourh)})",
                 alpha=0.85)
    axes[2].bar(x + w / 2, kw_df["phase2_pct_notes"], w,
                 color="#d62728", label=f"P2 (n={len(p2_fourh)})",
                 alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(kw_df["keyword"], rotation=45,
                              ha="right", fontsize=8)
    axes[2].set_ylabel("fraction of encounters mentioning")
    axes[2].set_title("Per-keyword mention rate "
                        "(normalised proportions)", fontsize=10)
    axes[2].legend(fontsize=7)
    fig.suptitle("Alcohol proxies — Phase-1 vs Phase-2 "
                  "(density-normalised)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "alcohol_density.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  alcohol_density.png")

    # ---------- Markdown report ----------
    def md_row(d: dict) -> str:
        cd = d.get("cohen_d", float("nan"))
        cd_str = f"{cd:+.2f}" if cd == cd else "—"
        return (
            f"| `{d['feature']}` | {d['phase1_n']} | {d['phase2_n']} | "
            f"{d['phase1_mean']:.3f} | {d['phase2_mean']:.3f} | "
            f"{d['phase1_pct_nonzero']:.1%} | "
            f"{d['phase2_pct_nonzero']:.1%} | {cd_str} |"
        )

    md = [
        "# Hydration & alcohol — Phase-1 vs Phase-2",
        "",
        "No release contains an explicit `triage_hydration_status` or "
        "`triage_alcohol_status` column. This report compares the "
        "two phases on the closest available clinical proxies, with "
        "**density-normalised plots** so the 261-vs-139 cohort-size "
        "difference doesn't distort the visual comparison.",
        "",
        "## Hydration proxies",
        "",
        "| Proxy | P1 n | P2 n | P1 mean | P2 mean | P1 % > 0 | P2 % > 0 | Cohen d |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["category"] == "hydration":
            md.append(md_row(r))
    md.append("")
    md.append("![hydration](eda_plots/hydration_density.png)")
    md.append("")
    md.append("**Interpretation cheat sheet** — what each proxy "
              "actually tells you about hydration status:")
    md.append("")
    md.append("- **Sodium**: a higher mean suggests more volume "
              "contraction in that cohort.")
    md.append("- **Anion gap**: elevated → metabolic acidosis from "
              "lactate / ketones (often seen in volume depletion).")
    md.append("- **Glucose**: hyperglycaemia can drive osmotic "
              "diuresis (treatment-relevant for dehydration).")
    md.append("- **Shock index (HR/SBP)**: > 0.9 → likely volume "
              "depletion / impending shock.")
    md.append("- **IV fluid bolus count**: clinician-administered "
              "rehydration; downstream proxy.")
    md.append("- **Narrative-keyword hits**: free-text clinician "
              "mentions of dehydration / hydration / IV-fluid words "
              "across the four narrative blocks. Useful when the "
              "structured fields don't capture it directly.")
    md.append("")

    md += [
        "## Alcohol proxies",
        "",
        "| Proxy | P1 n | P2 n | P1 mean | P2 mean | P1 % > 0 | P2 % > 0 | Cohen d |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["category"] == "alcohol":
            md.append(md_row(r))
    md.append("")
    md.append("![alcohol](eda_plots/alcohol_density.png)")
    md.append("")
    md.append("### Per-keyword mention-rate breakdown")
    md.append("")
    md.append("Fraction of encounters whose narrative notes mention "
              "each keyword at least once.")
    md.append("")
    md.append("| Keyword | P1 % notes | P2 % notes | Δ (pp) |")
    md.append("|---|---:|---:|---:|")
    for kr in alcohol_kw_rows:
        md.append(
            f"| `{kr['keyword']}` | {kr['phase1_pct_notes']:.2%} | "
            f"{kr['phase2_pct_notes']:.2%} | "
            f"{kr['delta']*100:+.1f} |"
        )
    md.append("")

    md += [
        "## Files",
        "",
        "- `derived/phase2/hydration_alcohol_table.csv`",
        "- `derived/phase2/alcohol_keyword_breakdown.csv`",
        "- `derived/phase2/eda_plots/hydration_density.png`",
        "- `derived/phase2/eda_plots/alcohol_density.png`",
        "",
    ]

    (PHASE2 / "hydration_alcohol_report.md").write_text(
        "\n".join(md), encoding="utf-8",
    )
    print("\nWrote hydration_alcohol_report.md + table CSVs")


if __name__ == "__main__":
    main()
