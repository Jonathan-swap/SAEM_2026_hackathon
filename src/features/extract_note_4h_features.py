"""Extract HPI / MDM narrative features from Four_Hour_Data.

Mirrors the 4 HPI/MDM features documented in
`task2_disposition/scripts/disposition_prediction.py` (which lists them
in its docstring but does NOT actually implement them in code). Parses
`narrative_notes_structured_hpi` and `narrative_notes_structured_mdm`
from the 4-hour sheet of the released xlsx.

Emits per-encounter (all 4h-only):
  hpi_word_count_4h      : int word count of HPI text (0 if missing)
  hpi_has_severe_4h      : binary, 1 if HPI mentions "severe(ly)"
  mdm_word_count_4h      : int word count of MDM text (0 if missing)
  mdm_severity_tier_4h   : ordinal 1/2/3 from MDM language, NaN if no
                           keyword found. 1 = low/mild/stable,
                           2 = moderate/intermediate,
                           3 = high/severe/critical.
                           Also matches explicit "tier 1/2/3" phrasing.

Time-leakage protection:
  - All emitted columns carry the `_4h` suffix so the leakage sentinel
    in extract_structured.py automatically rejects them from
    features_triage.csv.
  - This script ONLY merges into features_fourh.csv. features_triage.csv
    is never read or written.
  - A post-merge assertion confirms none of these columns are present
    in features_triage.csv.

CAVEAT (label leakage for severity target):
  `mdm_severity_tier_4h` may be a direct paraphrase of the clinician's
  severity judgment. The upstream task2 docstring warns:
  "mdm_severity_tier excluded from this target's [severity_score]
  feature set to avoid leakage". Downstream models predicting
  severity_score should drop this feature explicitly. It is retained
  here because it is a legitimate predictor for *disposition* (which
  is the project's primary Task-2 target).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
XLSX = ROOT / "data" / "Hackathon_Data_Release_1_SHARE.xlsx"

SEVERE_RE = re.compile(r"\bsever\w*\b", re.IGNORECASE)

# Severity tier patterns: try explicit "tier N" first, then keyword
# mapping. Order matters — explicit beats inferred. Tier 3 (high) is
# matched first so "severe" inside MDM elevates the row even when
# "moderate" also appears (clinical reality: worst-mentioned wins).
TIER_EXPLICIT_RE = re.compile(
    r"\btier\s+([1-3])\b", re.IGNORECASE,
)
TIER_3_RE = re.compile(
    r"\b(?:high(?:\s+severity)?|severe|critical|life[-\s]threatening)\b",
    re.IGNORECASE,
)
TIER_2_RE = re.compile(
    r"\b(?:moderate(?:\s+severity)?|intermediate)\b", re.IGNORECASE,
)
TIER_1_RE = re.compile(
    r"\b(?:low(?:\s+severity)?|mild|stable|minor)\b", re.IGNORECASE,
)


def parse_hpi(text: object) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"hpi_word_count_4h": 0, "hpi_has_severe_4h": 0}
    return {
        "hpi_word_count_4h": len(text.split()),
        "hpi_has_severe_4h": int(bool(SEVERE_RE.search(text))),
    }


def parse_mdm(text: object) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"mdm_word_count_4h": 0, "mdm_severity_tier_4h": float("nan")}
    out: dict = {"mdm_word_count_4h": len(text.split())}
    m = TIER_EXPLICIT_RE.search(text)
    if m:
        out["mdm_severity_tier_4h"] = int(m.group(1))
        return out
    if TIER_3_RE.search(text):
        out["mdm_severity_tier_4h"] = 3
    elif TIER_2_RE.search(text):
        out["mdm_severity_tier_4h"] = 2
    elif TIER_1_RE.search(text):
        out["mdm_severity_tier_4h"] = 1
    else:
        out["mdm_severity_tier_4h"] = float("nan")
    return out


def main() -> None:
    fourh = pd.read_excel(XLSX, sheet_name="Four_Hour_Data", engine="openpyxl")

    hpi_col = "narrative_notes_structured_hpi"
    mdm_col = "narrative_notes_structured_mdm"
    for col in (hpi_col, mdm_col):
        if col not in fourh.columns:
            print(f"FAIL: Four_Hour_Data is missing required column {col!r}")
            raise SystemExit(1)

    hpi = pd.DataFrame([parse_hpi(t) for t in fourh[hpi_col]])
    mdm = pd.DataFrame([parse_mdm(t) for t in fourh[mdm_col]])
    parsed = pd.concat([hpi, mdm], axis=1)
    parsed.insert(0, "encounter_id", fourh["encounter_id"].values)

    parsed_path = DERIVED / "note_4h_features.csv"
    parsed.to_csv(parsed_path, index=False)
    print(f"Wrote {parsed.shape} -> {parsed_path}")

    # ---- Coverage diagnostics (aggregate only — no row values) ----
    n = len(parsed)
    print(f"\nCoverage on {n} encounters:")
    print(f"  hpi non-empty (word_count > 0): "
          f"{(parsed['hpi_word_count_4h'] > 0).sum()} / {n}")
    print(f"  hpi_has_severe_4h == 1:         "
          f"{int(parsed['hpi_has_severe_4h'].sum())} / {n}")
    print(f"  mdm non-empty (word_count > 0): "
          f"{(parsed['mdm_word_count_4h'] > 0).sum()} / {n}")
    print(f"  mdm_severity_tier_4h matched:   "
          f"{parsed['mdm_severity_tier_4h'].notna().sum()} / {n}")
    if parsed["mdm_severity_tier_4h"].notna().any():
        tier_counts = (parsed["mdm_severity_tier_4h"]
                       .value_counts().sort_index().to_dict())
        print(f"  mdm_severity_tier_4h distribution: {tier_counts}")

    # ---- Merge into features_fourh.csv (idempotent) ----
    new_cols = [c for c in parsed.columns if c != "encounter_id"]
    fourh_path = DERIVED / "features_fourh.csv"
    if not fourh_path.exists():
        print(f"\nskip merge: {fourh_path} not present "
              "(run extract_structured.py first)")
        return
    df_fourh = pd.read_csv(fourh_path)
    prior = [c for c in df_fourh.columns if c in new_cols]
    if prior:
        df_fourh = df_fourh.drop(columns=prior)
    merged = df_fourh.merge(parsed, on="encounter_id", how="left")
    merged.to_csv(fourh_path, index=False)
    print(f"\nMerged into features_fourh.csv: {merged.shape}")

    # ---- Time-leakage guard: confirm cols absent from features_triage ----
    triage_path = DERIVED / "features_triage.csv"
    if triage_path.exists():
        df_triage = pd.read_csv(triage_path, nrows=0)
        leaked = [c for c in new_cols if c in df_triage.columns]
        if leaked:
            print(f"FAIL: 4h note features leaked into features_triage.csv: "
                  f"{leaked}")
            raise SystemExit(1)
        print(f"OK: features_triage.csv contains none of the new 4h note "
              f"features {new_cols}.")


if __name__ == "__main__":
    main()
