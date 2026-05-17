"""Extract structured features from `triage_brief_note`.

The triage brief note follows a recognizable template for festival
attendees:
  "<CC>. Festival attendee from <LOCATION> with symptom onset
   ~<MINUTES> minutes before arrival."

Non-festival presentations use unstructured phrasing.

Emits per-encounter:
  note_onset_minutes        : numeric, minutes between symptom onset
                              and arrival. NaN if not parseable.
  note_has_onset_phrase     : binary, 1 if the canonical phrase matched
  note_is_festival_template : binary, 1 if "Festival attendee" present
  note_location_*           : one-hot of {main_stage, medical_tent,
                              campground, food_village, shopping_area,
                              other, none}
  note_onset_bucket_*       : coarse buckets — fast (<60min), medium
                              (60-180min), slow (>180min), unknown
  note_char_len, note_word_count : structural

Clinical rationale for onset bucket:
  - sympathomimetic onset ~minutes -> rapid presentation
  - hallucinogen onset variable, often 60-120min
  - sedative onset delayed / re-distribution -> later presentation

Reads triage sheet of the xlsx directly. Merges output into both
features_triage.csv (no leakage — note is at triage) and
features_fourh.csv (note still available at 4h). Idempotent re-merge.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
XLSX = ROOT / "data" / "Hackathon_Data_Release_1_SHARE.xlsx"

ONSET_RE = re.compile(
    r"symptom onset\s*~?\s*(\d+(?:\.\d+)?)\s*min(?:ute)?s?\s*before\s*arrival",
    re.IGNORECASE,
)
FESTIVAL_TEMPLATE_RE = re.compile(
    r"festival attendee\s+from\s+([a-zA-Z\s]+?)\s+with\s+symptom",
    re.IGNORECASE,
)
LOCATIONS = {
    "main_stage":    re.compile(r"\bmain\s+stage\b", re.IGNORECASE),
    "medical_tent":  re.compile(r"\b(?:festival\s+)?medical\s+tent\b", re.IGNORECASE),
    "campground":    re.compile(r"\bcampground\b", re.IGNORECASE),
    "food_village":  re.compile(r"\bfood\s+village\b", re.IGNORECASE),
    "shopping_area": re.compile(r"\bshopping\s+area\b", re.IGNORECASE),
}


def parse_note(note: str) -> dict:
    out: dict = {
        "note_onset_minutes": float("nan"),
        "note_has_onset_phrase": 0,
        "note_is_festival_template": 0,
        "note_char_len": 0,
        "note_word_count": 0,
    }
    if not isinstance(note, str):
        return out
    out["note_char_len"] = len(note)
    out["note_word_count"] = len(note.split())

    m = ONSET_RE.search(note)
    if m:
        try:
            out["note_onset_minutes"] = float(m.group(1))
            out["note_has_onset_phrase"] = 1
        except ValueError:
            pass

    if FESTIVAL_TEMPLATE_RE.search(note):
        out["note_is_festival_template"] = 1

    # Location one-hot
    matched_any = False
    for loc, pat in LOCATIONS.items():
        hit = int(bool(pat.search(note)))
        out[f"note_location_{loc}"] = hit
        if hit:
            matched_any = True
    out["note_location_other"] = int(
        out["note_is_festival_template"] == 1 and not matched_any)
    out["note_location_none"] = int(
        out["note_is_festival_template"] == 0)

    # Onset bucket (only meaningful when phrase matched)
    onset = out["note_onset_minutes"]
    for b in ("fast", "medium", "slow", "unknown"):
        out[f"note_onset_bucket_{b}"] = 0
    if np.isnan(onset):
        out["note_onset_bucket_unknown"] = 1
    elif onset < 60:
        out["note_onset_bucket_fast"] = 1
    elif onset < 180:
        out["note_onset_bucket_medium"] = 1
    else:
        out["note_onset_bucket_slow"] = 1

    return out


def main() -> None:
    triage = pd.read_excel(XLSX, sheet_name="Triage_Data", engine="openpyxl")

    parsed = pd.DataFrame([parse_note(n) for n in triage["triage_brief_note"]])
    parsed.insert(0, "encounter_id", triage["encounter_id"].values)

    parsed_path = DERIVED / "note_features.csv"
    parsed.to_csv(parsed_path, index=False)
    print(f"Wrote {parsed.shape} -> {parsed_path}")

    # Coverage diagnostics
    n_onset = int(parsed["note_has_onset_phrase"].sum())
    n_festival = int(parsed["note_is_festival_template"].sum())
    print(f"\nOnset phrase parsed:        {n_onset}/{len(parsed)} "
          f"({n_onset/len(parsed)*100:.1f}%)")
    print(f"Festival template detected: {n_festival}/{len(parsed)} "
          f"({n_festival/len(parsed)*100:.1f}%)")

    onset_minutes = parsed["note_onset_minutes"].dropna()
    if not onset_minutes.empty:
        print(f"\nonset_minutes percentiles:")
        for p in (5, 25, 50, 75, 95):
            print(f"  p{p:>2d}: {onset_minutes.quantile(p/100):.0f}")
        print(f"  min: {onset_minutes.min():.0f}, "
              f"max: {onset_minutes.max():.0f}, "
              f"mean: {onset_minutes.mean():.1f}")

    print("\nLocation distribution (1 = present):")
    for col in [c for c in parsed.columns if c.startswith("note_location_")]:
        print(f"  {col:35s}: {int(parsed[col].sum())}")
    print("\nOnset bucket distribution:")
    for col in [c for c in parsed.columns if c.startswith("note_onset_bucket_")]:
        print(f"  {col:35s}: {int(parsed[col].sum())}")

    # Merge into features_triage.csv and features_fourh.csv (idempotent)
    new_cols = [c for c in parsed.columns if c != "encounter_id"]
    for target_name in ("features_triage.csv", "features_fourh.csv"):
        path = DERIVED / target_name
        if not path.exists():
            print(f"\nskip {target_name}: not present")
            continue
        df = pd.read_csv(path)
        # Drop any prior note_* columns we own
        prior = [c for c in df.columns if c in new_cols]
        if prior:
            df = df.drop(columns=prior)
        merged = df.merge(parsed, on="encounter_id", how="left")
        merged.to_csv(path, index=False)
        print(f"\nMerged into {target_name}: {merged.shape}")


if __name__ == "__main__":
    main()
