"""
probs_8: counterfactual / negative-evidence reasoning.

For each encounter, start with a flat prior over the 4 classes
(Kraken=sympathomimetic, Triton=sedative-hypnotic, Coral=hallucinogenic,
None=medical). Then, for each class, scan notes for EXCLUSIONARY evidence
(findings that argue against that class). Apply a multiplicative penalty
proportional to the exclusion count, renormalize, and cap confidence at
max(p_class) <= 0.75 by smoothing with the prior.

Output: derived/probs_8.csv with columns
  encounter_id, p_kraken, p_triton, p_coral, p_none
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NARR = ROOT / "derived" / "narratives.jsonl"
OUT = ROOT / "derived" / "probs_8.csv"

CLASSES = ["kraken", "triton", "coral", "none"]
PRIOR = {c: 0.25 for c in CLASSES}

# Penalty per exclusionary hit. floor at 0.1 so a class is never zeroed.
PENALTY_PER = 0.20
PENALTY_FLOOR = 0.10
CONF_CEILING = 0.75


# ---------------------------------------------------------------------------
# Text feature extraction
# ---------------------------------------------------------------------------

VITAL_RE = {
    "hr": re.compile(r"HR\s*(\d{2,3})"),
    "rr": re.compile(r"RR\s*(\d{1,3})"),
    "sbp": re.compile(r"BP\s*(\d{2,3})\s*/\s*(\d{2,3})"),
    "temp": re.compile(r"Temp\s*([\d.]+)\s*C"),
    "spo2": re.compile(r"SpO2\s*(\d{2,3})"),
    "gcs": re.compile(r"GCS\s*(\d{1,2})"),
}


def parse_vitals(text: str) -> dict:
    v: dict = {}
    if not text:
        return v
    for k, rx in VITAL_RE.items():
        m = rx.search(text)
        if not m:
            continue
        if k == "sbp":
            v["sbp"] = int(m.group(1))
            v["dbp"] = int(m.group(2))
        elif k == "temp":
            v["temp"] = float(m.group(1))
        else:
            v[k] = int(m.group(1))
    return v


# Keyword sets
PERCEPTUAL_KEYS = [
    "halluc", "perceptual", "visual distortion", "time-distortion",
    "time distortion", "depersonal", "derealiz", "kaleidoscop",
    "synesthes", "seeing things", "geometric", "trail",
]
ATAXIA_KEYS = ["ataxi", "unsteady", "wide-based gait", "gait instab"]
SYMPATH_KEYS = [
    "diaphor", "tremor", "agitat", "mydriasis", "dilated pup",
    "hyperthermi", "hypertens", "tachycard", "restless", "sympathetic excess",
]
SEDATIVE_KEYS = [
    "miosis", "pinpoint", "hypoventil", "bradypne", "obtund",
    "somnolen", "stupor", "respiratory depress", "low gcs",
]
MEDICAL_DX_KEYS = [
    "uti", "urinary tract", "pyelonephritis", "appendicitis",
    "cholecystitis", "pancreatitis", "diverticulitis", "c-diff",
    "c. diff", "clostridium difficile", "gastroenteritis",
    "pneumonia", "bronchitis", "asthma exac", "copd exac",
    "cad", "stemi", "nstemi", "acute coronary", "myocardial infarct",
    "pulmonary embol", "stroke", "cva", "ischemic stroke",
    "sprain", "fracture", "laceration", "concussion",
    "dka", "diabetic ketoacid", "hyperglycemic", "hypoglycem",
    "sepsis", "bacteremia", "cellulitis", "abscess",
    "migraine", "tension headache", "vertigo", "bppv",
    "gerd", "peptic ulcer", "ibs", "ibd",
    "ckd", "aki", "renal failure", "rhabdomyolysis",
    "chf exac", "heart failure", "afib", "atrial fibrillation",
]


def has_any(text: str, keys: list[str]) -> bool:
    low = text.lower()
    return any(k in low for k in keys)


def count_any(text: str, keys: list[str]) -> int:
    low = text.lower()
    return sum(1 for k in keys if k in low)


# ---------------------------------------------------------------------------
# Exclusion scoring (counterfactual): count reasons class X is NOT it
# ---------------------------------------------------------------------------

def exclusions_kraken(text: str, vitals: dict, pe: str) -> int:
    """Reasons this is NOT sympathomimetic."""
    n = 0
    hr = vitals.get("hr")
    sbp = vitals.get("sbp")
    temp = vitals.get("temp")
    gcs = vitals.get("gcs")

    # Bradycardia or normal HR without other sympathomimetic signs
    if hr is not None and hr < 90:
        n += 1
    # Hypotension excludes sympathomimetic
    if sbp is not None and sbp < 100:
        n += 1
    # Low GCS/sedation without sympathetic excess
    if gcs is not None and gcs <= 12:
        n += 1
    # Hypothermia / normothermia far from hyperthermia
    if temp is not None and temp < 37.0:
        n += 1
    # No diaphoresis / tremor / agitation / mydriasis on PE
    pe_low = (pe or "").lower()
    sympathetic_pe = any(k in pe_low for k in [
        "diaphor", "tremor", "agitat", "restless", "mydriasis", "dilated",
        "tachycard",
    ])
    if not sympathetic_pe and not has_any(text, SYMPATH_KEYS):
        n += 1
    # Explicit alternative medical Dx in MDM
    if has_any(text, MEDICAL_DX_KEYS):
        n += 1
    return n


def exclusions_triton(text: str, vitals: dict, pe: str) -> int:
    """Reasons this is NOT sedative-hypnotic."""
    n = 0
    hr = vitals.get("hr")
    sbp = vitals.get("sbp")
    rr = vitals.get("rr")
    gcs = vitals.get("gcs")

    # Tachycardia argues against sedative
    if hr is not None and hr > 105:
        n += 1
    # Hypertension argues against sedative
    if sbp is not None and sbp > 140:
        n += 1
    # Tachypnea argues against hypoventilation
    if rr is not None and rr > 22:
        n += 1
    # Normal/high GCS argues against sedative
    if gcs is not None and gcs >= 14:
        n += 1
    # No miosis/hypoventilation/somnolence mentioned
    pe_low = (pe or "").lower()
    sed_pe = any(k in pe_low for k in [
        "miosis", "pinpoint", "somnolen", "obtund", "stupor", "hypoventil",
    ])
    if not sed_pe and not has_any(text, SEDATIVE_KEYS):
        n += 1
    # Alternative medical Dx
    if has_any(text, MEDICAL_DX_KEYS):
        n += 1
    return n


def exclusions_coral(text: str, vitals: dict, pe: str) -> int:
    """Reasons this is NOT hallucinogen."""
    n = 0
    hr = vitals.get("hr")
    sbp = vitals.get("sbp")
    gcs = vitals.get("gcs")

    # No perceptual symptoms and no ataxia
    if not has_any(text, PERCEPTUAL_KEYS) and not has_any(text, ATAXIA_KEYS):
        n += 1
    # Severely abnormal vitals (hallucinogens tend to be mild)
    if hr is not None and hr > 130:
        n += 1
    if sbp is not None and (sbp > 170 or sbp < 90):
        n += 1
    # Severely depressed GCS argues against hallucinogen
    if gcs is not None and gcs <= 10:
        n += 1
    # Heavy sympathomimetic findings (overshadow hallucinogen)
    pe_low = (pe or "").lower()
    heavy_symp = sum(1 for k in [
        "diaphor", "tremor", "agitat", "mydriasis", "tachycard", "restless",
    ] if k in pe_low) >= 3
    if heavy_symp:
        n += 1
    # Alternative medical Dx
    if has_any(text, MEDICAL_DX_KEYS):
        n += 1
    return n


def exclusions_none(text: str, vitals: dict, pe: str) -> int:
    """Reasons this is NOT a 'medical / no drug' presentation.

    Drug-class findings argue against the medical class.
    """
    n = 0
    pe_low = (pe or "").lower()
    # Sympathomimetic PE findings argue against medical
    sympathetic_pe = sum(1 for k in [
        "diaphor", "tremor", "agitat", "restless", "mydriasis",
    ] if k in pe_low)
    if sympathetic_pe >= 2:
        n += 1
    # Sedative PE findings argue against medical
    sed_pe = any(k in pe_low for k in [
        "miosis", "pinpoint", "somnolen", "obtund", "stupor",
    ])
    if sed_pe:
        n += 1
    # Perceptual symptoms argue against medical
    if has_any(text, PERCEPTUAL_KEYS):
        n += 1
    # Hyperthermia argues against routine medical
    temp = vitals.get("temp")
    if temp is not None and temp >= 39.0:
        n += 1
    # Festival context + tox-metabolic working impression
    low = text.lower()
    if "festival" in low and "tox" in low:
        n += 1
    # Explicit "ingestion" / "substance" / "intoxication" wording
    if any(k in low for k in [
        "ingestion", "substance exposure", "intoxication",
        "drug exposure", "festival-related substance",
    ]):
        n += 1
    return n


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

def compute_probs(rec: dict) -> dict:
    parts = [
        rec.get("triage_brief_note", ""),
        rec.get("brief_hpi", ""),
        rec.get("hpi", ""),
        rec.get("mdm", ""),
        rec.get("clinical_course", ""),
        rec.get("chief_complaint", ""),
    ]
    text = "\n".join(p for p in parts if p)
    pe = rec.get("physical_exam_pertinent_positives", "") or ""
    vitals = parse_vitals(text)

    excl = {
        "kraken": exclusions_kraken(text, vitals, pe),
        "triton": exclusions_triton(text, vitals, pe),
        "coral": exclusions_coral(text, vitals, pe),
        "none": exclusions_none(text, vitals, pe),
    }

    # Multiplicative penalty
    p = {c: PRIOR[c] for c in CLASSES}
    if sum(excl.values()) == 0:
        # No exclusionary evidence anywhere — output flat prior
        return p

    for c in CLASSES:
        penalty = max(PENALTY_FLOOR, 1.0 - PENALTY_PER * excl[c])
        p[c] = PRIOR[c] * penalty

    # Renormalize
    s = sum(p.values())
    if s <= 0:
        return {c: 0.25 for c in CLASSES}
    p = {c: v / s for c, v in p.items()}

    # Confidence ceiling: smooth with the flat prior if needed
    mx = max(p.values())
    if mx > CONF_CEILING:
        # Convex blend: p' = a * p + (1-a) * prior, choose a so max <= ceiling
        # max(p') = a * mx + (1 - a) * 0.25 <= 0.75
        # => a <= (0.75 - 0.25) / (mx - 0.25) = 0.5 / (mx - 0.25)
        denom = mx - 0.25
        if denom > 1e-9:
            a = min(1.0, 0.5 / denom)
            p = {c: a * p[c] + (1 - a) * 0.25 for c in CLASSES}
            # Re-renormalize defensively
            s = sum(p.values())
            p = {c: v / s for c, v in p.items()}

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = []
    with NARR.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid = rec["encounter_id"]
            p = compute_probs(rec)
            rows.append(
                {
                    "encounter_id": eid,
                    "p_kraken": p["kraken"],
                    "p_triton": p["triton"],
                    "p_coral": p["coral"],
                    "p_none": p["none"],
                }
            )

    df = pd.DataFrame(rows, columns=[
        "encounter_id", "p_kraken", "p_triton", "p_coral", "p_none",
    ])

    # Sum validation
    sums = df[["p_kraken", "p_triton", "p_coral", "p_none"]].sum(axis=1)
    bad = ((sums - 1.0).abs() > 0.005).sum()
    assert bad == 0, f"{bad} rows have sums outside 1.0 +/- 0.005"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")

    # Summary
    print(f"Path: {OUT}")
    print(f"Rows: {len(df)}")
    print(f"Sum range: [{sums.min():.6f}, {sums.max():.6f}]")
    print("Marginal means:")
    for c in ["p_kraken", "p_triton", "p_coral", "p_none"]:
        print(f"  {c}: {df[c].mean():.4f}")
    max_p = df[["p_kraken", "p_triton", "p_coral", "p_none"]].max(axis=1)
    n_conf = int((max_p > 0.5).sum())
    print(f"Rows with max_prob > 0.5: {n_conf}")


if __name__ == "__main__":
    main()
