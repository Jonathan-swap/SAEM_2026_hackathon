"""Build probs_7.csv via classical toxidrome decision tree.

Walks the textbook tox decision tree (mental status, autonomic vitals,
pupils, motor/coordination) per record. Each branch a record matches
contributes additive evidence to one or more classes. Final probabilities
are proportional to total tree-branch evidence per class.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[3] / "derived"
SRC = HERE / "narratives_fourh.jsonl"
OUT = HERE / "probs_7.csv"

CLASSES = ("kraken", "triton", "coral", "none")


def parse_vitals(text: str) -> dict[str, float | None]:
    """Extract HR, RR, BP_sys, BP_dia, Temp, SpO2, GCS from a note."""
    v: dict[str, float | None] = dict(hr=None, rr=None, sbp=None, dbp=None,
                                      temp=None, spo2=None, gcs=None)
    m = re.search(r"\bHR\s+(\d+)", text)
    if m:
        v["hr"] = float(m.group(1))
    m = re.search(r"\bRR\s+(\d+)", text)
    if m:
        v["rr"] = float(m.group(1))
    m = re.search(r"\bBP\s+(\d+)\s*/\s*(\d+)", text)
    if m:
        v["sbp"] = float(m.group(1))
        v["dbp"] = float(m.group(2))
    m = re.search(r"Temp\s+([\d.]+)\s*C", text)
    if m:
        v["temp"] = float(m.group(1))
    m = re.search(r"SpO2\s+(\d+)\s*%", text)
    if m:
        v["spo2"] = float(m.group(1))
    m = re.search(r"GCS\s+(\d+)", text)
    if m:
        v["gcs"] = float(m.group(1))
    return v


def has_any(text: str, patterns: list[str]) -> bool:
    """Case-insensitive whole-phrase/regex presence test."""
    low = text.lower()
    return any(p in low for p in patterns)


# Phrase banks (lowercase) -----------------------------------------------------
AGITATED = [
    "agitat", "restless", "anxio", "panic", "marked restlessness",
    "combative", "hyperactive", "wired",
]
SEDATED = [
    "sedat", "lethargic", "somnolen", "drowsy", "obtund", "stupor",
    "unresponsive", "slow response", "slowed response", "hypoventilat",
    "depressed mental",
]
HALLUCINATING = [
    "hallucin", "perceptual", "time-distortion", "time distortion",
    "visual distortion", "visual trail", "synesth", "depersonali",
    "derealiz", "kaleido", "geometric pattern", "out of body",
]
CONFUSED_GENERIC = [
    "confus", "disorient", "altered mental",
]

DIAPHORETIC = ["diaphore", "sweat", "perspir"]
DRY_MUCOSA = ["dry mouth", "dry mucos", "dryness symptom", "mucous membranes dry"]
COOL_DRY = ["cool skin", "cool extrem", "clammy"]
TREMOR = ["tremor", "shaking", "trembl"]
CLONUS = ["clonus", "hyperreflex"]
ATAXIA = ["ataxia", "unstead", "wobbl", "wide-based gait", "wide based gait"]
FLACCID = ["flaccid", "hypotoni", "hyporeflex", "areflex", "diminished reflex"]

MYDRIASIS = ["mydriasis", "dilated pup", "pupils dilated", "large pupil"]
MIOSIS = ["miosis", "pinpoint", "constricted pup", "small pupil"]

HALLUCINOGEN_HINTS = [
    "ringing in ears", "tinnitus", "visual trail", "kaleido", "synesth",
    "time-distortion", "time distortion", "depersonali", "derealiz",
    "perceptual",
]


def tree_evidence(rec: dict) -> dict[str, float]:
    """Walk the toxidrome decision tree; return additive evidence per class."""
    parts = []
    for key in ("triage_brief_note", "brief_hpi", "hpi",
                "physical_exam_pertinent_positives", "mdm",
                "clinical_course", "chief_complaint"):
        val = rec.get(key)
        if val:
            parts.append(str(val))
    note = " \n ".join(parts)
    note_low = note.lower()

    v = parse_vitals(note)
    ev = {c: 0.0 for c in CLASSES}

    # ----- Branch 1: Mental status ------------------------------------------
    is_agit = has_any(note_low, AGITATED)
    is_sed = has_any(note_low, SEDATED) or (v["gcs"] is not None and v["gcs"] <= 12)
    is_hall = has_any(note_low, HALLUCINATING)
    is_conf_only = has_any(note_low, CONFUSED_GENERIC) and not (is_agit or is_sed or is_hall)

    if is_agit:
        # split between sympathomimetic and hallucinogen
        ev["kraken"] += 1.0
        ev["coral"] += 0.5
    if is_sed:
        ev["triton"] += 1.5
    if is_hall:
        ev["coral"] += 1.5
    if is_conf_only:
        ev["none"] += 1.0

    # ----- Branch 2: Autonomic / vitals -------------------------------------
    hr, sbp, temp, rr = v["hr"], v["sbp"], v["temp"], v["rr"]

    # Sympathomimetic pattern: HR up + BP up + Temp up + diaphoretic
    sym_score = 0.0
    if hr is not None and hr >= 110:
        sym_score += 0.4
    if hr is not None and hr >= 130:
        sym_score += 0.3
    if sbp is not None and sbp >= 150:
        sym_score += 0.4
    if temp is not None and temp >= 38.5:
        sym_score += 0.5
    if temp is not None and temp >= 39.5:
        sym_score += 0.4
    if has_any(note_low, DIAPHORETIC):
        sym_score += 0.4
    ev["kraken"] += sym_score

    # Sedative pattern: HR down + BP down + RR down + cool/dry
    sed_score = 0.0
    if hr is not None and hr <= 60:
        sed_score += 0.6
    if hr is not None and hr <= 50:
        sed_score += 0.4
    if sbp is not None and sbp <= 95:
        sed_score += 0.4
    if sbp is not None and sbp <= 85:
        sed_score += 0.4
    if rr is not None and rr <= 10:
        sed_score += 0.7
    if rr is not None and rr <= 8:
        sed_score += 0.4
    if v["spo2"] is not None and v["spo2"] <= 92 and (rr is not None and rr <= 14):
        sed_score += 0.3
    if has_any(note_low, COOL_DRY):
        sed_score += 0.4
    if v["gcs"] is not None and v["gcs"] <= 10:
        sed_score += 0.6
    ev["triton"] += sed_score

    # Hallucinogen pattern: HR mildly up + BP normal + temp normal + dry mucosa
    hall_score = 0.0
    hr_mild_up = (hr is not None and 95 <= hr < 120)
    bp_normal = (sbp is None) or (90 <= sbp <= 145)
    temp_normal = (temp is None) or (36.0 <= temp <= 38.2)
    if hr_mild_up:
        hall_score += 0.4
    if bp_normal and temp_normal and hr_mild_up:
        hall_score += 0.3
    if has_any(note_low, DRY_MUCOSA):
        hall_score += 0.4
    if has_any(note_low, HALLUCINOGEN_HINTS):
        hall_score += 0.5
    ev["coral"] += hall_score

    # Vitals stable + focal complaint (none of the above strongly) → None
    vitals_present = sum(1 for x in (hr, sbp, temp, rr) if x is not None)
    if vitals_present >= 3:
        stable = (
            (hr is None or 60 <= hr <= 100)
            and (sbp is None or 100 <= sbp <= 140)
            and (temp is None or 36.2 <= temp <= 37.8)
            and (rr is None or 12 <= rr <= 20)
        )
        if stable and not (is_agit or is_sed or is_hall):
            ev["none"] += 1.2
        elif stable:
            ev["none"] += 0.4

    # ----- Branch 3: Pupils --------------------------------------------------
    if has_any(note_low, MYDRIASIS):
        ev["kraken"] += 0.7
        ev["coral"] += 0.5
    if has_any(note_low, MIOSIS):
        ev["triton"] += 1.0

    # ----- Branch 4: Motor / coordination -----------------------------------
    if has_any(note_low, TREMOR):
        ev["kraken"] += 0.5
        ev["coral"] += 0.2  # serotonergic overlap
    if has_any(note_low, CLONUS):
        ev["kraken"] += 0.4
        ev["coral"] += 0.3
    if has_any(note_low, ATAXIA):
        ev["coral"] += 0.7
    if has_any(note_low, FLACCID):
        ev["triton"] += 0.7

    # ----- Treatment-plan hints (very mild prior nudge) ---------------------
    if "naloxone" in note_low or "reversal of sedation" in note_low:
        ev["triton"] += 0.6
    if "benzodiazepine for agitation" in note_low or "sympathetic excess" in note_low:
        ev["kraken"] += 0.3
        ev["coral"] += 0.1
    if "antipyretic" in note_low and temp is not None and temp >= 38.5:
        ev["kraken"] += 0.2

    return ev


def evidence_to_probs(ev: dict[str, float]) -> dict[str, float]:
    """Convert additive evidence to a probability vector with a mild baseline."""
    # baseline so no class is ever exactly zero
    base = 0.15
    scores = {c: ev[c] + base for c in CLASSES}
    # If almost no evidence anywhere, lean toward None (true medical pathology)
    total_ev = sum(ev.values())
    if total_ev < 0.4:
        scores["none"] += 1.0
    s = sum(scores.values())
    return {c: scores[c] / s for c in CLASSES}


def main() -> None:
    rows: list[tuple[str, float, float, float, float]] = []
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid = rec["encounter_id"]
            ev = tree_evidence(rec)
            p = evidence_to_probs(ev)
            rows.append((eid, p["kraken"], p["triton"], p["coral"], p["none"]))

    with OUT.open("w", encoding="utf-8", newline="") as f:
        f.write("encounter_id,p_kraken,p_triton,p_coral,p_none\n")
        for eid, pk, pt, pc, pn in rows:
            f.write(f"{eid},{pk:.6f},{pt:.6f},{pc:.6f},{pn:.6f}\n")

    # Sanity report ---------------------------------------------------------
    n = len(rows)
    sums = [pk + pt + pc + pn for _, pk, pt, pc, pn in rows]
    bad = [s for s in sums if abs(s - 1.0) > 0.005]
    mk = sum(r[1] for r in rows) / n
    mt = sum(r[2] for r in rows) / n
    mc = sum(r[3] for r in rows) / n
    mn = sum(r[4] for r in rows) / n
    none_dom = sum(1 for r in rows if r[4] > 0.5)
    print(f"path={OUT}")
    print(f"rows={n}")
    print(f"rows_with_bad_sum={len(bad)}  min_sum={min(sums):.6f}  max_sum={max(sums):.6f}")
    print(f"marginal_means kraken={mk:.4f} triton={mt:.4f} coral={mc:.4f} none={mn:.4f}")
    print(f"count_p_none_gt_0.5={none_dom}")


if __name__ == "__main__":
    main()
