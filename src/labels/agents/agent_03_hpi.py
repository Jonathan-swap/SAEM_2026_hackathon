"""Agent 3 — HPI-led toxidrome classifier.

Weight `brief_hpi` and `hpi` heaviest. Physical exam and meds are confirmation only.

Classes:
  - Kraken Candy   → SYMPATHOMIMETIC
  - Triton Tabs    → SEDATIVE-HYPNOTIC
  - Coral Dust     → HALLUCINOGENIC / serotonergic
  - None           → typical medical pathology unrelated to festival drugs
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

NARR = Path(r"C:/Users/rs3te/Work/Claude-safe/SAEM-Hackathon/derived/narratives_fourh.jsonl")
OUT = Path(r"C:/Users/rs3te/Work/Claude-safe/SAEM-Hackathon/derived/probs_3.csv")

# --- lexicons ---------------------------------------------------------------
# Each tuple: (regex pattern, weight). Patterns matched on HPI text (case-insensitive).

FESTIVAL_CONTEXT = [
    (r"festival", 1.0),
    (r"main stage", 1.0),
    (r"dance tent", 1.0),
    (r"campground", 1.0),
    (r"medical tent", 1.0),
    (r"festival.{0,20}substance", 1.5),
    (r"possible.{0,20}substance exposure", 1.2),
    (r"unknown.{0,20}substance", 1.0),
    (r"pill|tab|powder|edible|gummy", 0.8),
    (r"rave|edm|concert", 0.8),
    (r"crowd", 0.4),
    (r"heat exposure|hot weather|ambient heat", 0.4),
]

# Sympathomimetic (Kraken): tachycardia, HTN, hyperthermia, diaphoresis, agitation, tremor, mydriasis
KRAKEN_HPI = [
    (r"agitat", 1.2),
    (r"restless", 1.0),
    (r"tremor|shaking", 1.0),
    (r"diaphore|sweat|sweating profusely", 1.0),
    (r"rapid heartbeat|racing heart|pounding heart|palpitation", 1.2),
    (r"tachycard", 1.0),
    (r"hyperther|overheat|feels hot|body.{0,5}hot|elevated temp", 1.2),
    (r"hypertens|elevated blood pressure", 0.8),
    (r"mydria|dilated pupil", 1.3),
    (r"clench(ed)? jaw|teeth grinding|bruxism|jaw tension", 1.2),
    (r"euphori|stimulant|energi[sz]ed|wired", 1.0),
    (r"dancing continuously|hyperactive|nonstop motion", 0.9),
    (r"agitation/sympathetic excess", 1.0),  # MDM hint when present in HPI text
    (r"sympathomimetic", 2.0),
    (r"chest pain.{0,20}stimulant", 1.2),
    (r"anxious|anxiety/panic", 0.4),
]

# Sedative-hypnotic (Triton): brady, hypoTN, hypovent, low GCS, miosis, slow responses
TRITON_HPI = [
    (r"unresponsive|obtund|stupor|barely arousable|not responsive", 1.5),
    (r"sleepy|somnolen|drowsy|lethargic|sedat", 1.3),
    (r"slow(ed)? response|delayed response|slow speech", 1.1),
    (r"slurred speech", 1.0),
    (r"shallow breath|slow breath|hypovent|respiratory depression|low resp rate", 1.5),
    (r"snoring respir|stertor|gurgling", 1.2),
    (r"pinpoint pupil|miosis|constricted pupil", 1.6),
    (r"bradycard|slow heart rate", 1.0),
    (r"hypoten|low blood pressure", 0.9),
    (r"depressed mental status|altered mental status.{0,20}depress", 1.2),
    (r"low GCS|GCS .{0,5}[3-9]\b", 0.8),
    (r"cyanos|blue lips|hypoxi", 0.7),
    (r"found down|found unresponsive", 1.3),
    (r"naloxone|narcan", 2.0),
    (r"opioid|benzodiazepine ingestion|benzo ingestion", 1.5),
]

# Hallucinogenic / serotonergic (Coral): perceptual changes, mild tachy, ataxia, unsteady, hyperreflexia
CORAL_HPI = [
    (r"hallucinat|visual distort|seeing things|seeing pattern|visual trail|trail(ing|s)", 1.6),
    (r"perceptual (change|distort)|altered perception", 1.5),
    (r"time.{0,5}distort|time dilation|time slow", 1.4),
    (r"derealiz|depersonaliz", 1.3),
    (r"dissociat", 1.0),
    (r"synesthe", 1.4),
    (r"ataxi|unsteady|wide.based gait|staggering|wobbly", 1.2),
    (r"feeling unsteady", 1.1),
    (r"hyperreflex|clonus|myoclon", 1.4),
    (r"serotonergic|serotonin syndrome", 2.0),
    (r"ringing in ears|tinnitus", 0.4),
    (r"paranoi|bizarre thought|magical thinking", 0.7),
    (r"euphoric.{0,20}connect|spiritual|profound", 0.7),
    (r"mild(ly)? tachycard", 0.5),
    (r"mydria", 0.5),  # also present in hallucinogenics
    (r"emotional lability", 0.6),
    (r"intrusive thought|kaleidoscop", 1.3),
]

# Negative / "None" indicators in HPI: explicit medical pathology, denies festival exposure, etc.
NONE_HPI = [
    (r"denies (any )?(drug|substance|festival|recreational)", 1.5),
    (r"no (known )?(drug|substance|festival|recreational)", 1.0),
    (r"diabet|hyperglycemi|hypoglycemi|DKA|ketoacid", 1.5),
    (r"chest pain.{0,40}(STEMI|MI|cardiac|exertion)", 1.2),
    (r"pneumon|COPD exacerb|asthma exacerb", 1.4),
    (r"sepsis|septic shock|bacteremia|UTI", 1.3),
    (r"appendicit|cholecyst|pancreatit|diverticul", 1.5),
    (r"stroke|CVA|TIA|hemorrhag", 1.5),
    (r"trauma|MVC|fall from|assault|fracture", 1.3),
    (r"PE|pulmonary embolism|DVT", 1.4),
    (r"GI bleed|melena|hematemesis", 1.4),
    (r"renal failure|AKI|dialysis", 1.2),
    (r"c.diff|c\. ?difficile|gastroenter", 1.2),
    (r"cellulitis|abscess|wound infection", 1.1),
    (r"migraine|cluster headache", 0.9),
    (r"pregnancy|ectopic|miscarriage", 1.3),
    (r"psychiatric history|known psych|baseline psych|chronic anxiety|chronic depression", 0.6),
    (r"medication non.?adheren|missed (his|her|their) (med|dose)", 1.0),
    (r"home medication|chronic medication", 0.5),
]


def score_text(text: str, lexicon: list[tuple[str, float]]) -> float:
    if not text:
        return 0.0
    total = 0.0
    for pat, w in lexicon:
        if re.search(pat, text, re.IGNORECASE):
            total += w
    return total


def classify(rec: dict) -> tuple[float, float, float, float]:
    brief_hpi = rec.get("brief_hpi") or ""
    hpi = rec.get("hpi") or ""
    triage = rec.get("triage_brief_note") or ""
    moa = rec.get("mode_of_arrival") or ""
    # HPI-led: combine brief_hpi (weight 1.5) + hpi (1.0) + triage (0.5)
    # Physical/meds used only as confirmation/contradiction (low weight).
    pe = rec.get("physical_exam_pertinent_positives") or ""
    meds = rec.get("ed_meds_procedures") or ""
    mdm = rec.get("mdm") or ""

    # primary HPI signal
    hpi_text_primary = " ".join([brief_hpi, brief_hpi, hpi, triage, moa])  # double brief_hpi
    # confirmation channel (low weight)
    confirm_text = " ".join([pe, meds, mdm])

    festival = score_text(hpi_text_primary, FESTIVAL_CONTEXT)
    kraken = score_text(hpi_text_primary, KRAKEN_HPI) + 0.3 * score_text(confirm_text, KRAKEN_HPI)
    triton = score_text(hpi_text_primary, TRITON_HPI) + 0.3 * score_text(confirm_text, TRITON_HPI)
    coral = score_text(hpi_text_primary, CORAL_HPI) + 0.3 * score_text(confirm_text, CORAL_HPI)
    none = score_text(hpi_text_primary, NONE_HPI) + 0.2 * score_text(confirm_text, NONE_HPI)

    # Festival context boosts drug classes (HPI emphasis on exposure story).
    drug_boost = 1.0 + min(festival * 0.25, 1.5)
    kraken *= drug_boost
    triton *= drug_boost
    coral *= drug_boost

    # If absolutely no festival context AND strong NONE indicators, push p_none up.
    no_festival = festival < 0.5
    if no_festival:
        # baseline lift for None
        none += 1.5

    # Convert to probabilities via softmax-style with temperature.
    # If all scores ~ 0, return uniform.
    raw = [kraken, triton, coral, none]
    if sum(raw) < 0.2:
        return (0.25, 0.25, 0.25, 0.25)

    # Add small baseline so weak signals still produce some mass.
    baseline = 0.5
    raw = [r + baseline for r in raw]

    # Soft normalization using exponentiation with temperature ~1.4
    T = 1.4
    exps = [math.exp(r / T) for r in raw]
    s = sum(exps)
    probs = [e / s for e in exps]

    # If a drug-class probability dominates but festival context is missing, dampen it.
    if no_festival and max(probs[:3]) > probs[3]:
        # blend toward more None weight
        damp = 0.55
        probs[3] = probs[3] + (1 - damp) * (probs[0] + probs[1] + probs[2])
        probs[0] *= damp
        probs[1] *= damp
        probs[2] *= damp
        s = sum(probs)
        probs = [p / s for p in probs]

    # Non-festival medical: enforce p_none >= 0.7 when NONE score strongly dominates and no fest.
    if no_festival and (none > (kraken + triton + coral) * 1.3):
        if probs[3] < 0.7:
            deficit = 0.7 - probs[3]
            probs[3] = 0.7
            drug_sum = probs[0] + probs[1] + probs[2]
            if drug_sum > 0:
                factor = (1 - 0.7) / drug_sum
                probs[0] *= factor
                probs[1] *= factor
                probs[2] *= factor

    return tuple(probs)


def round_and_fix(p: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Round to 3 decimals and ensure sum to 1.0 by adjusting the largest component."""
    rounded = [round(x, 3) for x in p]
    diff = round(1.0 - sum(rounded), 3)
    if abs(diff) >= 0.001:
        idx = max(range(4), key=lambda i: rounded[i])
        rounded[idx] = round(rounded[idx] + diff, 3)
    return tuple(rounded)


def main() -> None:
    records = []
    with NARR.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    out_rows = []
    for rec in records:
        eid = rec.get("encounter_id")
        probs = classify(rec)
        probs = round_and_fix(probs)
        out_rows.append((eid, *probs))

    # Validate sums
    sum_ok = 0
    for row in out_rows:
        s = sum(row[1:])
        if abs(s - 1.0) <= 0.005:
            sum_ok += 1

    # Marginal means + p_none > 0.5 count
    n = len(out_rows)
    mk = sum(r[1] for r in out_rows) / n
    mt = sum(r[2] for r in out_rows) / n
    mc = sum(r[3] for r in out_rows) / n
    mn = sum(r[4] for r in out_rows) / n
    pn_high = sum(1 for r in out_rows if r[4] > 0.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encounter_id", "p_kraken", "p_triton", "p_coral", "p_none"])
        for row in out_rows:
            w.writerow(row)

    print(f"path={OUT}")
    print(f"records={n}")
    print(f"sum_ok={sum_ok}/{n}")
    print(f"mean_p_kraken={mk:.4f}")
    print(f"mean_p_triton={mt:.4f}")
    print(f"mean_p_coral={mc:.4f}")
    print(f"mean_p_none={mn:.4f}")
    print(f"p_none>0.5 count={pn_high}")


if __name__ == "__main__":
    main()
