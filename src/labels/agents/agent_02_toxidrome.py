"""Agent 2 — toxidrome-led probability estimator.

Weighting:
- physical_exam_pertinent_positives: heaviest
- ed_meds_procedures: heaviest
- hpi / mdm / clinical_course: tie-breakers only

Classes: kraken (sympathomimetic), triton (sedative-hypnotic),
coral (hallucinogen/serotonergic), none (non-festival pathology).
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

IN = Path(r"C:\Users\rs3te\Work\Claude-safe\SAEM-Hackathon\derived\narratives_fourh.jsonl")
OUT = Path(r"C:\Users\rs3te\Work\Claude-safe\SAEM-Hackathon\derived\probs_2.csv")


# Physical exam tokens — primary signal
EXAM_KRAKEN = {
    "diaphoretic": 3.0,
    "tachycardic": 2.2,
    "hypertensive": 2.5,
    "hyperthermic": 2.8,
    "agitated": 2.6,
    "tremor": 1.8,
    "mild_tremor": 1.5,
    "moderate_tremor": 2.0,
    "mydriasis": 2.6,
    "dilated_pupils": 2.6,
    "restless": 1.6,
    "hyperreflexia": 1.0,  # overlap with coral; weaker for kraken
    "flushed": 1.2,
    "tachypneic": 1.0,
    "clenched_jaw": 1.6,
    "bruxism": 1.6,
    "hyperactive": 1.8,
    "irritable": 0.8,
    "dry_mucosa": 0.4,  # mild stimulant overlap
}

EXAM_TRITON = {
    "bradycardic": 3.0,
    "hypotensive": 2.5,
    "hypoventilation": 3.0,
    "bradypneic": 2.8,
    "low_gcs": 3.0,
    "depressed_loc": 3.0,
    "miosis": 2.8,
    "pinpoint_pupils": 2.8,
    "slow_responses": 2.4,
    "somnolent": 2.6,
    "lethargic": 2.2,
    "obtunded": 3.0,
    "drowsy": 1.8,
    "shallow_respirations": 2.4,
    "decreased_responsiveness": 2.4,
    "snoring_respirations": 2.0,
    "hyporeflexia": 1.6,
    "flaccid": 1.5,
    "ataxia": 0.4,  # overlap with coral; weak triton
    "reduced_tracking": 1.6,
    "unresponsive": 2.6,
}

EXAM_CORAL = {
    "ataxia": 2.6,
    "unsteady_gait": 2.6,
    "wide_based_gait": 2.4,
    "hyperreflexia": 2.4,
    "clonus": 2.4,
    "myoclonus": 2.2,
    "nystagmus": 2.0,
    "perceptual_disturbance": 2.8,
    "visual_hallucinations": 2.8,
    "auditory_hallucinations": 2.6,
    "disoriented": 1.4,
    "mild_tremor": 0.6,  # overlap weak
    "tremor": 0.5,
    "reduced_tracking": 0.8,
    "diaphoretic": 0.6,  # mild — serotonergic overlap
    "tachycardic": 0.6,
}

EXAM_NONE = {
    "fatigued_appearance": 0.6,
    "pale": 0.6,
    "icteric": 1.2,
    "abdominal_tenderness": 1.4,
    "focal_weakness": 1.4,
    "rash": 1.0,
    "wheezing": 1.2,
    "rhonchi": 1.0,
    "crackles": 1.0,
    "murmur": 1.0,
    "edema": 0.8,
    "guarding": 1.2,
    "rebound": 1.2,
    "cva_tenderness": 1.4,
}


# Treatments — primary signal
TX_KRAKEN_PHRASES = [
    ("benzodiazepine for agitation", 2.4),
    ("benzodiazepine for agitation/sympathetic excess", 2.8),
    ("antipyretic", 1.6),
    ("cooling", 2.0),
    ("active cooling", 2.2),
    ("evaporative cooling", 2.2),
    ("ice pack", 1.6),
    ("iv crystalloid bolus", 0.6),  # generic but supports stim
]

TX_TRITON_PHRASES = [
    ("intubation", 3.0),
    ("endotracheal intubation", 3.0),
    ("bag-valve-mask", 2.4),
    ("bvm", 2.4),
    ("naloxone", 3.2),
    ("flumazenil", 3.2),
    ("reversal agent", 2.8),
    ("airway support", 2.4),
    ("noninvasive positive pressure", 2.4),
    ("nippv", 2.4),
    ("cpap", 2.0),
    ("bipap", 2.2),
    ("supplemental oxygen via nrb", 1.4),
    ("oral airway", 2.0),
    ("nasopharyngeal airway", 2.0),
]

TX_CORAL_PHRASES = [
    ("supportive care", 1.4),
    ("reassurance", 1.6),
    ("low-stimulation", 1.6),
    ("quiet environment", 1.6),
    ("cyproheptadine", 3.0),
    ("benzodiazepine if agitated", 1.4),
    ("oral hydration", 0.8),
]

TX_NONE_PHRASES = [
    ("antibiotic", 2.0),
    ("ceftriaxone", 2.4),
    ("vancomycin", 2.4),
    ("piperacillin", 2.4),
    ("foley", 1.2),
    ("ct ", 1.0),
    ("ultrasound", 0.8),
    ("ekg", 0.4),
    ("troponin", 1.4),
    ("blood cultures", 1.6),
    ("nitroglycerin", 1.6),
    ("aspirin", 1.0),
    ("heparin", 1.6),
    ("insulin", 1.6),
    ("d50", 1.4),
    ("anticoagulation", 1.4),
    ("antiemetic", 0.4),
]


# Tie-breaker tokens (HPI/MDM/clinical_course) — small weights
HPI_KRAKEN = [
    ("sympathetic excess", 0.8),
    ("hyperthermia", 0.6),
    ("hyperadrenergic", 0.8),
    ("stimulant", 0.8),
    ("severe agitation", 0.6),
    ("psychomotor agitation", 0.6),
]

HPI_TRITON = [
    ("respiratory depression", 1.0),
    ("hypoventilation", 0.8),
    ("decreased level of consciousness", 0.8),
    ("airway compromise", 0.8),
    ("sedative", 0.6),
    ("opioid-like", 0.8),
    ("low gcs", 0.8),
]

HPI_CORAL = [
    ("perceptual", 0.6),
    ("hallucinog", 0.6),
    ("serotonergic", 0.8),
    ("time-distortion", 0.6),
    ("visual distortion", 0.6),
    ("dissociative", 0.6),
    ("derealization", 0.6),
]

HPI_NONE = [
    ("post-traumatic", 0.6),
    ("c-diff", 1.4),
    ("sepsis", 1.4),
    ("dka", 1.6),
    ("stemi", 1.6),
    ("nstemi", 1.4),
    ("pulmonary embolism", 1.6),
    ("appendicitis", 1.6),
    ("pyelonephritis", 1.4),
    ("uti", 1.0),
    ("pneumonia", 1.4),
    ("stroke", 1.4),
    ("seizure disorder", 1.0),
    ("cholecystitis", 1.4),
    ("pregnancy", 0.8),
    ("trauma", 0.6),
    ("foreign body", 0.8),
    ("anaphylaxis", 1.4),
    ("not consistent with festival", 1.8),
    ("unrelated to festival", 1.8),
]


def score_exam(tokens: list[str]) -> tuple[float, float, float, float]:
    sk = st = sc = sn = 0.0
    for tok in tokens:
        sk += EXAM_KRAKEN.get(tok, 0.0)
        st += EXAM_TRITON.get(tok, 0.0)
        sc += EXAM_CORAL.get(tok, 0.0)
        sn += EXAM_NONE.get(tok, 0.0)
    return sk, st, sc, sn


def score_phrases(text: str, phrases: list[tuple[str, float]]) -> float:
    total = 0.0
    t = text.lower()
    for p, w in phrases:
        if p in t:
            total += w
    return total


def festival_signal(rec: dict) -> float:
    """Higher = more festival-like context (lowers p_none baseline)."""
    s = 0.0
    text = " ".join(
        str(rec.get(k, "") or "")
        for k in ("triage_brief_note", "brief_hpi", "hpi", "mdm", "clinical_course", "mode_of_arrival")
    ).lower()
    if "festival" in text:
        s += 2.0
    if "main stage" in text or "side stage" in text or "dance tent" in text:
        s += 1.2
    if "medical tent" in text:
        s += 1.2
    if "rave" in text or "concert" in text:
        s += 1.0
    if "festival-related" in text:
        s += 1.2
    if "tox-metabolic" in text:
        s += 0.6
    if "substance exposure" in text:
        s += 1.0
    if "ingestion" in text:
        s += 0.4
    return s


def score_record(rec: dict) -> tuple[float, float, float, float]:
    exam = (rec.get("physical_exam_pertinent_positives") or "").strip()
    exam_tokens = [t.strip().lower() for t in re.split(r"[;,]", exam) if t.strip()]

    meds = (rec.get("ed_meds_procedures") or "").lower()
    mdm = (rec.get("mdm") or "").lower()
    course = (rec.get("clinical_course") or "").lower()
    hpi = (rec.get("hpi") or "").lower()
    brief = (rec.get("brief_hpi") or "").lower()

    # Primary: exam (weight 1.0) + treatments (weight 1.0)
    ek, et, ec, en = score_exam(exam_tokens)
    tk = score_phrases(meds, TX_KRAKEN_PHRASES) + 0.4 * score_phrases(mdm + " " + course, TX_KRAKEN_PHRASES)
    tt = score_phrases(meds, TX_TRITON_PHRASES) + 0.4 * score_phrases(mdm + " " + course, TX_TRITON_PHRASES)
    tc = score_phrases(meds, TX_CORAL_PHRASES) + 0.4 * score_phrases(mdm + " " + course, TX_CORAL_PHRASES)
    tn = score_phrases(meds, TX_NONE_PHRASES) + 0.4 * score_phrases(mdm + " " + course, TX_NONE_PHRASES)

    # Tie-breakers: HPI/MDM (weight 0.5)
    hk = 0.5 * score_phrases(hpi + " " + brief + " " + mdm, HPI_KRAKEN)
    ht = 0.5 * score_phrases(hpi + " " + brief + " " + mdm, HPI_TRITON)
    hc = 0.5 * score_phrases(hpi + " " + brief + " " + mdm, HPI_CORAL)
    hn = 0.5 * score_phrases(hpi + " " + brief + " " + mdm + " " + course, HPI_NONE)

    # Combine — exam and tx are heaviest (full weight 1.0 each)
    sk = ek + tk + hk
    st = et + tt + ht
    sc = ec + tc + hc
    sn = en + tn + hn

    # Festival context biases AWAY from "none"
    fest = festival_signal(rec)
    # Festival baseline lowers none, raises drug classes a touch
    none_penalty = 0.6 * fest
    drug_boost = 0.15 * fest

    sk += drug_boost
    st += drug_boost
    sc += drug_boost
    sn = max(0.0, sn - none_penalty)

    # Default baseline so empty records lean uncertain
    baseline = 0.6
    sk += baseline
    st += baseline
    sc += baseline
    sn += baseline + 0.4  # mild prior toward none for any non-festival ambiguous case

    return sk, st, sc, sn


def softmax(scores: tuple[float, float, float, float], tau: float = 1.6) -> tuple[float, float, float, float]:
    m = max(scores)
    exps = [math.exp((s - m) / tau) for s in scores]
    z = sum(exps)
    return tuple(e / z for e in exps)  # type: ignore[return-value]


def round_and_renorm(probs: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    r = [round(p, 3) for p in probs]
    s = sum(r)
    diff = round(1.0 - s, 3)
    if abs(diff) >= 0.0005:
        # Adjust the largest probability by the diff
        idx = max(range(4), key=lambda i: r[i])
        r[idx] = round(r[idx] + diff, 3)
    return tuple(r)  # type: ignore[return-value]


def main() -> None:
    rows: list[dict] = []
    with IN.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    results: list[tuple[str, float, float, float, float]] = []
    for rec in rows:
        scores = score_record(rec)
        probs = softmax(scores)
        probs = round_and_renorm(probs)
        results.append((rec["encounter_id"], *probs))

    # Validation
    bad = 0
    sums = []
    pk_sum = pt_sum = pc_sum = pn_sum = 0.0
    pnone_gt_half = 0
    for eid, pk, pt, pc, pn in results:
        s = pk + pt + pc + pn
        sums.append(s)
        if abs(s - 1.0) > 0.005:
            bad += 1
        pk_sum += pk
        pt_sum += pt
        pc_sum += pc
        pn_sum += pn
        if pn > 0.5:
            pnone_gt_half += 1

    n = len(results)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encounter_id", "p_kraken", "p_triton", "p_coral", "p_none"])
        for row in results:
            w.writerow(row)

    print(f"path: {OUT}")
    print(f"record_count: {n}")
    print(f"sum_valid_count: {n - bad} / {n}")
    print(f"sum_invalid_count: {bad}")
    print(f"mean p_kraken: {pk_sum / n:.4f}")
    print(f"mean p_triton: {pt_sum / n:.4f}")
    print(f"mean p_coral:  {pc_sum / n:.4f}")
    print(f"mean p_none:   {pn_sum / n:.4f}")
    print(f"p_none > 0.5 count: {pnone_gt_half}")


if __name__ == "__main__":
    main()
