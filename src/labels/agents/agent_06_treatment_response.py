"""Classify 261 ED encounters into 4 classes using treatment-response trajectory.

Independent hypothesis: treatment response is more diagnostic than initial
presentation. We score based on what was given, whether it worked, and whether
the patient escalated.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

INPUT = Path("C:/Users/rs3te/Work/Claude-safe/SAEM-Hackathon/derived/narratives.jsonl")
OUTPUT = Path("C:/Users/rs3te/Work/Claude-safe/SAEM-Hackathon/derived/probs_6.csv")

CLASSES = ["p_kraken", "p_triton", "p_coral", "p_none"]


def _blob(record: dict) -> str:
    parts = [
        record.get("ed_meds_procedures", ""),
        record.get("clinical_course", ""),
        record.get("mdm", ""),
        record.get("disposition", ""),
        record.get("physical_exam_pertinent_positives", ""),
        record.get("hpi", ""),
        record.get("brief_hpi", ""),
        record.get("triage_brief_note", ""),
        record.get("chief_complaint", ""),
    ]
    return " \n ".join(str(p) for p in parts).lower()


def _has(text: str, *needles: str) -> bool:
    return any(n in text for n in needles)


def score(record: dict) -> dict[str, float]:
    """Return raw scores per class, before softmax."""
    meds = (record.get("ed_meds_procedures") or "").lower()
    course = (record.get("clinical_course") or "").lower()
    mdm = (record.get("mdm") or "").lower()
    disp = (record.get("disposition") or "").lower()
    exam = (record.get("physical_exam_pertinent_positives") or "").lower()
    hpi = (record.get("hpi") or "").lower()
    cc = (record.get("chief_complaint") or "").lower()
    text = _blob(record)

    s = {c: 0.0 for c in CLASSES}

    # ---- Treatment-response trajectory signals ----

    benzo_given = _has(meds, "benzodiazepine", "benzo", "lorazepam", "midazolam", "diazepam")
    antipyretic_given = _has(meds, "antipyretic", "acetaminophen", "cooling")
    ivf_given = _has(meds, "iv crystalloid", "ivf", "fluid bolus", "crystalloid")
    nippv_given = _has(meds, "noninvasive positive pressure", "nippv", "bipap", "cpap")
    intubation_given = _has(meds, "intubation", "endotracheal", "rapid sequence", "rsi")
    reversal_given = _has(meds, "naloxone", "flumazenil", "narcan", "reversal")
    antibiotics_given = _has(meds, "antibiotic", "ceftriaxone", "vancomycin", "azithromycin",
                              "piperacillin", "amoxicillin")
    nsaid_given = _has(meds, "nsaid", "ibuprofen", "ketorolac", "toradol")
    steroids_given = _has(meds, "steroid", "methylprednisolone", "dexamethasone", "prednisone")
    bronchodilator = _has(meds, "albuterol", "bronchodilator", "nebulizer", "ipratropium")
    ondansetron = _has(meds, "ondansetron", "zofran")
    supportive_only = _has(meds, "supportive care", "observation only", "reassurance",
                            "oral hydration") and not (benzo_given or antipyretic_given
                            or nippv_given or intubation_given or reversal_given
                            or antibiotics_given or steroids_given)

    # Disposition / escalation
    icu = "icu" in disp or "icu admission" in course
    discharged = "discharge" in disp
    floor = "floor" in disp or "ward" in disp or "admit" in disp

    # Trajectory phrases
    escalated = _has(course, "persistent instability requiring escalation",
                     "deteriorat", "worsen", "escalation", "required escalation",
                     "did not respond", "not respond", "failed to respond",
                     "inadequate response")
    stabilized = _has(course, "partial stabilization", "stabilized", "improved",
                      "resolved", "responded", "improvement", "normaliz")

    # ---- Class scoring ----

    # KRAKEN: sympathomimetic — benzo + antipyretic/cooling + IVF, hyperthermia
    if benzo_given and antipyretic_given:
        s["p_kraken"] += 3.0
        if stabilized and not escalated:
            s["p_kraken"] += 2.0  # benzo worked → strong Kraken
    if benzo_given and ivf_given and antipyretic_given:
        s["p_kraken"] += 1.5
    if antipyretic_given and not nippv_given and not intubation_given:
        s["p_kraken"] += 1.0
    if _has(exam, "diaphoretic", "tachycardic", "agitated", "tremor", "mydriasis", "hyperthermi"):
        s["p_kraken"] += 0.8
    if _has(hpi, "temp 38", "temp 39", "temp 40", "hyperthermi"):
        s["p_kraken"] += 0.8
    if _has(cc, "agitation", "hyperthermia", "palpitations"):
        s["p_kraken"] += 0.3

    # TRITON: sedative-hypnotic — airway support, reversal, NIPPV/intubation,
    # benzo failed (rare to give benzo here)
    if reversal_given:
        s["p_triton"] += 4.0  # very specific
    if intubation_given:
        s["p_triton"] += 2.5
    if nippv_given and not benzo_given:
        s["p_triton"] += 2.0
    if nippv_given and benzo_given and escalated:
        # benzo failed → escalated to airway
        s["p_triton"] += 2.5
    if _has(exam, "miosis", "bradycardic", "hypoventilation", "low_gcs",
            "decreased_resp", "somnolent", "obtund", "unresponsive"):
        s["p_triton"] += 1.2
    if _has(hpi, "gcs 8", "gcs 7", "gcs 6", "gcs 9", "gcs 10",
            "rr 8", "rr 9", "rr 10", "hr 4", "hr 5"):
        s["p_triton"] += 0.8
    if _has(cc, "altered", "unresponsive", "respiratory depression", "decreased loc"):
        s["p_triton"] += 0.5
    if icu and (nippv_given or intubation_given):
        s["p_triton"] += 1.0

    # CORAL: hallucinogenic / serotonergic — supportive, mild benzo, IVF only,
    # no airway, no antipyretic typically, stabilizes
    if supportive_only:
        s["p_coral"] += 2.5
    if ivf_given and not antipyretic_given and not nippv_given and not intubation_given \
            and not reversal_given and not antibiotics_given and not steroids_given:
        if benzo_given:
            s["p_coral"] += 1.5  # mild benzo + IVF
        else:
            s["p_coral"] += 1.0
    if _has(exam, "hyperreflex", "ataxia", "clonus", "hallucinat", "perceptual",
            "tremor") and not antipyretic_given and not intubation_given:
        s["p_coral"] += 1.0
    if _has(hpi, "hallucinat", "perceptual", "time-distortion", "visual distortion",
            "seeing things", "kaleidoscop"):
        s["p_coral"] += 1.2
    if _has(cc, "hallucination", "perceptual"):
        s["p_coral"] += 0.6
    if discharged and benzo_given and not antipyretic_given and not nippv_given:
        s["p_coral"] += 0.8

    # NONE: targeted medical treatment, no festival-tox cocktail
    targeted = (antibiotics_given or (nsaid_given and not benzo_given)
                or steroids_given or bronchodilator
                or (ondansetron and not benzo_given and not antipyretic_given))
    if targeted:
        s["p_none"] += 3.0
    if _has(course, "post-traumatic", "fracture", "sprain", "laceration",
            "appendicitis", "cholecystitis", "pneumonia", "uti", "cellulitis",
            "asthma exacerbation", "copd", "renal colic", "kidney stone",
            "migraine", "diabetic ketoacid", "dka"):
        s["p_none"] += 2.0
    if _has(mdm, "fracture", "sprain", "laceration", "appendicitis", "pneumonia",
            "uti", "cellulitis", "asthma", "copd", "renal colic", "migraine",
            "dka", "pulmonary embol", "cardiac", "stroke", "stemi"):
        s["p_none"] += 1.5
    if _has(cc, "chest pain", "shortness of breath", "abdominal pain", "back pain",
            "headache", "cough", "fever", "dysuria", "diarrhea", "vomiting") \
            and not _has(text, "festival", "substance exposure", "drug exposure"):
        s["p_none"] += 0.6
    # If no festival drug mentions at all and targeted care
    if not _has(text, "festival-related tox", "substance exposure",
                "festival-related substance", "drug exposure"):
        s["p_none"] += 0.8

    # Treatment-cocktail signature for festival tox (kraken/triton/coral)
    festival_tox = _has(mdm, "festival-related tox-metabolic",
                         "festival-related tox", "festival tox")
    if festival_tox:
        s["p_none"] -= 1.5

    # Escalation penalty on coral (mild hallucinogen self-resolves)
    if escalated:
        s["p_coral"] -= 1.0
        if not (reversal_given or intubation_given or nippv_given):
            s["p_kraken"] += 0.5  # severe sympathomimetic

    # ICU pattern: lean Triton (airway) or severe Kraken (hyperthermia)
    if icu:
        s["p_coral"] -= 0.5
        s["p_none"] -= 0.3
        if antipyretic_given:
            s["p_kraken"] += 0.8
        if nippv_given or intubation_given or reversal_given:
            s["p_triton"] += 1.0

    # Discharged + minimal intervention → coral or none
    if discharged and supportive_only:
        s["p_coral"] += 0.5
        s["p_none"] += 0.3

    return s


def softmax(scores: dict[str, float], temperature: float = 0.9) -> dict[str, float]:
    vals = [scores[c] / temperature for c in CLASSES]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    z = sum(exps)
    return {c: e / z for c, e in zip(CLASSES, exps)}


def classify(record: dict) -> dict[str, float]:
    raw = score(record)
    # Add small baseline so ambiguous records aren't pinned
    for c in CLASSES:
        raw[c] += 0.1
    probs = softmax(raw, temperature=0.9)
    # Renormalize (defensive)
    total = sum(probs.values())
    return {c: probs[c] / total for c in CLASSES}


def main() -> None:
    records = []
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    rows = []
    for rec in records:
        eid = rec["encounter_id"]
        probs = classify(rec)
        total = sum(probs.values())
        if abs(total - 1.0) > 0.005:
            probs = {c: v / total for c, v in probs.items()}
        rows.append((eid, probs["p_kraken"], probs["p_triton"],
                     probs["p_coral"], probs["p_none"]))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["encounter_id", "p_kraken", "p_triton", "p_coral", "p_none"])
        for r in rows:
            w.writerow([r[0]] + [f"{x:.6f}" for x in r[1:]])

    # Summary
    n = len(rows)
    sums = [r[1] + r[2] + r[3] + r[4] for r in rows]
    sum_ok = sum(1 for s in sums if abs(s - 1.0) <= 0.005)
    means = {c: sum(r[i + 1] for r in rows) / n for i, c in enumerate(CLASSES)}
    p_none_high = sum(1 for r in rows if r[4] > 0.5)

    print(f"path: {OUTPUT}")
    print(f"records: {n}")
    print(f"sum-validation (within 0.005 of 1.0): {sum_ok}/{n}")
    print(f"marginal means: " + ", ".join(f"{c}={means[c]:.4f}" for c in CLASSES))
    print(f"p_none > 0.5 count: {p_none_high}")


if __name__ == "__main__":
    main()
