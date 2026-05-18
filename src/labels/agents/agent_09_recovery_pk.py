"""Agent 9: Recovery-pattern pharmacokinetics classifier.

Classifies SAEM 2026 hackathon ED encounters into 4 classes
(Kraken/Sympathomimetic, Triton/Sedative, Coral/Hallucinogen, None/medical)
based on the temporal arc of the encounter as recorded in
clinical_course, mdm, and hpi.

Independent hypothesis: PK trajectory shape carries the toxidrome signal.
- Kraken: rapid onset, peak <2h, resolves within 4-8h.
- Triton: delayed peak / re-sedation, prolonged recovery, plateau at 4h.
- Coral: gradual self-resolution over 2-6h with supportive care.
- None: condition-specific trajectory (antibiotics improvement, post-CT,
  cardiology w/u stable, etc.).

Fully synthetic data — no PHI.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3] / "derived"
NARR = HERE / "narratives_fourh.jsonl"
OUT = HERE / "probs_9.csv"


def score_text(text: str) -> dict[str, float]:
    """Return raw class scores for one encounter based on PK-trajectory cues."""
    t = text.lower()
    s = {"kraken": 0.0, "triton": 0.0, "coral": 0.0, "none": 0.0}

    # ------------------------------------------------------------------
    # KRAKEN — rapid onset, fast resolution, sympathetic surge clearing
    # ------------------------------------------------------------------
    kraken_recovery = [
        ("settled with benzo", 3.0),
        ("rapidly responsive to cooling", 3.0),
        ("vitals normalized by hour 2", 3.0),
        ("vitals normalized", 1.5),
        ("rapid improvement", 2.0),
        ("rapidly improved", 2.0),
        ("resolved within 4", 2.5),
        ("symptoms resolved", 1.5),
        ("fully resolved", 1.8),
        ("normalized within 2", 2.5),
        ("normalized within hour", 1.5),
        ("hr trended down", 1.5),
        ("tachycardia resolved", 2.0),
        ("rapid stabilization", 1.8),
        ("temperature trended down", 1.5),
        ("temperature normalized", 1.5),
        ("agitation improved with benzo", 2.0),
        ("responded to benzodiazepine", 1.5),
        ("active cooling", 1.0),
        ("rapid de-escalation", 2.0),
    ]
    # Onset and presentation cues that fit sympathomimetic PK
    kraken_onset = [
        ("rapid onset", 1.0),
        ("acute onset", 0.6),
        ("symptoms peaked within", 1.5),
        ("peaked within 1 hour", 2.0),
        ("peaked within 2 hours", 2.0),
        ("pre-hospital", 0.5),
        ("ems administered cooling", 1.0),
        ("agitation/sympathetic excess", 0.8),
        ("hyperthermia", 1.0),
        ("severe hypertension", 0.8),
        ("severe tachycardia", 0.8),
        ("diaphoretic", 0.4),
        ("dilated pupils", 0.5),
        ("mydriasis", 0.5),
    ]

    # ------------------------------------------------------------------
    # TRITON — prolonged / re-sedation / minimal improvement
    # ------------------------------------------------------------------
    triton_recovery = [
        ("remained somnolent", 3.0),
        ("remained sedated", 3.0),
        ("required prolonged airway", 3.0),
        ("prolonged airway support", 3.0),
        ("minimal improvement despite reversal", 3.0),
        ("re-sedation", 3.0),
        ("re-sedated", 3.0),
        ("recurrent sedation", 2.5),
        ("required repeat naloxone", 2.0),
        ("required repeat flumazenil", 2.0),
        ("required intubation", 2.0),
        ("intubated", 1.5),
        ("persistent instability requiring escalation", 2.0),
        ("persistent hypoventilation", 2.5),
        ("persistent bradypnea", 2.5),
        ("persistent somnolence", 2.5),
        ("prolonged observation", 2.0),
        ("prolonged recovery", 2.5),
        ("delayed peak", 2.5),
        ("plateau", 1.5),
        ("slow to wake", 2.5),
        ("did not improve", 2.0),
        ("minimal response", 2.0),
        ("admitted for monitoring beyond 4", 1.5),
        ("admit to icu", 1.0),
        ("icu admission", 1.0),
        ("required bipap", 1.5),
        ("required mechanical ventilation", 2.0),
        ("airway protected", 1.2),
        ("naloxone administered", 1.0),
        ("flumazenil", 1.5),
        ("respiratory depression", 1.5),
        ("pinpoint pupils", 0.8),
        ("miosis", 0.8),
        ("bradypnea", 1.0),
        ("hypoventilation", 1.0),
        ("gcs 8", 1.2),
        ("gcs 7", 1.5),
        ("gcs 6", 1.5),
        ("gcs 5", 1.8),
        ("hypothermia", 0.8),
        ("airway support", 1.0),
        ("supplemental oxygen", 0.3),
    ]

    # ------------------------------------------------------------------
    # CORAL — gradual self-resolution, perceptual symptoms wane
    # ------------------------------------------------------------------
    coral_recovery = [
        ("perceptual symptoms improving", 3.0),
        ("perceptual symptoms resolved", 3.0),
        ("hallucinations improving", 2.5),
        ("hallucinations resolved", 2.5),
        ("visual phenomena improving", 2.5),
        ("remained calm with quiet room", 3.0),
        ("calm with quiet environment", 2.5),
        ("low-stimulation environment", 2.5),
        ("reduced stimulation", 1.8),
        ("dim lighting", 1.5),
        ("tolerated po at hour 3", 3.0),
        ("tolerated po", 1.5),
        ("gradual resolution", 2.5),
        ("gradually improved", 2.0),
        ("waning over", 2.0),
        ("symptoms waned", 2.0),
        ("symptoms wane", 2.0),
        ("reality testing returned", 2.5),
        ("reoriented", 1.5),
        ("became oriented", 1.5),
        ("oriented x3 by hour", 1.8),
        ("supportive care only", 1.5),
        ("supportive care", 0.8),
        ("verbal reassurance", 1.5),
        ("reassurance and observation", 1.5),
        ("perceptual disturbance", 1.5),
        ("visual hallucination", 1.5),
        ("auditory hallucination", 1.5),
        ("synesthesia", 2.0),
        ("time-distortion", 1.8),
        ("time distortion", 1.8),
        ("derealization", 1.5),
        ("depersonalization", 1.5),
        ("dissociation", 1.0),
        ("kaleidoscopic", 2.0),
        ("ego dissolution", 2.0),
        ("trippy", 1.5),
        ("seeing patterns", 1.8),
        ("colors brighter", 1.5),
        ("geometric patterns", 2.0),
    ]

    # ------------------------------------------------------------------
    # NONE — targeted medical workup, condition-specific course
    # ------------------------------------------------------------------
    none_recovery = [
        ("improved with antibiotics", 3.0),
        ("improved after antibiotics", 3.0),
        ("started on antibiotics", 2.0),
        ("ceftriaxone", 1.8),
        ("vancomycin", 1.8),
        ("piperacillin", 1.8),
        ("azithromycin", 1.5),
        ("metronidazole", 1.5),
        ("antibiotic therapy", 1.5),
        ("stable post-ct", 2.5),
        ("ct head negative", 1.5),
        ("ct head unremarkable", 1.5),
        ("ct abdomen", 1.0),
        ("ct angiography", 1.5),
        ("non-contrast ct", 1.0),
        ("appendicitis", 2.0),
        ("cholecystitis", 2.0),
        ("pancreatitis", 2.0),
        ("pyelonephritis", 2.0),
        ("pneumonia", 2.0),
        ("urinary tract infection", 2.0),
        ("uti", 1.5),
        ("sepsis", 1.5),
        ("dka", 2.5),
        ("diabetic ketoacidosis", 2.5),
        ("ischemic stroke", 2.5),
        ("hemorrhagic stroke", 2.5),
        ("stroke alert", 2.0),
        ("stemi", 2.5),
        ("nstemi", 2.5),
        ("acute coronary syndrome", 2.0),
        ("troponin elevated", 1.5),
        ("ekg showed", 1.0),
        ("ekg demonstrated", 1.0),
        ("admitted to cardiology", 2.0),
        ("cardiology consulted", 1.5),
        ("neurology consulted", 1.5),
        ("surgery consulted", 1.5),
        ("orthopedics consulted", 1.5),
        ("gi consult", 1.2),
        ("fracture identified", 2.0),
        ("laceration repair", 2.0),
        ("sutures placed", 1.5),
        ("reduction performed", 1.5),
        ("foreign body removed", 1.8),
        ("incision and drainage", 1.8),
        ("started on insulin", 1.8),
        ("anticoagulation initiated", 1.5),
        ("tpa administered", 2.5),
        ("blood cultures drawn", 1.0),
        ("urine culture", 0.8),
        ("trauma workup", 1.2),
        ("pe ruled out", 1.5),
        ("d-dimer", 0.8),
        ("ct pulmonary angiogram", 1.8),
        ("pulmonary embolism", 2.0),
        ("migraine cocktail", 2.0),
        ("anaphylaxis", 2.0),
        ("epinephrine im", 1.5),
        ("asthma exacerbation", 2.0),
        ("copd exacerbation", 2.0),
        ("chf exacerbation", 2.0),
        ("acute decompensation of chf", 2.0),
        ("seizure resolved", 1.5),
        ("status epilepticus", 2.0),
        ("hyperglycemia", 1.2),
        ("hypoglycemia", 1.2),
        ("dextrose administered", 1.0),
        ("electrolyte repletion", 1.2),
        ("symptoms attributable to underlying", 1.5),
        ("chronic condition exacerbation", 1.5),
        ("specific medical diagnosis", 1.5),
    ]

    # Negative cues that argue AGAINST a tox-festival presentation
    none_negative_for_tox = [
        ("denies substance use", 1.5),
        ("denies drug use", 1.5),
        ("denies festival", 1.0),
        ("no festival attendance", 1.5),
        ("no recreational substance", 1.5),
        ("no toxic ingestion", 1.5),
        ("toxic screen negative", 1.2),
        ("urine drug screen negative", 1.2),
        ("uds negative", 1.0),
    ]

    # Apply scoring
    for kw, w in kraken_recovery + kraken_onset:
        if kw in t:
            s["kraken"] += w
    for kw, w in triton_recovery:
        if kw in t:
            s["triton"] += w
    for kw, w in coral_recovery:
        if kw in t:
            s["coral"] += w
    for kw, w in none_recovery + none_negative_for_tox:
        if kw in t:
            s["none"] += w

    # Festival/tox-metabolic context is a strong NEGATIVE for "none"
    festival_context = [
        ("festival", -0.8),
        ("rave", -0.8),
        ("concert", -0.4),
        ("music festival", -1.0),
        ("dance tent", -0.8),
        ("substance exposure", -1.2),
        ("possible ingestion", -1.0),
        ("recreational drug", -1.2),
        ("street drug", -1.2),
        ("club drug", -1.2),
        ("tox-metabolic presentation", -1.5),
        ("tox-metabolic", -1.0),
    ]
    for kw, w in festival_context:
        if kw in t:
            s["none"] += w  # negative weight reduces None

    # Disposition tail — gentle PK shape cues
    if "icu" in t or "intensive care" in t:
        # Severe end-organ injury fits Triton (prolonged) or severe None
        s["triton"] += 1.0
        s["none"] += 0.4
    if re.search(r"\bdischarge\b", t):
        # Quick discharge fits Kraken or Coral (rapid/gradual self-resolution)
        s["kraken"] += 0.4
        s["coral"] += 0.4
    if "observation" in t and "discharge" in t:
        s["kraken"] += 0.3
        s["coral"] += 0.3
    if "admitted to medical floor" in t or "admit to medicine" in t:
        s["none"] += 0.8
        s["triton"] += 0.3

    # Severity tier mentions
    if "severity tier high" in t:
        s["triton"] += 0.6
    if "severity tier low" in t:
        s["kraken"] += 0.3
        s["coral"] += 0.3
    if "severity tier moderate" in t:
        pass  # neutral

    return s


def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Convert raw scores into a probability vector via softmax."""
    keys = ["kraken", "triton", "coral", "none"]
    vals = np.array([scores[k] / temperature for k in keys], dtype=float)
    # Stabilise
    vals = vals - vals.max()
    exps = np.exp(vals)
    probs = exps / exps.sum()
    return dict(zip(keys, probs.tolist()))


def smooth(probs: dict[str, float], alpha: float = 0.05) -> dict[str, float]:
    """Mix in a small uniform prior so no class is ever exactly zero."""
    keys = ["kraken", "triton", "coral", "none"]
    out = {k: (1 - alpha) * probs[k] + alpha * 0.25 for k in keys}
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def classify_record(rec: dict) -> dict[str, float]:
    """Score one record. Uses hpi, mdm, clinical_course (PK signal)."""
    parts = [
        rec.get("brief_hpi", ""),
        rec.get("hpi", ""),
        rec.get("mdm", ""),
        rec.get("clinical_course", ""),
        rec.get("ed_meds_procedures", ""),
        rec.get("disposition", ""),
        rec.get("chief_complaint", ""),
        rec.get("physical_exam_pertinent_positives", ""),
    ]
    text = "\n".join(str(p) for p in parts if p)
    raw = score_text(text)
    probs = softmax(raw, temperature=1.6)
    probs = smooth(probs, alpha=0.06)
    return probs


def main() -> None:
    rows = []
    with NARR.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            enc_id = rec["encounter_id"]
            p = classify_record(rec)
            rows.append((enc_id, p["kraken"], p["triton"], p["coral"], p["none"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        f.write("encounter_id,p_kraken,p_triton,p_coral,p_none\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f}\n")

    # Validation report
    n = len(rows)
    sums = [r[1] + r[2] + r[3] + r[4] for r in rows]
    sum_ok = all(abs(s - 1.0) <= 0.005 for s in sums)
    mean_k = sum(r[1] for r in rows) / n
    mean_t = sum(r[2] for r in rows) / n
    mean_c = sum(r[3] for r in rows) / n
    mean_n = sum(r[4] for r in rows) / n
    p_none_majority = sum(1 for r in rows if r[4] > 0.5)

    print(f"path={OUT}")
    print(f"records={n}")
    print(f"sum_validation_ok={sum_ok} min_sum={min(sums):.6f} max_sum={max(sums):.6f}")
    print(
        f"marginal_means k={mean_k:.4f} t={mean_t:.4f} "
        f"c={mean_c:.4f} n={mean_n:.4f}"
    )
    print(f"p_none_gt_0.5_count={p_none_majority}")


if __name__ == "__main__":
    main()
