"""Agent 1 — Equal Weighting toxidrome classifier.

Reads `derived/narratives.jsonl`, applies equal-weight evidence scoring across
the eight narrative fields, and writes `derived/probs_1.csv` with 4-class
probabilities per encounter.

Classes (FIXED mapping for this batch):
    kraken -> sympathomimetic / stimulant
    triton -> sedative-hypnotic / depressant
    coral  -> hallucinogenic / serotonergic
    none   -> non-festival medical pathology
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(r"C:\Users\rs3te\Work\Claude-safe\SAEM-Hackathon")
IN_PATH = ROOT / "derived" / "narratives.jsonl"
OUT_PATH = ROOT / "derived" / "probs_1.csv"

FIELDS = [
    "brief_hpi",
    "hpi",
    "physical_exam_pertinent_positives",
    "mdm",
    "clinical_course",
    "ed_meds_procedures",
    "triage_brief_note",
    "chief_complaint",
]

# ---------------------------------------------------------------------------
# Lexicons. Each is a list of (regex, weight). Weights calibrated against the
# observed signal frequency in this corpus:
#   - "benzodiazepine for agitation/sympathetic excess", "antipyretic therapy",
#     "IV crystalloid bolus", "cooling" are GENERIC festival templates — we
#     down-weight them.
#   - PE positive tokens (diaphoretic, restless, tachycardic, agitated, ataxia,
#     slow_responses, distractible, unsteady_gait, dry_mucosa, etc.) are the
#     strongest class signals.
#   - Procedures like "endotracheal intubation", "noninvasive positive pressure",
#     "targeted reversal trial" strongly favor triton.
# ---------------------------------------------------------------------------

KRAKEN_PATTERNS: list[tuple[str, float]] = [
    # PE token forms (semicolon-delimited in physical_exam_pertinent_positives)
    (r"\bdiaphoretic\b", 2.2),
    (r"\brestless\b", 1.6),
    (r"\btachycardic\b", 2.0),
    (r"\bagitated\b", 2.0),
    (r"\btremulous\b", 1.6),
    (r"\bhyperthermic\b", 2.2),
    (r"\bflushed\b", 1.0),
    (r"\bmydriatic|dilated_pupil", 2.5),
    (r"\bhypertensive\b", 1.5),
    # HPI / MDM phrasing
    (r"\bdiaphores", 2.0),
    (r"\bagitation\b", 1.0),
    (r"\bmarked\s+restless", 1.5),
    (r"\bsympathetic\s+excess", 0.5),  # generic template — small weight
    (r"\bstimulant", 2.5),
    (r"\brapid\s+heartbeat|palpitation", 1.2),
    (r"\bchills\b", 0.4),
    (r"\bbruxism|teeth\s+grinding|jaw\s+clench", 2.0),
    (r"\bhyperactiv", 1.2),
    (r"\brhabdo", 1.5),
    (r"\bseizure", 0.8),
    # vitals
    (r"hr\s*1[3-9]\d|hr\s*[2-9]\d{2}", 1.6),  # HR 130+
    (r"temp\s*3[89]\.[0-9]|temp\s*4\d", 1.6),  # ≥38.0
    (r"bp\s*1[6-9]\d|bp\s*2\d{2}", 1.0),
    # generic stimulant-ish treatments (LOW weight — template language)
    (r"\bcooling\b", 0.3),
    (r"\bantipyretic\s+therapy", 0.2),
    (r"\bbenzodiazepine\s+for\s+agitation", 0.3),
]

TRITON_PATTERNS: list[tuple[str, float]] = [
    # PE tokens that map to depressant state
    (r"\bdistractible\b", 1.6),
    (r"\bslow_responses\b|\bslow\s+responses\b", 2.2),
    (r"\breduced_tracking\b", 2.0),
    (r"\bintermittent_disorientation\b", 1.8),
    (r"\btachypneic_effort\b|\btachypneic\s+effort\b", 1.5),  # respiratory failure
    (r"\bsomnolent|drowsy|lethargic|obtunded?\b|stuporous", 2.5),
    (r"\bunresponsive\b", 2.5),
    (r"\bbradycardic\b", 2.5),
    (r"\bbradypnea|hypoventil|apnea|shallow_breath", 2.5),
    (r"\bmiotic|pinpoint|constricted_pupil|miosis", 2.5),
    (r"\bhypotensive\b", 1.8),
    # HPI phrasing
    (r"\blow\s+gcs|gcs\s*[3-9]\b|gcs\s*1[0-1]\b", 2.0),
    (r"\brespiratory\s+depression|cns\s+depression", 2.5),
    (r"\bsedat(ed|ion|ive)", 2.0),
    (r"\bdepressant|opioid|opiate", 2.5),
    (r"\bnodding\b|nodded\s+off", 1.5),
    (r"\bsnoring\s+respir", 2.0),
    # procedures that strongly indicate triton
    (r"\bendotracheal\s+intubat", 2.5),
    (r"\bintubat", 2.0),
    (r"\bnoninvasive\s+positive\s+pressure|nippv|bipap|cpap", 2.5),
    (r"\bairway\s+support|bag[-\s]?mask", 2.0),
    (r"\btargeted\s+reversal", 3.0),
    (r"\bnaloxone|narcan", 3.5),
    (r"\bflumazenil", 3.5),
    (r"\breversal\s+agent|reversal\s+trial", 2.5),
    # vitals
    (r"hr\s*[3-5]\d\b", 2.0),
    (r"spo2\s*[5-8]\d\b", 1.5),
    (r"bp\s*[5-8]\d/", 1.5),
]

CORAL_PATTERNS: list[tuple[str, float]] = [
    # PE tokens
    (r"\bataxia\b", 2.0),
    (r"\bunsteady_gait\b|\bunsteady\s+gait\b", 2.0),
    (r"\bdry_mucosa\b", 1.4),
    (r"\bhyperreflex", 2.0),
    (r"\bclonus\b", 2.0),
    (r"\bmildly_tachycardic\b|\bmild\s+tachy", 1.0),
    (r"\bfatigued_appearance\b", 0.3),  # weak / shared signal
    # HPI / MDM phrasing — perceptual / hallucinogenic
    (r"\bperceptual\s+(change|distort|disturb|alter)", 2.5),
    (r"\bvisual\s+(disturb|halluc|distort)", 2.2),
    (r"\bhallucinat", 2.5),
    (r"\btime[-\s]?distort|time\s+dilation", 2.5),
    (r"\bsynesth", 2.5),
    (r"\bderealiz|depersonaliz|dissociat", 2.0),
    (r"\bwave[-\s]like|wavy\s+vision|kaleidoscop|vivid\s+(color|imagery)", 2.5),
    (r"\bsensory\s+(overload|amplif|trigger)", 1.2),
    (r"\bserotonergic|serotonin\s+syndrome", 2.5),
    (r"\bpsychedelic|hallucinogen", 2.5),
    (r"\bunsteady\b", 0.8),
    # treatments: bare "supportive care" / "antipyretic therapy" alone often
    # accompanies the milder hallucinogenic picture
    (r"\bserial\s+monitoring\s+and\s+supportive\s+care", 1.2),
    (r"\bdim/?quiet\s+room|quiet\s+dim|dim\s+room", 2.5),
]

# Non-festival pathology indicators -> push p_none up. Note: these are
# clinically-coded findings that point to a different primary problem.
NONE_PATTERNS: list[tuple[str, float]] = [
    (r"\bc-?diff", 2.5),
    (r"\bcellulit", 2.5),
    (r"\bappendicit", 2.5),
    (r"\bcholecystit", 2.5),
    (r"\bpancreatit", 2.5),
    (r"\bdiverticulit", 2.5),
    (r"\bdiabetic\s+ketoacid|dka\b", 2.5),
    (r"\bpyelonephrit", 2.5),
    (r"\burinary\s+tract\s+infection|\buti\b", 2.0),
    (r"\bpneumon", 2.0),
    (r"\bsepsis\b|septic\s+shock", 2.0),
    (r"\bstroke\b|cva\b|tia\b", 2.5),
    (r"\bmyocardial\s+infarct|stemi|nstemi|\bacs\b", 2.5),
    (r"\bpulmonary\s+embol|\bpe\b", 2.0),
    (r"\bgi\s+bleed|hematemesis|melena", 2.2),
    (r"\bkidney\s+stone|nephrolithia|renal\s+colic", 2.2),
    (r"\bectopic\s+pregnan", 2.5),
    (r"\bvaginal\s+bleed", 1.5),
    (r"\bfracture", 2.0),
    (r"\blaceration", 1.5),
    (r"\bdental\s+pain|toothache", 2.0),
    (r"\botitis|sinusit|tonsillit|pharyngit", 2.0),
    (r"\bvertigo|bppv\b", 1.8),
    (r"\bnon[-\s]festival|unrelated\s+to\s+festival", 3.0),
    (r"\bprimary\s+medical\s+pathology", 2.5),
    # NEGATIVE weights: things that REDUCE p_none (i.e., suggest festival tox)
    (r"\bfestival\b", -2.0),
    (r"\bfestival[-\s]related\s+tox", -2.5),
    (r"\bsubstance\s+exposure|drug\s+ingest|unknown\s+ingest", -2.0),
    (r"\btox[-\s]?metabolic", -2.5),
    (r"\bbenzodiazepine\s+for\s+agitation", -1.0),
    (r"\bendotracheal\s+intubat", -1.0),
    (r"\btargeted\s+reversal", -1.5),
    (r"\bnaloxone|flumazenil", -2.0),
    (r"\bdance\s+tent|main\s+stage|crowd", -1.0),
]


def score_field(text: str, patterns: list[tuple[str, float]]) -> float:
    """Sum weight contributions from regex hits (log-dampened for multi-hit)."""
    if not text:
        return 0.0
    t = text.lower()
    total = 0.0
    for pat, w in patterns:
        hits = len(re.findall(pat, t))
        if hits <= 0:
            continue
        total += w * math.log1p(hits)
    return total


def score_encounter(rec: dict) -> tuple[float, float, float, float]:
    """Equal-weight per-field scoring: every FIELD contributes equally."""
    sums = [0.0, 0.0, 0.0, 0.0]
    n = 0
    for fld in FIELDS:
        text = rec.get(fld) or ""
        if not isinstance(text, str):
            text = str(text)
        sums[0] += score_field(text, KRAKEN_PATTERNS)
        sums[1] += score_field(text, TRITON_PATTERNS)
        sums[2] += score_field(text, CORAL_PATTERNS)
        sums[3] += score_field(text, NONE_PATTERNS)
        n += 1
    if n > 0:
        sums = [s / n for s in sums]
    return tuple(sums)  # type: ignore[return-value]


def to_probs(scores: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert raw scores to a 4-class posterior with a uniform fallback."""
    k, t, c, n = scores
    raw = [k, t, c, n]
    # If virtually no class-discriminating signal, return uniform.
    if max(raw) < 0.12 and abs(n) < 0.12:
        return (0.25, 0.25, 0.25, 0.25)

    # Clamp negatives (the NONE category can go negative when festival cues fire)
    clamped = [max(v, 0.0) for v in raw]
    # Prior pseudo-evidence keeps things calibrated and never zero-collapses.
    prior = 0.30
    vals = [v + prior for v in clamped]

    # Softmax with moderate temperature.
    temp = 0.95
    z = [v / temp for v in vals]
    m = max(z)
    exps = [math.exp(zi - m) for zi in z]
    s = sum(exps)
    probs = [e / s for e in exps]
    return tuple(probs)  # type: ignore[return-value]


def round_and_normalize(probs: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Round to 3 decimals; absorb residual into the largest entry."""
    r = [round(p, 3) for p in probs]
    diff = round(1.0 - sum(r), 3)
    if diff != 0.0:
        idx = max(range(4), key=lambda i: r[i])
        r[idx] = round(r[idx] + diff, 3)
    r = [max(0.0, x) for x in r]
    s2 = sum(r)
    if abs(s2 - 1.0) > 0.005:
        r = [x / s2 for x in r]
        r = [round(x, 3) for x in r]
        diff2 = round(1.0 - sum(r), 3)
        if diff2 != 0.0:
            idx = max(range(4), key=lambda i: r[i])
            r[idx] = round(r[idx] + diff2, 3)
    return tuple(r)  # type: ignore[return-value]


def main() -> None:
    records: list[dict] = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    rows = []
    for rec in records:
        eid = rec.get("encounter_id", "")
        raw = score_encounter(rec)
        probs = to_probs(raw)
        probs = round_and_normalize(probs)
        rows.append((eid, *probs))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        f.write("encounter_id,p_kraken,p_triton,p_coral,p_none\n")
        for eid, pk, pt, pc, pn in rows:
            f.write(f"{eid},{pk:.3f},{pt:.3f},{pc:.3f},{pn:.3f}\n")

    # ---- verification ----
    n_records = len(rows)
    n_bad = 0
    sum_k = sum_t = sum_c = sum_n = 0.0
    high_none_rows: list[tuple[str, float]] = []
    for eid, pk, pt, pc, pn in rows:
        s = pk + pt + pc + pn
        if abs(s - 1.0) > 0.005:
            n_bad += 1
        sum_k += pk
        sum_t += pt
        sum_c += pc
        sum_n += pn
        if pn > 0.5:
            high_none_rows.append((eid, pn))

    print(f"path: {OUT_PATH}")
    print(f"records: {n_records}")
    print(f"sum_validation_ok: {n_records - n_bad}/{n_records}")
    print(
        "marginal_means: "
        f"p_kraken={sum_k/n_records:.4f} "
        f"p_triton={sum_t/n_records:.4f} "
        f"p_coral={sum_c/n_records:.4f} "
        f"p_none={sum_n/n_records:.4f} "
        f"total={(sum_k+sum_t+sum_c+sum_n)/n_records:.4f}"
    )
    print(f"high_p_none_rows_count: {len(high_none_rows)}")
    for eid, pn in high_none_rows:
        print(f"  {eid}\tp_none={pn:.3f}")


if __name__ == "__main__":
    main()
