"""Agent 4 — MDM-LED classifier for SAEM 2026 hackathon.

Reasoning emphasis: MDM and clinical_course are weighted heaviest. The
clinician's working impression and severity language are the closest thing
to ground truth.

Key structural insight in this dataset:
- Tox-presentation MDMs use one of three templated leads ("Differential
  included...", "Risk stratification...", "Medical decision-making
  prioritized..."). 219 of 261 records use these templates.
- Medical-pathology MDMs LEAD with the clinician's actual impression
  (e.g., "Elderly female with CAD/HTN...", "Presentation consistent with
  lower urinary tract infection...") even though the canned phrase
  "Working impression favored acute undifferentiated festival-related
  tox-metabolic presentation" is appended. 41 records start this way.

The treatment plan ("treatment plan emphasized X, Y, Z") in MDM and the
"ED course included X, Y, Z" in clinical_course are the strongest signals
for distinguishing Kraken / Triton / Coral within the tox cohort.

Class signature mapping:
  Kraken (sympathomimetic):
    - benzodiazepine for agitation/sympathetic excess
    - antipyretic therapy (for hyperthermia)
    - tachycardia, hypertension, hyperthermia, diaphoresis, agitation,
      mydriasis, tremor
  Triton (sedative-hypnotic):
    - targeted reversal trial (naloxone / flumazenil)
    - noninvasive positive pressure support, endotracheal intubation
      (respiratory depression)
    - bradycardia, hypotension, low GCS, low RR, miosis, somnolence
  Coral (hallucinogenic / serotonergic):
    - serial monitoring and supportive care (no reversal, no sympathetic
      excess management, no resp support — just observation)
    - perceptual changes, ataxia, unsteady gait, mild tachycardia,
      hyperreflexia
  None: medical pathology in lead clause of MDM.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(r"C:/Users/rs3te/Work/Claude-safe/SAEM-Hackathon")
IN_PATH = ROOT / "derived" / "narratives_fourh.jsonl"
OUT_PATH = ROOT / "derived" / "probs_4.csv"

# --- Detect medical-pathology lead in MDM --------------------------------

TOX_TEMPLATE_LEADS = (
    "Differential included",
    "Risk stratification incorporated",
    "Medical decision-making prioritized",
    "Given reported condom use",  # HIV PEP framing -- still tox
    "Working impression favored acute undifferentiated festival-related",  # bare start
)

# Phrases in a non-template lead that strongly indicate medical (none)
NONE_KEYWORDS = [
    r"\bsprain\b", r"\bcontusion\b", r"\bfracture\b", r"\bdislocation\b",
    r"\blaceration\b", r"\bcellulitis\b", r"\bcandidiasis\b",
    r"\buti\b", r"urinary tract infection", r"urethritis",
    r"chest pressure", r"\bcad\b", r"\bhtn\b",
    r"\bmigraine\b", r"\bheadache syndrome\b",
    r"\bpneumonia\b", r"\bbronchodilator\b", r"\basthma\b", r"\bcroup\b",
    r"\bconstipation\b", r"\bhernia\b", r"\bcholecystitis\b",
    r"vulvovaginal", r"\bdental\b",
    r"intracranial process", r"\bdvt\b",
    r"venous insufficiency", r"dependent edema",
    r"musculoskeletal", r"chest wall", r"costochondral",
    r"foreign body", r"\bcrush\b",
    r"\beye\b.*\b(?:tennis|trauma)", r"orbital",
    r"\barm\b dislocation",
    r"\bmvc\b",
    r"\bH1/H2\b", r"\bsteroid\b", r"\bantihistamine\b", r"\bnsaid\b",
    r"\bdtap\b|\btdap\b",
    r"\bcardiomegaly\b", r"pleural effusion", r"pulmonary vascular congestion",
    r"\bn/v/d\b", r"alcohol intake",
    r"\bhpv\b|\bsti\b",
    r"\bpsychiatric\b", r"safety plan",
    r"hypotension and worsening renal",
    r"acute febrile illness",
    r"discharge instructions",
    r"pharyngeal erythema",
    r"croup", r"dexamethasone",
    r"pelvic|hip injury",
    r"acute substernal",
    r"vulvovaginal",
    r"bridge refills",
    r"plan for symptomatic management",
    r"plan for urgent dental",
    r"return precautions",
]
NONE_LEAD_RE = re.compile("|".join(NONE_KEYWORDS), re.IGNORECASE)


def mdm_lead_class(mdm: str) -> str:
    """Return 'tox' if MDM begins with templated tox lead, else 'medical' if
    a non-template lead is present, else 'tox' as default."""
    if not mdm:
        return "tox"
    # If starts with one of the templated leads, it's a tox case.
    if mdm.startswith(TOX_TEMPLATE_LEADS):
        return "tox"
    # Otherwise the lead clause is the clinician's actual impression.
    # Take the first sentence and check for medical keywords.
    first_sentence = mdm.split(".")[0]
    if NONE_LEAD_RE.search(first_sentence):
        return "medical"
    # Default conservative: still tox if no clear medical signal.
    return "tox-uncertain"


# --- Treatment-plan parsing ----------------------------------------------

TX_RE = re.compile(r"treatment plan emphasized ([^.;]+)", re.IGNORECASE)
COURSE_RE = re.compile(r"ED course included ([^.;]+)", re.IGNORECASE)


def parse_treatments(mdm: str, course: str) -> set[str]:
    """Return set of normalized treatment tokens."""
    tx: set[str] = set()
    for src in (mdm, course):
        m = TX_RE.search(src) or COURSE_RE.search(src)
        if m:
            for item in m.group(1).split(","):
                tx.add(item.strip().lower())
    return tx


def severity_tier(mdm: str) -> str:
    m = re.search(r"severity tier (\w+)", mdm, re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"


def trajectory(course: str) -> str:
    c = course.lower()
    if "persistent instability" in c or "requiring escalation" in c:
        return "unstable"
    if "partial stabilization" in c:
        return "partial"
    if "improving" in c or "improvement" in c:
        return "improving"
    return "unknown"


# --- Vitals extraction ---------------------------------------------------

def extract_vitals(hpi: str) -> dict:
    out: dict = {}
    if not hpi:
        return out
    for label, pat, cast in [
        ("hr", r"HR\s+(\d+)", int),
        ("rr", r"RR\s+(\d+)", int),
        ("sbp", r"BP\s+(\d+)/\d+", int),
        ("temp", r"Temp\s+([\d.]+)\s*C", float),
        ("spo2", r"SpO2\s+(\d+)", int),
        ("gcs", r"GCS\s+(\d+)", int),
    ]:
        m = re.search(pat, hpi)
        if m:
            out[label] = cast(m.group(1))
    return out


def vital_signature(v: dict) -> dict:
    """Return per-class scores from vitals."""
    s = {"k": 0.0, "t": 0.0, "c": 0.0}
    hr = v.get("hr")
    bp = v.get("sbp")
    temp = v.get("temp")
    rr = v.get("rr")
    gcs = v.get("gcs")
    spo2 = v.get("spo2")
    if hr is not None:
        if hr >= 130:
            s["k"] += 2.5
        elif hr >= 110:
            s["k"] += 1.2
            s["c"] += 0.4
        elif hr <= 55:
            s["t"] += 2.5
        elif hr <= 65:
            s["t"] += 1.0
    if bp is not None:
        if bp >= 160:
            s["k"] += 1.5
        elif bp <= 90:
            s["t"] += 2.0
        elif bp <= 100:
            s["t"] += 0.8
    if temp is not None:
        if temp >= 39.5:
            s["k"] += 3.0
        elif temp >= 38.5:
            s["k"] += 1.5
        elif temp <= 35.5:
            s["t"] += 1.5
    if rr is not None:
        if rr <= 10:
            s["t"] += 2.5
        elif rr <= 12:
            s["t"] += 1.2
        elif rr >= 28:
            s["k"] += 1.0
    if gcs is not None:
        if gcs <= 8:
            s["t"] += 3.0
        elif gcs <= 12:
            s["t"] += 1.5
    if spo2 is not None and spo2 <= 92:
        s["t"] += 0.8
    return s


# --- Exam features -------------------------------------------------------

EXAM_KRAKEN = {"diaphoretic", "tachycardic", "agitated", "restless", "mydriasis", "hypertensive"}
EXAM_TRITON = {"reduced_tracking", "somnolent", "lethargic", "miosis", "bradycardic",
               "bradypneic", "hypotensive", "obtunded"}
EXAM_CORAL = {"ataxia", "unsteady_gait", "mild_tremor", "tremor", "dry_mucosa",
              "hyperreflexia", "clonus"}


def exam_signature(exam: str) -> dict:
    s = {"k": 0.0, "t": 0.0, "c": 0.0}
    tokens = {tok.strip().lower() for tok in re.split(r"[;,]", exam) if tok.strip()}
    for tok in tokens:
        if tok in EXAM_KRAKEN:
            s["k"] += 1.5
        if tok in EXAM_TRITON:
            s["t"] += 1.8
        if tok in EXAM_CORAL:
            s["c"] += 1.2
    # ataxia+unsteady together is very specific to coral
    if {"ataxia", "unsteady_gait"} & tokens and len(tokens & EXAM_CORAL) >= 2:
        s["c"] += 1.0
    # diaphoretic+tachycardic together is very specific to kraken
    if {"diaphoretic", "tachycardic"} <= tokens:
        s["k"] += 1.0
    return s


# --- Treatment signature -------------------------------------------------

def treatment_signature(tx: set[str]) -> dict:
    """Return per-class scores from treatment plan."""
    s = {"k": 0.0, "t": 0.0, "c": 0.0}
    tx_str = " | ".join(tx)
    has_benzo = any("benzodiazepine" in t for t in tx)
    has_antipyretic = any("antipyretic" in t for t in tx)
    has_reversal = any("reversal" in t for t in tx)
    has_nippv = any("noninvasive positive pressure" in t for t in tx)
    has_intubation = any("intubation" in t for t in tx) or any("endotracheal" in t for t in tx)
    has_serial = any("serial monitoring" in t or "supportive care" in t for t in tx)
    has_ivf = any("crystalloid" in t for t in tx)
    has_central_line = any("central venous" in t for t in tx)

    if has_benzo:
        s["k"] += 3.0
    if has_antipyretic:
        s["k"] += 2.0
    if has_reversal:
        s["t"] += 3.5
    if has_nippv:
        s["t"] += 2.0
    if has_intubation:
        s["t"] += 2.5
    if has_serial and not has_benzo and not has_reversal and not has_nippv and not has_intubation:
        # Pure supportive care -> coral signature (mild perceptual/ataxia)
        s["c"] += 2.5
    if has_ivf and not has_benzo and not has_reversal and not has_nippv and not has_intubation and not has_antipyretic:
        # Pure fluids -> mild presentation, slight coral lean
        s["c"] += 1.0
    if has_central_line:
        # Severe presentation, doesn't pick a class but increases confidence in
        # whichever is signaled
        pass
    return s


# --- Main classifier -----------------------------------------------------

def classify(rec: dict) -> tuple[float, float, float, float]:
    mdm = rec.get("mdm", "") or ""
    course = rec.get("clinical_course", "") or ""
    hpi = rec.get("hpi", "") or ""
    exam = rec.get("physical_exam_pertinent_positives", "") or ""
    brief = rec.get("brief_hpi", "") or ""

    lead = mdm_lead_class(mdm)
    sev = severity_tier(mdm)
    traj = trajectory(course)
    tx = parse_treatments(mdm, course)
    vitals = extract_vitals(hpi)

    v_sig = vital_signature(vitals)
    e_sig = exam_signature(exam)
    t_sig = treatment_signature(tx)

    # Combine class scores (Kraken, Triton, Coral)
    k = v_sig["k"] + e_sig["k"] + t_sig["k"]
    t = v_sig["t"] + e_sig["t"] + t_sig["t"]
    c = v_sig["c"] + e_sig["c"] + t_sig["c"]

    # If no tox signal at all but clearly a tox case, give small uniform baseline
    if k + t + c < 0.5 and lead == "tox":
        # Default lean toward coral (mild) when no signal
        c += 0.5
        k += 0.3
        t += 0.3

    # Severity tier modifies confidence (high severity sharpens the dominant
    # signal; low severity flattens)
    if sev == "high" or traj == "unstable":
        amp = 1.3
    elif sev == "low":
        amp = 0.9
    else:
        amp = 1.0
    k *= amp; t *= amp; c *= amp

    # Build the four class scores; none score depends on the MDM lead
    if lead == "medical":
        # The clinician's lead clause is a medical diagnosis. p_none should
        # be high. Keep some residual on the tox classes proportional to
        # whatever signal exists (synthetic dataset is messy).
        n_raw = 6.0  # strong none
        # Dampen tox signals
        k *= 0.4; t *= 0.4; c *= 0.4
    elif lead == "tox":
        n_raw = 0.3  # minimal none -- canned tox lead
    else:  # tox-uncertain
        n_raw = 1.0

    raw = [max(0.0, x) for x in (k, t, c, n_raw)]

    # Softmax with temperature
    T = 1.8
    mx = max(raw) if max(raw) > 0 else 1.0
    exps = [math.exp((x - mx) / T) for x in raw]
    Z = sum(exps)
    probs = [e / Z for e in exps]

    # Floor each prob at 0.02 to avoid 0/1 extremes
    FLOOR = 0.02
    probs = [max(p, FLOOR) for p in probs]
    Z = sum(probs)
    probs = [p / Z for p in probs]

    # If the medical lead was clear, ensure p_none >= 0.7 per ground rules
    if lead == "medical" and probs[3] < 0.7:
        # Pull p_none up to 0.72, redistribute the rest proportionally
        rest_total = 1.0 - 0.72
        rest = [probs[i] for i in range(3)]
        rest_sum = sum(rest) or 1.0
        rest = [r * rest_total / rest_sum for r in rest]
        probs = [rest[0], rest[1], rest[2], 0.72]

    return tuple(probs)


def round_and_normalize(p: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    r = [round(x, 3) for x in p]
    diff = round(1.0 - sum(r), 3)
    if abs(diff) > 1e-9:
        idx = r.index(max(r))
        r[idx] = round(r[idx] + diff, 3)
    return tuple(r)


def main() -> None:
    records = []
    with IN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    rows = []
    for rec in records:
        probs = classify(rec)
        probs = round_and_normalize(probs)
        rows.append((rec["encounter_id"], *probs))

    sum_ok = sum(1 for row in rows if abs(sum(row[1:]) - 1.0) <= 0.005)
    means = [sum(row[i] for row in rows) / len(rows) for i in range(1, 5)]
    n_none_gt_50 = sum(1 for row in rows if row[4] > 0.5)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["encounter_id", "p_kraken", "p_triton", "p_coral", "p_none"])
        for row in rows:
            w.writerow(row)

    print(f"Wrote {OUT_PATH}")
    print(f"Records: {len(rows)}")
    print(f"Sum-validation count (|sum-1| <= 0.005): {sum_ok}/{len(rows)}")
    print(
        f"Marginal means -- kraken: {means[0]:.4f}, triton: {means[1]:.4f}, "
        f"coral: {means[2]:.4f}, none: {means[3]:.4f}"
    )
    print(f"p_none > 0.5 count: {n_none_gt_50}")


if __name__ == "__main__":
    main()
