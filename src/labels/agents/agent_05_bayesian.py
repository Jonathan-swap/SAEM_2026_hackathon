"""Agent 5: Conservative / Bayesian classifier for SAEM hackathon narratives.

Produces probability distributions over {kraken, triton, coral, none}.
Bias: reserve high confidence only when >=2 narrative fields independently
point to the same toxidrome; otherwise spread probability mass.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

IN = Path(r"C:\Users\rs3te\Work\Claude-safe\SAEM-Hackathon\derived\narratives.jsonl")
OUT = Path(r"C:\Users\rs3te\Work\Claude-safe\SAEM-Hackathon\derived\probs_5.csv")

# ----------------------------- Lexicons -----------------------------------
# Each lexicon is a list of regex patterns. Hits in distinct fields are counted
# as independent evidence (Bayesian intuition: corroboration > repetition).

KRAKEN = [  # SYMPATHOMIMETIC
    r"\btachycard",
    r"\bhypertens",
    r"\bhypertherm",
    r"\bdiaphor",
    r"\bagitat",
    r"\btremor",
    r"\bmydriasis|dilated pupil",
    r"\bsympathomimetic|sympathetic excess|sympathetic surge",
    r"\bhyperthermi|temp\s*(?:39|40|41|42)",
    r"\bstimulant",
    r"\brestless",
    r"\bjaw\s*clench|bruxism|teeth\s*grind",
    r"\bchest pain.*(?:young|stimulant)",
    r"\bseizure",
    r"\bcombative|aggressive",
]

TRITON = [  # SEDATIVE-HYPNOTIC
    r"\bbradycard",
    r"\bhypotens",
    r"\bhypoventil|respiratory depress|low respiratory rate|RR\s*[0-9]\b|RR\s*1[0-1]\b",
    r"\bmiosis|pinpoint pupil|constricted pupil",
    r"\bsomnolen|obtund|stupor|lethargic|drowsy|sedated|sedation",
    r"\bGCS\s*(?:[3-9]|1[0-1])\b",
    r"\bslurred speech",
    r"\bshallow (?:breath|respir)",
    r"\bunresponsive|unconscious|altered mental status",
    r"\bnarcan|naloxone|flumazenil",
    r"\bopioid|opiate|benzo(?:diazepine)? overdose",
    r"\bairway protect|intubat",
    r"\bbag(?:-|\s)?valve|BVM",
]

CORAL = [  # HALLUCINOGENIC / SEROTONERGIC
    r"\bhallucinat",
    r"\bperceptual",
    r"\bvisual disturbance|visual change|seeing things|illusions",
    r"\bataxi",
    r"\bunsteady (?:gait|on feet)|wide-?based gait",
    r"\bhyperreflex",
    r"\bclonus|myoclon",
    r"\bserotonergic|serotonin syndrome",
    r"\btime[-\s]?distortion|sense of unreality|derealization|depersonalization",
    r"\bdissociat",
    r"\beuphori",
    r"\bsynesthesia",
    r"\btrippy|tripping",
    r"\bmild tachycard",
    r"\bjaw\s*tremor",
]

NONE_HINTS = [  # Hints at non-festival medical pathology
    r"\bappendic",
    r"\bcholecyst|gallstone|biliary",
    r"\bpancreatit",
    r"\bdiverticul",
    r"\bpyelonephr|UTI|cystitis",
    r"\bpneumonia|COPD|asthma exacerbation",
    r"\bSTEMI|NSTEMI|ACS|myocardial infarc",
    r"\bstroke|CVA|TIA",
    r"\bDKA|diabetic ketoacidosis",
    r"\bsepsis|septic shock",
    r"\bfracture|laceration|MVA|motor vehicle|fall from",
    r"\bmigraine",
    r"\bgastroenter|c-?diff|c\.\s*diff",
    r"\bpregnan|ectopic",
    r"\bsickle cell",
    r"\bhyperglycemi|hypoglycemi",
    r"\brenal colic|kidney stone|nephrolithiasis",
    r"\banemia|hemorrhag|GI bleed",
    r"\basthma|wheezing",
    r"\bvertigo|BPPV",
    r"\bcellulit|abscess",
]

FESTIVAL_CONTEXT = [
    r"\bfestival\b",
    r"\bdance tent\b",
    r"\brave\b",
    r"\bmusic event",
    r"\bconcert",
    r"\bmedical tent",
    r"\bMDMA|ecstasy|ketamine|LSD|psilocybin|mushroom",
    r"\bdrug exposure|substance exposure|ingestion of unknown",
    r"\btox(?:-|\s)?metabolic",
]


def hits(field_text: str, patterns: list[str]) -> int:
    """Count distinct pattern hits within a single field."""
    if not field_text:
        return 0
    n = 0
    for p in patterns:
        if re.search(p, field_text, re.IGNORECASE):
            n += 1
    return n


def field_evidence(rec: dict, patterns: list[str]) -> tuple[int, int]:
    """Return (total_hits, fields_with_at_least_one_hit) across narrative fields."""
    fields = [
        rec.get("chief_complaint", ""),
        rec.get("triage_brief_note", ""),
        rec.get("brief_hpi", ""),
        rec.get("hpi", ""),
        rec.get("physical_exam_pertinent_positives", ""),
        rec.get("mdm", ""),
        rec.get("clinical_course", ""),
        rec.get("ed_meds_procedures", ""),
    ]
    total = 0
    fwh = 0
    for f in fields:
        h = hits(f, patterns)
        total += h
        if h > 0:
            fwh += 1
    return total, fwh


def parse_vitals(text: str) -> dict:
    """Pull HR, RR, SBP, Temp, GCS, pupils-like cues from triage/HPI text."""
    out = {}
    m = re.search(r"\bHR\s*(\d{2,3})\b", text)
    if m:
        out["hr"] = int(m.group(1))
    m = re.search(r"\bRR\s*(\d{1,2})\b", text)
    if m:
        out["rr"] = int(m.group(1))
    m = re.search(r"\bBP\s*(\d{2,3})\s*/\s*(\d{2,3})", text)
    if m:
        out["sbp"] = int(m.group(1))
        out["dbp"] = int(m.group(2))
    m = re.search(r"\bTemp\s*([\d.]+)\s*C", text)
    if m:
        try:
            out["temp"] = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"\bGCS\s*(\d{1,2})\b", text)
    if m:
        out["gcs"] = int(m.group(1))
    m = re.search(r"\bSpO2\s*(\d{2,3})", text)
    if m:
        out["spo2"] = int(m.group(1))
    return out


def vital_evidence(vit: dict) -> dict:
    """Map raw vitals into per-class signal scores (each 0..N)."""
    k = 0  # kraken / sympathomimetic
    t = 0  # triton / sedative
    c = 0  # coral / hallucinogenic-serotonergic
    if "hr" in vit:
        if vit["hr"] >= 120:
            k += 2
        elif vit["hr"] >= 105:
            k += 1
            c += 1  # mild tachy fits coral too
        elif vit["hr"] <= 55:
            t += 2
        elif vit["hr"] <= 65:
            t += 1
    if "sbp" in vit:
        if vit["sbp"] >= 160:
            k += 1
        elif vit["sbp"] <= 95:
            t += 1
    if "rr" in vit:
        if vit["rr"] <= 10:
            t += 2
        elif vit["rr"] >= 28:
            k += 1
    if "temp" in vit:
        if vit["temp"] >= 39.5:
            k += 2
        elif vit["temp"] >= 38.5:
            k += 1
        elif vit["temp"] <= 35.5:
            t += 1
    if "gcs" in vit:
        if vit["gcs"] <= 11:
            t += 2
        elif vit["gcs"] <= 13:
            t += 1
    return {"k": k, "t": t, "c": c}


def softmax_like(scores: dict, temperature: float = 1.0) -> dict:
    """Normalize raw scores into a proper probability dist with floors."""
    import math

    keys = ["k", "t", "c", "n"]
    vals = [scores[x] / temperature for x in keys]
    mx = max(vals)
    exps = [math.exp(v - mx) for v in vals]
    s = sum(exps)
    probs = {k: e / s for k, e in zip(keys, exps)}
    return probs


def classify(rec: dict) -> dict:
    text_all = " \n ".join(
        rec.get(f, "") or ""
        for f in (
            "chief_complaint",
            "triage_brief_note",
            "brief_hpi",
            "hpi",
            "physical_exam_pertinent_positives",
            "mdm",
            "clinical_course",
            "ed_meds_procedures",
        )
    )

    # Festival context
    fest_hits = sum(
        1 for p in FESTIVAL_CONTEXT if re.search(p, text_all, re.IGNORECASE)
    )
    festival = fest_hits >= 1

    # Drug-class lexical hits
    k_total, k_fwh = field_evidence(rec, KRAKEN)
    t_total, t_fwh = field_evidence(rec, TRITON)
    c_total, c_fwh = field_evidence(rec, CORAL)
    n_total, n_fwh = field_evidence(rec, NONE_HINTS)

    # Vitals
    vit = parse_vitals(text_all)
    vev = vital_evidence(vit)

    # Raw composite scores (mostly count corroboration via fields-with-hit)
    k_score = 0.7 * k_total + 1.0 * k_fwh + 0.8 * vev["k"]
    t_score = 0.7 * t_total + 1.0 * t_fwh + 1.0 * vev["t"]
    c_score = 0.7 * c_total + 1.0 * c_fwh + 0.5 * vev["c"]
    n_score = 0.8 * n_total + 1.2 * n_fwh

    # Baseline prior: if no festival context AND no toxidrome signal -> default to none
    any_drug_signal = (k_fwh + t_fwh + c_fwh) >= 1 or any(
        v > 0 for v in vev.values()
    )

    # ---- Bayesian-flavored mixing -------------------------------------
    # Build a base "uncertain" distribution and add evidence on top.
    # Default soft prior favors none mildly because most ED cases are medical.
    base = {"k": 0.18, "t": 0.18, "c": 0.18, "n": 0.46}

    # If festival context is strong and there's no medical confounder, flatten
    # base toward (0.3, 0.3, 0.3, 0.1).
    if festival and n_fwh == 0:
        base = {"k": 0.30, "t": 0.30, "c": 0.30, "n": 0.10}
    elif festival and n_fwh >= 1:
        base = {"k": 0.22, "t": 0.22, "c": 0.22, "n": 0.34}

    # Evidence increments
    evidence = {
        "k": k_score,
        "t": t_score,
        "c": c_score,
        "n": n_score,
    }

    # Compute scaled boost: each unit of evidence adds a small bump.
    # Use a low gain to keep distributions soft.
    GAIN = 0.18
    boosted = {cls: base[cls] + GAIN * evidence[cls] for cls in base}

    # Clip and renormalize
    total = sum(boosted.values())
    probs = {cls: v / total for cls, v in boosted.items()}

    # --- Conservative caps -----------------------------------------------
    # Reserve p > 0.85 only for cases with corroboration from >=2 fields
    # AND vital evidence pointing the same way.
    corroborated = {
        "k": k_fwh >= 2 and vev["k"] >= 2,
        "t": t_fwh >= 2 and vev["t"] >= 2,
        "c": c_fwh >= 2,  # vitals for coral are softer
        "n": n_fwh >= 2 and not festival,
    }

    cap = 0.85
    high_cap = 0.92
    for cls in ("k", "t", "c", "n"):
        if probs[cls] > cap and not corroborated[cls]:
            probs[cls] = cap
        if probs[cls] > high_cap:
            probs[cls] = high_cap

    # If nothing pointed anywhere and no festival -> mostly none but soft
    if not any_drug_signal and not festival and n_fwh == 0:
        probs = {"k": 0.18, "t": 0.18, "c": 0.18, "n": 0.46}

    # If absolutely no signal at all (very rare) -> flat uncertain
    if (
        not any_drug_signal
        and not festival
        and n_fwh == 0
        and not vit
    ):
        probs = {"k": 0.25, "t": 0.25, "c": 0.25, "n": 0.25}

    # If clear non-festival medical (>=2 medical hits, no festival, no drug
    # signal), allow p_none up to 0.88 cap.
    if n_fwh >= 2 and not festival and (k_fwh + t_fwh + c_fwh) == 0:
        # Concentrate mass on none but stay soft.
        probs = {"k": 0.04, "t": 0.04, "c": 0.04, "n": 0.88}

    # Ambiguous-pair handling: if two classes are within 20% of each other
    # AND both have >=1 field hit, split mass between them.
    drug_probs = {"k": probs["k"], "t": probs["t"], "c": probs["c"]}
    top2 = sorted(drug_probs.items(), key=lambda x: -x[1])
    if (
        top2[0][1] > 0.30
        and top2[1][1] > 0.0
        and top2[0][1] - top2[1][1] < 0.12
    ):
        # check both have field-hit support
        fwh_map = {"k": k_fwh, "t": t_fwh, "c": c_fwh}
        a, b = top2[0][0], top2[1][0]
        if fwh_map[a] >= 1 and fwh_map[b] >= 1:
            avg = (probs[a] + probs[b]) / 2
            probs[a] = avg
            probs[b] = avg

    # Renormalize to be safe
    total = sum(probs.values())
    probs = {cls: v / total for cls, v in probs.items()}

    return probs


def main() -> None:
    rows = []
    with IN.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = classify(rec)
            rows.append(
                (
                    rec["encounter_id"],
                    round(p["k"], 3),
                    round(p["t"], 3),
                    round(p["c"], 3),
                    round(p["n"], 3),
                )
            )

    # Adjust rounding drift so each row sums to exactly 1.000
    adjusted = []
    for eid, pk, pt, pc, pn in rows:
        s = pk + pt + pc + pn
        diff = round(1.0 - s, 3)
        if abs(diff) > 0:
            # Add the residual to the largest probability
            arr = [pk, pt, pc, pn]
            idx = arr.index(max(arr))
            arr[idx] = round(arr[idx] + diff, 3)
            pk, pt, pc, pn = arr
        adjusted.append((eid, pk, pt, pc, pn))

    # Validation
    bad = 0
    for eid, pk, pt, pc, pn in adjusted:
        s = pk + pt + pc + pn
        if abs(s - 1.0) > 0.005:
            bad += 1

    # Marginal means
    n = len(adjusted)
    mk = sum(r[1] for r in adjusted) / n
    mt = sum(r[2] for r in adjusted) / n
    mc = sum(r[3] for r in adjusted) / n
    mn = sum(r[4] for r in adjusted) / n

    n_none_gt_05 = sum(1 for r in adjusted if r[4] > 0.5)
    n_soft = sum(1 for r in adjusted if max(r[1:]) < 0.5)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        f.write("encounter_id,p_kraken,p_triton,p_coral,p_none\n")
        for eid, pk, pt, pc, pn in adjusted:
            f.write(f"{eid},{pk:.3f},{pt:.3f},{pc:.3f},{pn:.3f}\n")

    print(f"OUT_PATH={OUT}")
    print(f"RECORDS={n}")
    print(f"SUM_VALID_WITHIN_005={n - bad}/{n}")
    print(
        f"MEANS: kraken={mk:.4f} triton={mt:.4f} coral={mc:.4f} none={mn:.4f}"
    )
    print(f"P_NONE_GT_0.5_COUNT={n_none_gt_05}")
    print(f"SOFT_ROWS_MAX_LT_0.5={n_soft}")


if __name__ == "__main__":
    main()
