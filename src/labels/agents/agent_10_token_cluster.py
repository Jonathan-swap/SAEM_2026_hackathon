"""Probs_10: Bottom-up token-cluster prevalence classifier.

Approach: treat physical_exam_pertinent_positives as a bag of tokens,
compute Jaccard-like similarity to three pre-defined reference clusters
(K=Kraken/stimulant, T=Triton/sedative, C=Coral/hallucinogen), and any
out-of-cluster tokens (plus medical-diagnosis keywords in MDM/HPI) boost
p_none. Probabilities = softmax over similarity scores with uniform priors.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(r"C:\Users\rs3te\Work\Claude-safe\SAEM-Hackathon")
INPUT = ROOT / "derived" / "narratives.jsonl"
OUTPUT = ROOT / "derived" / "probs_10.csv"

# Reference clusters (lowercase, underscore-normalized).
CLUSTER_K = {
    "tachycardic", "diaphoretic", "mild_tremor", "restless", "agitated",
    "hyperthermic", "mydriasis", "anxious", "hypertensive", "flushed",
}
CLUSTER_T = {
    "slow_responses", "reduced_tracking", "intermittent_disorientation",
    "distractible", "somnolent", "bradypneic", "hypotonic", "miosis",
    "low_gcs", "lethargic", "fatigued_appearance",
}
CLUSTER_C = {
    "ataxia", "unsteady_gait", "dry_mucosa", "hyperreflexic",
    "perceptual_distortion", "tremor_fine", "intermittent_hallucinations",
}

# Medical-diagnosis keyword patterns to boost None.
MED_PATTERNS = [
    r"chest[\s_]+pain[\s_]*cardiac",
    r"focal[\s_]+neuro",
    r"abd[\s_]+tenderness",
    r"abdominal[\s_]+tenderness",
    r"fever[\s_]+localized[\s_]+source",
    r"dyspnea[\s_]+pulmonary",
    r"focal[\s_]+infection",
    r"appendicitis",
    r"pneumonia",
    r"sepsis",
    r"stroke",
    r"myocardial[\s_]+infarction",
    r"cholecystitis",
    r"pyelonephritis",
    r"gi[\s_]+bleed",
    r"diabetic[\s_]+ketoacidosis",
    r"dka\b",
    r"pulmonary[\s_]+embolism",
    r"meningitis",
]
MED_RE = re.compile("|".join(MED_PATTERNS), re.IGNORECASE)


def tokenize_pe(pe: str) -> set[str]:
    """Split pertinent-positives bag by ; , or whitespace; normalize."""
    if not pe:
        return set()
    parts = re.split(r"[;,]+", pe)
    out: set[str] = set()
    for p in parts:
        t = p.strip().lower().replace(" ", "_").replace("-", "_")
        if t:
            out.add(t)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def classify(rec: dict) -> tuple[float, float, float, float]:
    pe_tokens = tokenize_pe(rec.get("physical_exam_pertinent_positives", ""))

    # Vague/empty fallback.
    if not pe_tokens:
        return (0.20, 0.20, 0.20, 0.40)

    # Jaccard-like similarities (count overlap / size-of-cluster-union-tokens).
    sim_k = jaccard(pe_tokens, CLUSTER_K)
    sim_t = jaccard(pe_tokens, CLUSTER_T)
    sim_c = jaccard(pe_tokens, CLUSTER_C)

    # Out-of-cluster tokens -> evidence of "something else" -> none.
    in_any = CLUSTER_K | CLUSTER_T | CLUSTER_C
    out_of_cluster = pe_tokens - in_any
    # None similarity: fraction of tokens that fall outside all drug clusters.
    sim_n_tokens = len(out_of_cluster) / len(pe_tokens)

    # Medical-diagnosis keyword boost from MDM + HPI.
    text = " ".join(
        str(rec.get(k, "") or "")
        for k in ("mdm", "hpi", "brief_hpi", "clinical_course")
    )
    n_med_hits = len(MED_RE.findall(text))
    med_boost = min(n_med_hits * 0.15, 0.6)

    sim_n = sim_n_tokens + med_boost

    # Uniform priors: 0.10 per drug class, 0.15 for None.
    prior_k = math.log(0.10)
    prior_t = math.log(0.10)
    prior_c = math.log(0.10)
    prior_n = math.log(0.15)

    # Scale similarity to a logit (multiplier puts scores in usable softmax range).
    SCALE = 4.0
    logits = [
        prior_k + SCALE * sim_k,
        prior_t + SCALE * sim_t,
        prior_c + SCALE * sim_c,
        prior_n + SCALE * sim_n,
    ]

    return tuple(softmax(logits))  # type: ignore[return-value]


def main() -> None:
    records: list[dict] = []
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in records:
        eid = rec.get("encounter_id", "")
        p_k, p_t, p_c, p_n = classify(rec)
        rows.append((eid, p_k, p_t, p_c, p_n))

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encounter_id", "p_kraken", "p_triton", "p_coral", "p_none"])
        for eid, pk, pt, pc, pn in rows:
            w.writerow([eid, f"{pk:.6f}", f"{pt:.6f}", f"{pc:.6f}", f"{pn:.6f}"])

    # Validation report.
    n = len(rows)
    sums = [pk + pt + pc + pn for _, pk, pt, pc, pn in rows]
    bad = sum(1 for s in sums if abs(s - 1.0) > 0.005)
    mean_k = sum(r[1] for r in rows) / n
    mean_t = sum(r[2] for r in rows) / n
    mean_c = sum(r[3] for r in rows) / n
    mean_n = sum(r[4] for r in rows) / n
    n_high_none = sum(1 for r in rows if r[4] > 0.5)

    print(f"path: {OUTPUT}")
    print(f"record_count: {n}")
    print(f"rows_failing_sum_check: {bad}")
    print(f"mean p_kraken: {mean_k:.4f}")
    print(f"mean p_triton: {mean_t:.4f}")
    print(f"mean p_coral:  {mean_c:.4f}")
    print(f"mean p_none:   {mean_n:.4f}")
    print(f"p_none > 0.5 count: {n_high_none}")


if __name__ == "__main__":
    main()
