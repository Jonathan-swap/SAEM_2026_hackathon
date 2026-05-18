"""Run 10 independent 5-class scoring lenses on the Phase-2 dataset.

This is the v7 / Phase-2 entry point for the "10-agent" ensemble. The
original 10 agent scripts (agent_01..10) each had bespoke 4-class
narrative rubrics with inconsistent (and now-broken) hardcoded paths
and assumed Phase-1 labels existed. For the Phase-2 twist — a brand
new drug "Siren Spark" added as a 5th class with **no ground truth
and no published clinical profile** — we re-cast the ensemble as 10
lenses inside this single driver:

  1. Equal weighting across all narrative fields.
  2. Toxidrome-led — PE tokens + peak labs (features).
  3. HPI-led — defining-token search in brief_hpi + hpi.
  4. MDM-led — focus on mdm + clinical_course (boilerplate dampened).
  5. Conservative / Bayesian — softer scores, more uncertainty mass
     routed to Siren Spark.
  6. Treatment-response — emphasis on ed_meds_procedures (procedures
     are strong class signals).
  7. Autonomic / vitals-only — narrative ignored; relies on the
     structured features file.
  8. Counterfactual / negative-evidence — high weight on absence of
     known signals (so weakly-fit rows → Siren Spark).
  9. Recovery-pattern pharmacokinetics — clinical_course emphasis.
  10. Token-cluster — chief_complaint + triage_brief_note only
      (minute-0 lens).

Each lens produces 4-class scores for {None, Kraken, Triton, Coral}
using a shared regex lexicon, then extends to 5 classes by calling
the shared `add_siren_spark` utility (which routes residual mass to
Siren Spark based on confidence + feature anomaly vs Phase-1).

Inputs (Phase-2):
  derived/phase2/narratives_fourh.jsonl   — 139 records
  derived/phase2/features_fourh.csv       — feature anomaly signal
  derived/features_triage.csv             — Phase-1 reference stats

Outputs:
  derived/phase2/probs_1.csv .. derived/phase2/probs_10.csv
    columns: encounter_id, p_none, p_kraken, p_triton, p_coral,
             p_siren_spark   (rows sum to 1.0)

Run:
  .venv/Scripts/python.exe src/labels/agents/run_5class_agents.py

Then merge with src/labels/merge_probabilities.py.
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _siren_spark import add_siren_spark  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DERIVED = ROOT / "derived"
PHASE2 = DERIVED / "phase2"

NARR_FIELDS = ["brief_hpi", "hpi", "physical_exam_pertinent_positives",
               "mdm", "clinical_course", "ed_meds_procedures",
               "triage_brief_note", "chief_complaint"]


# ---------------------------------------------------------------------
# Shared lexicons (derived from agent_01 with PRE-READ v6 adjustments)
# ---------------------------------------------------------------------

KRAKEN_PATTERNS: list[tuple[str, float]] = [
    (r"\bdiaphoretic\b", 2.2), (r"\brestless\b", 1.6),
    (r"\btachycardic\b", 2.0), (r"\bagitated\b", 2.0),
    (r"\btremulous\b|\bmild_tremor\b|\bmild\s+tremor\b", 1.8),
    (r"\bhyperthermic\b", 2.2), (r"\bflushed\b", 1.0),
    (r"\bmydriatic|dilated_pupil", 2.0),
    (r"\bhypertensive\b", 1.5),
    (r"\bdiaphores", 2.0), (r"\bagitation\b", 1.0),
    (r"\bmarked\s+restless", 1.5),
    (r"\bsympathetic\s+excess", 0.4),
    (r"\bstimulant|sympathomimetic", 2.5),
    (r"\brapid\s+heartbeat|palpitation", 1.0),
    (r"\bbruxism|teeth\s+grinding|jaw\s+clench", 2.0),
    (r"\bhyperactiv", 1.2), (r"\brhabdo", 2.0),
    (r"\bseizure", 0.8),
    (r"hr\s*1[3-9]\d|hr\s*[2-9]\d{2}", 1.6),
    (r"temp\s*3[89]\.[0-9]|temp\s*4\d", 1.6),
    (r"bp\s*1[6-9]\d|bp\s*2\d{2}", 1.0),
    (r"\bcooling\b", 0.3), (r"\bantipyretic\s+therapy", 0.2),
    (r"\bbenzodiazepine\s+for\s+agitation", 0.3),
    (r"\bpeak\s+lactate\s+5|peak_lactate_5plus", 1.5),
    (r"\bpeak\s+cpk|peak_cpk_1000plus", 1.5),
    (r"\bpeak\s+troponin", 1.0),
]

TRITON_PATTERNS: list[tuple[str, float]] = [
    (r"\bdistractible\b", 1.6),
    (r"\bslow_responses\b|\bslow\s+responses\b", 2.2),
    (r"\breduced_tracking\b", 2.0),
    (r"\bintermittent_disorientation\b", 1.6),
    (r"\btachypneic_effort\b|\btachypneic\s+effort\b", 1.4),
    (r"\bsomnolent|drowsy|lethargic|obtunded?\b", 2.2),
    (r"\bunresponsive\b", 2.0),
    (r"\bringing\s+in\s+ears|tinnitus", 2.5),
    (r"\bpalpitation|cardiac\s+awareness", 2.0),
    (r"\bsubjective\s+tachycardia", 2.0),
    (r"\blow\s+gcs|gcs\s*[3-9]\b|gcs\s*1[0-1]\b", 1.6),
    (r"\bcns\s+depression|respiratory\s+depression", 1.8),
    (r"\bsedat(ed|ion|ive)", 1.8),
    (r"\bnodding\b|nodded\s+off", 1.4),
    (r"\bspeaker|dance\s+tent|dj\s+area|sound\s+system", 1.5),
    (r"\bendotracheal\s+intubat", 1.5),
    (r"\bnoninvasive\s+positive\s+pressure|nippv|bipap|cpap", 1.5),
    (r"\bbenzo\b|\bdepressant", 1.8),
    (r"hr\s*[3-5]\d\b", 1.2),
]

CORAL_PATTERNS: list[tuple[str, float]] = [
    (r"\bataxia\b", 2.2),
    (r"\bunsteady_gait\b|\bunsteady\s+gait\b", 2.0),
    (r"\bdry_mucosa\b", 1.4),
    (r"\bmildly_tachycardic\b|\bmild\s+tachy", 1.0),
    (r"\bfatigued_appearance\b", 0.3),
    (r"\bperceptual\s+(change|distort|disturb|alter)", 2.5),
    (r"\bvisual\s+(disturb|halluc|distort)", 2.2),
    (r"\bhallucinat", 2.5),
    (r"\btime[-\s]?distort|time\s+dilation", 2.5),
    (r"\bsynesth", 2.5),
    (r"\bderealiz|depersonaliz|dissociat", 2.0),
    (r"\bwave[-\s]like|wavy\s+vision|kaleidoscop|vivid", 2.5),
    (r"\bspatial\s+(disorient|distort)|internal\s+unease", 2.5),
    (r"\bwaxing\s+and\s+waning|wax\s+and\s+wane", 1.8),
    (r"\bhallucinogen|psychedelic", 2.5),
    (r"\bunsteady\b", 0.8),
    (r"\bdim/?quiet\s+room|quiet\s+dim|dim\s+room", 2.2),
    (r"\breef[-\s]?colored\s+powder|insufflat", 2.5),
]

NONE_PATTERNS: list[tuple[str, float]] = [
    (r"\bc-?diff|cellulit|appendicit|cholecystit|pancreatit", 2.5),
    (r"\bdiverticulit|pyelonephrit|pneumon", 2.0),
    (r"\bdiabetic\s+ketoacid|dka\b", 2.5),
    (r"\bsepsis\b|septic\s+shock", 2.0),
    (r"\bstroke\b|\bcva\b|tia\b", 2.5),
    (r"\bmyocardial\s+infarct|stemi|nstemi|\bacs\b", 2.5),
    (r"\bpulmonary\s+embol|\bpe\b", 2.0),
    (r"\bgi\s+bleed|hematemesis|melena", 2.2),
    (r"\bkidney\s+stone|nephrolithia|renal\s+colic", 2.0),
    (r"\bectopic\s+pregnan|vaginal\s+bleed", 1.8),
    (r"\bfracture|laceration", 1.5),
    (r"\bdental\s+pain|toothache", 2.0),
    (r"\botitis|sinusit|pharyngit|tonsillit", 1.8),
    (r"\bvertigo|bppv\b", 1.6),
    (r"\bnon[-\s]festival|unrelated\s+to\s+festival", 3.0),
    (r"\bprimary\s+medical\s+pathology", 2.5),
    # Negative (suggest festival drug):
    (r"\bfestival\b", -1.5),
    (r"\bsubstance\s+exposure|drug\s+ingest", -1.8),
    (r"\btox[-\s]?metabolic", -2.0),
    (r"\bnaloxone|flumazenil|targeted\s+reversal", -1.5),
]


# ---------------------------------------------------------------------
# Generic scoring core
# ---------------------------------------------------------------------

def _compile(patterns: list[tuple[str, float]]) -> list[tuple[re.Pattern, float]]:
    return [(re.compile(p, re.IGNORECASE), w) for p, w in patterns]


_KP = _compile(KRAKEN_PATTERNS)
_TP = _compile(TRITON_PATTERNS)
_CP = _compile(CORAL_PATTERNS)
_NP = _compile(NONE_PATTERNS)


def score_text(text: str, compiled: list[tuple[re.Pattern, float]]) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    s = 0.0
    for rx, w in compiled:
        n_hits = len(rx.findall(text))
        if n_hits:
            s += w * min(n_hits, 3)
    return s


def lens_score_4class(rec: dict, field_weights: dict[str, float]
                       ) -> np.ndarray:
    """Apply per-field weights, score each class, return (4,) raw scores."""
    raw = np.zeros(4, dtype=float)  # [None, Kraken, Triton, Coral]
    for field, fw in field_weights.items():
        text = rec.get(field, "") or ""
        if not text or fw == 0:
            continue
        raw[0] += fw * score_text(text, _NP)
        raw[1] += fw * score_text(text, _KP)
        raw[2] += fw * score_text(text, _TP)
        raw[3] += fw * score_text(text, _CP)
    return raw


def softmax_with_floor(raw: np.ndarray, temperature: float = 1.0,
                        floor: float = 0.02) -> np.ndarray:
    """Stable softmax with a per-class floor so no class collapses to
    zero (matters for the Siren Spark scoring downstream)."""
    z = raw / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    p = e / e.sum()
    p = np.maximum(p, floor)
    p = p / p.sum()
    return p


# ---------------------------------------------------------------------
# 10 lens definitions
# ---------------------------------------------------------------------

@dataclass
class Lens:
    n: int
    name: str
    field_weights: dict[str, float]
    temperature: float
    # Siren Spark knobs:
    alpha: float       # weight on confidence vs feature-anomaly signal
    sharpness: float   # how aggressively to expand p_siren
    cap: float         # max allowed p_siren per row


LENSES: list[Lens] = [
    Lens(1, "equal_weighting",
         {f: 1.0 for f in NARR_FIELDS}, temperature=1.0,
         alpha=0.6, sharpness=1.0, cap=0.65),
    Lens(2, "toxidrome_pe_labs",
         {"physical_exam_pertinent_positives": 2.5,
          "hpi": 1.0, "mdm": 1.0, "clinical_course": 0.5,
          "triage_brief_note": 0.7, "chief_complaint": 0.5,
          "brief_hpi": 0.5, "ed_meds_procedures": 0.5},
         temperature=0.9,
         alpha=0.3, sharpness=1.1, cap=0.70),
    Lens(3, "hpi_led",
         {"hpi": 2.5, "brief_hpi": 1.5,
          "chief_complaint": 0.7, "triage_brief_note": 0.7,
          "physical_exam_pertinent_positives": 0.5, "mdm": 0.5,
          "clinical_course": 0.3, "ed_meds_procedures": 0.3},
         temperature=1.0,
         alpha=0.6, sharpness=1.0, cap=0.65),
    Lens(4, "mdm_led_debolierplated",
         {"mdm": 2.0, "clinical_course": 1.5,
          "physical_exam_pertinent_positives": 0.8,
          "hpi": 0.5, "brief_hpi": 0.5,
          "triage_brief_note": 0.3, "chief_complaint": 0.3,
          "ed_meds_procedures": 0.3},
         temperature=1.2,  # softer: MDM has more boilerplate
         alpha=0.5, sharpness=1.0, cap=0.65),
    Lens(5, "conservative_bayesian",
         {f: 1.0 for f in NARR_FIELDS},
         temperature=1.6,  # softer everywhere
         alpha=0.7, sharpness=1.3, cap=0.75),  # more Siren mass
    Lens(6, "treatment_response",
         {"ed_meds_procedures": 3.0, "clinical_course": 1.5,
          "mdm": 0.8, "hpi": 0.6, "brief_hpi": 0.5,
          "physical_exam_pertinent_positives": 0.8,
          "triage_brief_note": 0.4, "chief_complaint": 0.3},
         temperature=1.0,
         alpha=0.5, sharpness=1.0, cap=0.65),
    Lens(7, "vitals_only",
         {f: 0.0 for f in NARR_FIELDS},  # narrative ignored
         temperature=1.0,
         alpha=0.0, sharpness=1.3, cap=0.75),  # Siren signal purely from features
    Lens(8, "counterfactual",
         {f: 1.0 for f in NARR_FIELDS},
         temperature=0.7,  # sharper 4-class confidence
         alpha=0.85, sharpness=1.4, cap=0.80),  # weakly-fit rows -> high Siren
    Lens(9, "recovery_pattern_pk",
         {"clinical_course": 2.5, "ed_meds_procedures": 1.5,
          "mdm": 0.8, "hpi": 0.6, "brief_hpi": 0.5,
          "physical_exam_pertinent_positives": 0.5,
          "triage_brief_note": 0.4, "chief_complaint": 0.3},
         temperature=1.0,
         alpha=0.5, sharpness=1.0, cap=0.65),
    Lens(10, "minute0_tokens",
         {"chief_complaint": 2.5, "triage_brief_note": 2.0,
          "hpi": 0.4, "brief_hpi": 0.4,
          "physical_exam_pertinent_positives": 0.3, "mdm": 0.3,
          "clinical_course": 0.2, "ed_meds_procedures": 0.2},
         temperature=1.0,
         alpha=0.6, sharpness=1.0, cap=0.65),
]


# ---------------------------------------------------------------------
# Phase-2 driver
# ---------------------------------------------------------------------

def _load_narratives(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _features_df_for_anomaly(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def run() -> None:
    narr_path = PHASE2 / "narratives_fourh.jsonl"
    feat_path = PHASE2 / "features_fourh.csv"
    if not narr_path.exists():
        raise SystemExit(f"missing {narr_path} — run the narratives extractor first")

    records = _load_narratives(narr_path)
    print(f"Loaded {len(records)} Phase-2 narrative records")

    features = _features_df_for_anomaly(feat_path)
    if features is None:
        print(f"WARN: {feat_path} not present — Siren Spark anomaly signal disabled "
              "(only the confidence signal will be used)")

    PHASE2.mkdir(parents=True, exist_ok=True)

    for lens in LENSES:
        print(f"\n--- Lens {lens.n}: {lens.name} ---")
        rows = []
        for rec in records:
            raw = lens_score_4class(rec, lens.field_weights)
            p4 = softmax_with_floor(raw, temperature=lens.temperature)
            rows.append({"encounter_id": rec["encounter_id"],
                          "p_none": p4[0], "p_kraken": p4[1],
                          "p_triton": p4[2], "p_coral": p4[3]})
        probs_4class = pd.DataFrame(rows)
        probs_5class = add_siren_spark(
            probs_4class,
            features_df=features,
            alpha=lens.alpha, sharpness=lens.sharpness, cap=lens.cap,
            phase1_features_path=DERIVED / "features_triage.csv",
        )
        out_path = PHASE2 / f"probs_{lens.n}.csv"
        probs_5class.to_csv(out_path, index=False)
        # Class-distribution summary (argmax)
        cols = ["p_none", "p_kraken", "p_triton", "p_coral", "p_siren_spark"]
        names = ["None", "Kraken", "Triton", "Coral", "Siren Spark"]
        argmax = probs_5class[cols].to_numpy().argmax(axis=1)
        dist = {names[i]: int((argmax == i).sum()) for i in range(5)}
        print(f"  argmax distribution: {dist}")
        print(f"  mean p_siren: {probs_5class['p_siren_spark'].mean():.3f}  "
              f"max p_siren: {probs_5class['p_siren_spark'].max():.3f}")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    run()
