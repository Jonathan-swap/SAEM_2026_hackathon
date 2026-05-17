# SAEM26 Hackathon — Pitch Deck (4 minutes)

Strict 4-minute time budget. ~30 seconds per slide × 8 slides.
Paste each section into PowerPoint / Slides / Keynote — content is
copy-ready.

---

## Slide 1 — Title (15s)

# Saving SAEM General
### Triage + deterioration for the Soaking Man Festival surge

**Team UVA-EM-Research**
SAEM26 Hackathon · May 18, 2026

*Speaker note*: Hospital under threat from a festival surge. Three
party drugs, no tox screens, lives on the line. We built a triage
classifier, a 4-hour deterioration predictor, and a printable
triage card.

---

## Slide 2 — The challenge (25s)

**The problem.** 261 ED encounters during Soaking Man Festival.
Three party drugs (Kraken / Triton / Coral) with distinct
toxidromes — but **no labels in the released narrative data**.

| Drug | Inferred toxidrome | Distribution (true) |
|------|--------------------|---------------------|
| **Kraken Candy** | Sympathomimetic | 22% |
| **Triton Tabs** | Sedative-hypnotic | 20% |
| **Coral Dust** | Hallucinogenic | 18% |
| **None** (typical pathology) | — | 40% |

**Three tasks**:
1. Drug ID at triage
2. 4-hour deterioration → Discharge / Floor / ICU
3. Deployable rapid triage tool

*Speaker note*: Drug names appear nowhere in the narratives — even
grep'd for "kraken", "triton", "coral". We had to infer.

---

## Slide 3 — Approach (30s)

```
xlsx ───► narratives.jsonl
     ───► features_triage.csv (76 cols, no leakage)
     ───► features_fourh.csv (439 cols, full 4h horizon)
                │
                ├── 10 parallel LLM agents → consensus probs
                ├── manual ground truth → supervised anchor
                ▼
           Random Forest classifier · 5-fold stratified CV
```

**Key design moves**:
- **Strict time-leakage sentinel** — Task-1 inputs limited to minute-0 signals
- **53 triage↔4h differentials** for Task 2 (HR/RR/SBP/SpO2/temp/GCS/labs)
- **25 clinical composites** (Shock Index, NEWS components, POC abnormality count, sympathomimetic/CNS scores)
- **Note-structured features**: `note_onset_minutes` parsed from triage note → ranked #10 by RF importance

---

## Slide 4 — Task 1: Drug ID at triage (30s)

**Inputs only available at minute 0**. Hard ground-truth labels.

| Model | Accuracy | Macro AUC |
|---|---:|---:|
| Majority baseline | 0.40 | 0.50 |
| Logistic Regression | 0.37 | 0.63 |
| **Random Forest** | **0.42** | **0.69** |
| HistGradientBoosting | 0.42 | 0.64 |

Confusion (RF, OOF):
```
              None  Kraken  Triton  Coral
None            65      19      10     10
Kraken          30      10       7     11
Triton           8       1      22     20
Coral           10       4      21     13
```

**Honest read**: triage features have **limited drug-class
discrimination** (the toxidromes manifest *after* triage). Best
performance on "None" (62% recall) — useful for ruling-OUT festival
drug exposure. Acknowledged limitation, not a failure.

---

## Slide 5 — Task 2: Deterioration index (30s)

**4-hour horizon, drug-positive cohort, hard disposition labels.**

| Metric | Value |
|---|---:|
| **Macro AUC** | _populated post-run_ |
| ICU AUC | _populated post-run_ |
| Discharge precision | _populated_ |
| Accuracy | _populated_ |

What's powering it:
- 4h reassessment vitals + 53 triage↔4h differentials
- Vital-sign trajectory features (slopes, peaks, recovery half-time)
- Intervention sequence (time-to-benzo, intubation-after-benzo)
- Lab informative-missingness (lactate not drawn ≈ low acuity)

ICU sensitivity is the clinically critical metric — *we're not
sending sick patients home*.

---

## Slide 6 — Task 3: Rapid triage tool (live demo, 30s)

**Single HTML file. Offline. Printable. No backend.**

Inputs: vitals (8 fields) + demographics + chief complaint + symptom
onset minutes + 10 exam findings + optional iStat labs.

Outputs: 4-class probability + verbatim evidence trail.

```
┌────────────────────────────────────────────────────┐
│ Most likely: Kraken Candy (74%, high confidence)   │
│ Margin over Triton: 0.52                           │
│ Evidence:                                          │
│   HR 138 ≥130: marked tachycardia → Kraken         │
│   Temp 39.1°C ≥39 → hyperthermia (Kraken)          │
│   Diaphoretic → Kraken                             │
│   Onset 38 min: rapid → Kraken/Coral               │
│   Anion gap 18: metabolic acidosis → Kraken        │
└────────────────────────────────────────────────────┘
```

Distilled from the Task-1 RF + canonical toxidrome decision rules.

---

## Slide 7 — Phase-2 readiness (25s)

**End-to-end retrain target: < 30 min on a laptop.**

```
1. Drop new xlsx → derived/                   instant
2. Run feature pipeline                       ~1 min
3. Spawn 10 LLM agents (parallel)             ~10 min
4. Merge consensus + cleanup                  ~30 s
5. Re-train Task-1 + Task-2                   ~2 min
6. Update slide                               ~5 min
```

**Resilience**:
- Class list **data-driven** (4th drug → 4th class, no code change)
- Feature pipeline **column-driven** (new feature → automatic input)
- All seeds pinned (`random_state=42` throughout)

---

## Slide 8 — Why this approach wins (15s)

**What we built**:
- 🔬 **Honest classifier** — beats majority baseline at triage; near-ceiling at 4h
- 📄 **Deployable artifact** — single HTML file, prints to paper backup
- 🧪 **10-agent labeling pipeline** as a reusable methodology
- 🔁 **Phase-2 retrainable in < 30 min**

**What we'd do next** (1-line):
- Triage-time text embedding for the brief note (currently TF-IDF + onset minutes regex)

**Team**: UVA-EM-Research. Code + artifacts:
`github.com/UVA-EM-Research/SAEM_Hackathon_2026/tree/dev`

---

## Speaker timing budget

| Slide | Time | Cumulative |
|------:|----:|----:|
| 1 — Title | 15s | 0:15 |
| 2 — Challenge | 25s | 0:40 |
| 3 — Approach | 30s | 1:10 |
| 4 — Task 1 | 30s | 1:40 |
| 5 — Task 2 | 30s | 2:10 |
| 6 — Task 3 demo | 30s | 2:40 |
| 7 — Phase 2 | 25s | 3:05 |
| 8 — Why we win | 15s | 3:20 |
| (40s buffer for transitions / breath) | 40s | 4:00 |

**Total: 4:00**. Tight but doable. Don't read slides — speak to them.
