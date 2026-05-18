# Subagent Prompts — Probability Extraction (10 Agents, v6-aligned)

These are the exact prompts sent to ten independent Claude Code
subagents (Opus model) that produced `derived/probs_1.csv` through
`derived/probs_10.csv`. Each agent runs in its own context with no
visibility into the others' outputs; agreement across them is the
cross-validation signal.

All ten use the same fixed drug → toxidrome mapping for cross-
comparability when averaging. They differ in **which fields they
consult** (Agents 1–5) and **what reasoning paradigm they apply**
(Agents 6–10).

| Agent | Lens | Output |
|------:|------|--------|
| 1 | Equal weighting across all narrative fields | `probs_1.csv` |
| 2 | Toxidrome-led — PE tokens + peak labs | `probs_2.csv` |
| 3 | HPI-led — defining-token search | `probs_3.csv` |
| 4 | MDM minus boilerplate — differential + severity tier only | `probs_4.csv` |
| 5 | Conservative / Bayesian — quantified uncertainty | `probs_5.csv` |
| 6 | Treatment-response trajectory (de-boilerplated) | `probs_6.csv` |
| 7 | Autonomic-only decision tree (no pupils in dataset) | `probs_7.csv` |
| 8 | Counterfactual / negative-evidence | `probs_8.csv` |
| 9 | Recovery-pattern pharmacokinetics | `probs_9.csv` |
| 10 | v6 PE-token + peak-lab cluster matching | `probs_10.csv` |

Each agent emits a CSV with columns `encounter_id, p_kraken,
p_triton, p_coral, p_none` for all 261 records; every row sums to
1.0 (±0.005).

`src/labels/merge_probabilities.py` auto-discovers all
`probs_<N>.csv` files, averages them, and writes the consensus to
`derived/probs_avg.csv`.

---

## v6 PRE-READ — REQUIRED FOR ALL AGENTS

This block must precede every agent's reasoning. It supersedes the
old textbook toxidrome mapping and the original PROMPTS.md
(pre-v6).

### Fixed drug → toxidrome mapping (v6, evidence-anchored)

- **Kraken Candy** → **Sympathomimetic** (PCP / amphetamine-like).
  Outward, explosive, motor-dominant. **Defining**: agitation as
  chief complaint, marked restlessness / impulsive movement, rapid
  escalation. PE findings: diaphoretic (p=0.001), mild_tremor
  (p<0.001), tachycardic (p=0.008), agitated (p=0.003), restless
  (p=0.042), fatigued_appearance (p=0.001). Triage vitals: HR mean
  108, RR 23.8, Temp 37.73, anion_gap 12.5, glucose 115. **4-hour
  peak-lab anchors (near-exclusive to Kraken)**: peak_lactate>5.0
  (13/15 = Kraken), peak_CPK>1000 (13/14 = Kraken), peak_troponin>0.09
  (17/21 = Kraken), peak_HR>130 (14/16 = Kraken), peak_temp>38.5°C
  (4/4 = Kraken).

- **Triton Tabs** → **CNS depressant with cardiac awareness**
  (THC / benzo-like + mild auditory hallucinations). Inward, fading,
  sedation-dominant but with **subjective tachycardia awareness and
  ringing in ears**. *Not* classical bradycardia/hypotension/miosis.
  **Defining**: palpitations or cardiac-awareness chief complaint;
  ringing in ears (auditory hallmark); psychomotor slowing; slow
  response latency. PE findings: reduced_tracking (p=0.026),
  slow_responses (p=0.022), distractible. Triage vitals: HR mean 87
  (near-normal), RR 19.3, Temp 37.14, anion_gap 9.2 (often <12), pH
  often >7.35. All 4-hour peak labs near-normal (lactate 2.09, CPK
  306, troponin 0.030). Often near speaker banks / dance tents /
  DJ areas in the HPI.

- **Coral Dust** → **Hallucinogen** (LSD / psilocybin-like, likely
  insufflated — "reef-colored powder"). Immersive, perceptual,
  wave-dominant. Patient is in another world but eyes open and
  reactive. **Defining**: time-distortion sensation; perceptual
  alteration; spatial disorientation; sense of internal unease;
  waxing/waning + perceptual/somatic content. PE findings:
  unsteady_gait (29.2%), ataxia (22.9%), intermittent_disorientation
  (8.3%); **identified largely by ABSENCE of Kraken and Triton PE
  findings**. All 4-hour peak labs near-normal (lactate 2.22, CPK
  328, troponin 0.040).

- **None** → typical medical pathology unrelated to festival drugs.
  Specific alternative diagnosis named in MDM (CAD, UTI, sprain,
  appendicitis, etc.); targeted treatment given.

### v6 Discriminator Hierarchy (apply in order)

1. peak_lactate > 5.0 OR peak_CPK > 1000 → **KRAKEN** (rhabdomyolysis)
2. Diaphoretic + tachycardic on PE → **KRAKEN** (64% PPV)
3. Marked restlessness / impulsive movement / inability to remain
   still in HPI → **KRAKEN**
4. Agitation chief + rapid escalation + mild_tremor on PE → **KRAKEN**
5. Reduced_tracking + slow_responses on PE, no diaphoresis →
   **TRITON** (53% PPV)
6. Ringing in ears in HPI → **TRITON**
7. Palpitations chief + near speaker/DJ + clean peak labs → **TRITON**
8. pH > 7.35 + psychomotor slowing + clean peak labs → **TRITON**
9. Time-distortion sensation in HPI → **CORAL**
10. Perceptual alteration + normal peak labs → **CORAL**
11. Waxing/waning + perceptual/somatic + no diaphoresis + normal peak
    labs → **CORAL**
12. Unsteady gait + spatial disorientation + normal vitals → **CORAL**
13. Ambiguous with clean peak labs → **CORAL**
14. Ambiguous with elevated peak labs (lactate 2–5, CPK 400–1000) →
    **KRAKEN**

### Critical methodology warnings

1. **MDM boilerplate contamination**: the phrase *"benzodiazepine
   for agitation/sympathetic excess"* appears in **69/157 (44%) of
   MDMs across all three drugs**. It is a templated treatment line,
   not a drug-class signal. **Behavioral classification must use HPI
   text only.** If you're scoring from MDM, strip this phrase before
   reasoning.
2. **Pupils are absent**: mydriasis / miosis (the most discriminating
   classical toxicology finding) is **not present in this dataset**.
   Do not condition any reasoning on pupil size.
3. **Severity–Kraken correlation is real**: 18 of 24 high-severity
   cases are Kraken (75%). This is a pharmacological feature of
   sympathomimetics, not a labeling artifact. Do not try to
   decorrelate severity from drug class.
4. **Drug brand names DO NOT appear in the notes** — infer from
   clinical signature only.
5. **No polypharmacy** (organizer-stated): every drug-positive case
   has exactly one of {Kraken, Triton, Coral}.

### Output spec (every agent)

Path: `C:\Users\rs3te\Work\Claude-safe\SAEM-Hackathon\SAEM_2026_hackathon\derived\probs_<N>.csv`
Columns: `encounter_id, p_kraken, p_triton, p_coral, p_none`
Records: all 261, in the same order as `narratives.jsonl`. Each row
must sum to 1.0 ±0.005 after 3-decimal rounding (renormalize if drift).
A maximally uncertain encounter ≈ (0.25, 0.25, 0.25, 0.25). A clearly
non-festival case → p_none ≥ 0.7.

### Input spec (every agent)

Path: `C:\Users\rs3te\Work\Claude-safe\SAEM-Hackathon\SAEM_2026_hackathon\derived\narratives.jsonl`
Per-record fields: `encounter_id`, `chief_complaint`, `triage_brief_note`,
`brief_hpi`, `hpi`, `physical_exam_pertinent_positives` (semicolon-
delimited token list, 14 finding vocabulary), `mdm`, `clinical_course`,
`ed_meds_procedures`, `disposition`.

Venv: `C:/Users/rs3te/Work/Claude-safe/SAEM-Hackathon/.venv/Scripts/python.exe`.

Do NOT read any of: other `probs_<N>.csv` files, `labels_*.csv`,
`derived_labels.csv`, `Task1_Two_Tier_Input_Data.csv` (the ground
truth — peeking would defeat the cross-validation), or
`ground_truth*.csv`.

---

## Agent 1 — Equal weighting

```text
You are estimating probability distributions over 4 classes for 261
synthetic ED encounters from the SAEM 2026 Hackathon. Data is fully
synthetic — no PHI concerns.

# Pre-read

Read the v6 PRE-READ block at the top of PROMPTS.md — the fixed
mapping, the 14-rule discriminator hierarchy, and the methodology
warnings. They override any prior toxidrome textbook knowledge.

# Your reasoning emphasis (AGENT 1 — EQUAL WEIGHTING)

Treat all narrative fields as equally informative. Score evidence
for each toxidrome class from each field independently, then combine
into a posterior. Do not over-weight any single field. When fields
conflict, distribute probability mass between conflicting candidates
rather than collapsing to one. Use the v6 hierarchy as a tie-breaker
when multiple classes have similar evidence counts.

Apply MDM-boilerplate stripping (warning #1) before scoring MDM
evidence.

# Implementation

Use the venv. Write a Python script that reads narratives.jsonl and
emits the 4 probabilities per record. Verify all 261 rows sum to 1.0
±0.005 (renormalize after rounding). Output to probs_1.csv.

Do NOT read other probs_<N>.csv files or ground truth files.

Report back, under 200 words: (1) path, (2) record count, (3) sum-
validation count (should be 261/261), (4) marginal mean per class
(should sum to 1.0), (5) rows where p_none > 0.5.
```

---

## Agent 2 — Toxidrome-led (PE + peak labs)

```text
[Same Pre-read + Implementation + Output as Agent 1, swap to probs_2.csv.]

# Your reasoning emphasis (AGENT 2 — TOXIDROME-LED)

Weight `physical_exam_pertinent_positives` and the **4-hour peak-lab
threshold rules** heaviest — v6 shows these are the most p-value-
significant discriminators.

Operationalize from the v6 PRE-READ:

- The 14 PE tokens that matter (from v6, with class lean):
  KRAKEN: diaphoretic, mild_tremor, tachycardic, agitated, restless,
          fatigued_appearance
  TRITON: reduced_tracking, slow_responses, distractible
  CORAL:  unsteady_gait, ataxia, intermittent_disorientation
  NEUTRAL: anything else
- PE combinations (high-PPV):
  diaphoretic + tachycardic        → 64% PPV Kraken
  reduced_tracking + slow_responses → 53% PPV Triton
- Peak-lab anchors (from clinical_course numeric mentions or the
  `lab_*` columns of any 4hr summary you can parse):
  peak_lactate > 5.0  → very strong Kraken
  peak_CPK > 1000     → very strong Kraken
  peak_troponin > 0.15 → strong Kraken
  All peak labs near-normal (CPK<200, lactate<1.5, troponin<0.05) →
  supports Triton/Coral

Use HPI/MDM only as tie-breakers. **Do not use the MDM boilerplate
phrase "benzodiazepine for agitation/sympathetic excess" as
evidence** (44% prevalence across all classes).

[Same implementation + reporting requirements as Agent 1, output
to probs_2.csv.]
```

---

## Agent 3 — HPI-led (defining-token search)

```text
[Same Pre-read + Implementation + Output as Agent 1, swap to probs_3.csv.]

# Your reasoning emphasis (AGENT 3 — HPI-LED)

Weight `brief_hpi` and `hpi` heaviest. Search for the v6 *defining*
HPI tokens before falling back to soft-match phrases.

**Kraken DEFINING tokens**:
  - "agitation" / "agitated" as chief complaint
  - "marked restlessness", "inability to remain still"
  - "rapid escalation", "explosive escalation"
  - "psychomotor agitation"
  - "pressured speech", "erratic speech"
  - "police involvement", "security involvement"

**Triton DEFINING tokens**:
  - "ringing in ears", "tinnitus" — DEFINING
  - "palpitations" / "cardiac awareness" / "feel my heart"
  - "near speaker bank", "near DJ", "near dance tent"
  - "psychomotor slowing", "slow response", "withdrawn"
  - "brief staring spells", "intermittent somnolence"

**Coral DEFINING tokens**:
  - "time distortion", "time distortion sensation" — DEFINING
  - "perceptual alteration" — STRONG
  - "spatial disorientation", "blurry vision" (transient)
  - "internal unease", "somatic discomfort", "wave-like body sensation"
  - "waxing and waning" co-occurring with perceptual content

Use PE / MDM / labs only to confirm or contradict. If HPI contains a
defining token for one class AND no contradicting peak-lab anchor,
score p_<class> ≥ 0.7. Strip MDM boilerplate before any MDM glance.

[Same implementation + reporting requirements as Agent 1, output
to probs_3.csv.]
```

---

## Agent 4 — MDM minus boilerplate (differential + severity tier)

```text
[Same Pre-read + Implementation + Output as Agent 1, swap to probs_4.csv.]

# Your reasoning emphasis (AGENT 4 — MDM MINUS BOILERPLATE)

The MDM field is **contaminated** with templated text — see
methodology warning #1. The phrase "benzodiazepine for
agitation/sympathetic excess" appears in 44% of MDMs across all
three drug classes. **Strip that phrase (and obvious variants)
before doing any scoring.**

After stripping, MDM still carries two useful signals:

1. **Differential-diagnosis statements** — any explicit non-tox
   diagnosis named ("CAD ruled out", "concern for sepsis",
   "right-lower-quadrant tenderness") → strong evidence for **None**.
   If the MDM names a specific medical condition without a tox
   complement, score p_none ≥ 0.7.

2. **Severity tier language** — "low risk", "moderate", "high
   acuity", "ICU candidate", "discharged home". v6 shows Kraken is
   enriched among high-severity cases (75% of high-severity = Kraken),
   so high-severity language is a positive prior for Kraken when no
   alternative is named.

Do NOT score from:
- The benzo boilerplate phrase
- Any specific drug-class word in the MDM (the clinician's
  speculation is biased and circular)
- Treatment plan items (those are covered by Agent 6)

When stripped MDM is empty or only contains the boilerplate, defer
to a wide prior (0.25 / 0.25 / 0.25 / 0.25) and let the consensus
average dilute this agent's contribution.

[Same implementation + reporting requirements as Agent 1, output
to probs_4.csv.]
```

---

## Agent 5 — Conservative / Bayesian

```text
[Same Pre-read + Implementation + Output as Agent 1, swap to probs_5.csv.]

# Your reasoning emphasis (AGENT 5 — CONSERVATIVE / BAYESIAN)

Think like a Bayesian who is honest about uncertainty. Avoid
overconfident probabilities.

- **Reserve p > 0.85** only for cases where ≥2 independent v6
  discriminator rules fire AND don't contradict.
- **For cases that fire a v6 RULE 1 anchor** (peak_lactate>5 OR
  peak_CPK>1000): p_kraken ≥ 0.85 is appropriate — these are
  near-exclusive (13/15 and 13/14 in the ground truth).
- **For cases that fire a DEFINING HPI token** (ringing in ears for
  Triton, time-distortion for Coral, marked restlessness for Kraken):
  p_<class> ≈ 0.7.
- **For ambiguous cases** with overlapping signals, split mass
  proportionally — e.g. 0.40 / 0.40 / 0.10 / 0.10.
- **For clear non-festival presentations** (specific medical
  diagnosis named in MDM after boilerplate strip): p_none ≥ 0.85.
- **For thin clinical signal** (festival + nonspecific HPI + no
  defining PE + no peak-lab anchor): default to a wider
  (0.25, 0.25, 0.25, 0.25) or (0.33, 0.33, 0.33, 0.01).

Your output should have a higher proportion of "soft" distributions
than the other agents — that's the point.

# Reporting

[Same as Agent 1, plus: count of rows where max probability < 0.5
("soft" rows) — should be the highest of any agent.]
```

---

## Agent 6 — Treatment-response trajectory (de-boilerplated)

```text
[Same Pre-read + Implementation + Output as Agent 1, swap to probs_6.csv.]

# YOUR INDEPENDENT HYPOTHESIS: TREATMENT-RESPONSE TRAJECTORY

Reason about each case by asking: **what was tried, did it work,
did the patient escalate?**

**Important**: in v6, the phrase "benzodiazepine for agitation /
sympathetic excess" is in 44% of MDMs across all drugs. Treat the
mere mention of benzodiazepine in MDM as NO signal. Look at
`ed_meds_procedures` and `clinical_course` for what actually
happened, not what was templated.

Class-conditional patterns:

- **Cooling / antipyretics required, hyperthermia documented** →
  strong Kraken (peak_temp>38.5°C is 4/4 Kraken in v6).
- **Aggressive fluid resuscitation + CK monitoring + cardiology
  consult / troponin reassessment** → strong Kraken (rhabdomyolysis
  workup pattern).
- **Restraint orders, security calls, multiple benzo doses
  documented in `ed_meds_procedures`** (not boilerplate one-liner) →
  Kraken.
- **Airway intervention (NIPPV / intubation) and reversal agents
  (naloxone / flumazenil)** in `ed_meds_procedures` → was historically
  considered Triton, but v6 reframes Triton as cardiac-awareness
  depressant — reversal-agent administration is rare in this dataset's
  Triton cohort and is **NOT a reliable Triton signal**. Treat
  airway/reversal mentions as severity markers, not class markers.
- **Supportive care only — quiet room, observation, oral hydration,
  brief monitoring** → Coral or mild Triton.
- **Targeted medical treatment (antibiotics, NSAID for sprain, IV
  steroids for asthma, cardiac workup with definite findings, etc.)**
  → strong None.
- **Trajectory: rapid normalization within 4h** → Kraken (resolves
  on its own as drug metabolizes) or Coral (gradual self-resolution).
- **Trajectory: persistent / plateau / requires monitoring beyond
  4h** → Triton (the slowing persists).

Confidence high (≥0.7) when the trajectory matches one class
cleanly; spread mass for ambiguous response.

[Same implementation + reporting requirements as Agent 1, output
to probs_6.csv.]
```

---

## Agent 7 — Autonomic-only decision tree (no pupils)

```text
[Same Pre-read + Implementation + Output as Agent 1, swap to probs_7.csv.]

# YOUR INDEPENDENT HYPOTHESIS: AUTONOMIC-ONLY DECISION TREE

**Pupils are absent from this dataset** (methodology warning #2).
The classical toxidrome decision tree's pupil branch cannot run.
This agent replaces it with autonomic-skin-thermal-motor branches
that ARE available.

Walk the tree per record:

1. **Skin / sweat / motor (PE tokens)**:
   - diaphoretic + tachycardic → strong **KRAKEN** (64% PPV)
   - diaphoretic + mild_tremor → strong **KRAKEN**
   - none of {diaphoretic, mild_tremor, agitated} present → lean
     AGAINST Kraken

2. **Mental-status arousal (PE + HPI)**:
   - agitated + restless + rapid escalation in HPI → **KRAKEN**
   - reduced_tracking + slow_responses + no diaphoresis → strong
     **TRITON** (53% PPV)
   - intermittent_disorientation + perceptual-alteration HPI →
     **CORAL**
   - calm + oriented + no defining HPI → could be None or mild Coral

3. **Vitals direction (triage + 4h trajectory)**:
   - HR mean > 110 OR peak_HR > 130 OR peak_temp > 38.0°C →
     supports KRAKEN
   - HR mean ~85–95 AND temp normal AND peak_HR < 110 → supports
     TRITON or CORAL
   - HR mildly ↑ during waxing/waning + normal labs → CORAL

4. **Peak-lab autonomic markers (rhabdomyolysis chain)**:
   - peak_lactate > 5.0 OR peak_CPK > 1000 → near-exclusive
     **KRAKEN** (v6 rule #1)
   - all peak labs near-normal → exclude KRAKEN ⇒ TRITON or CORAL

5. **Motor / coordination (PE)**:
   - unsteady_gait + ataxia → **CORAL**
   - mild_tremor without ataxia → **KRAKEN**

Encode "fields-with-evidence" — every branch the record matches
contributes additive evidence. Distribute probability proportional
to total tree-branch evidence per class. If no branch fires, default
to the flat prior (0.25, 0.25, 0.25, 0.25).

[Same implementation + reporting requirements as Agent 1, output
to probs_7.csv.]
```

---

## Agent 8 — Counterfactual / negative-evidence

```text
[Same Pre-read + Implementation + Output as Agent 1, swap to probs_8.csv.]

# YOUR INDEPENDENT HYPOTHESIS: COUNTERFACTUAL / NEGATIVE-EVIDENCE

For each record, instead of asking "what supports class X?", ask
"**what makes this NOT class X?**" and subtract.

1. Start with a flat prior: (0.25, 0.25, 0.25, 0.25).
2. Apply v6 exclusion rules — each fires a multiplicative penalty
   on the named class:
   - PE includes diaphoretic OR mild_tremor → exclude TRITON
     and CORAL (those drugs lean against these findings; p<0.05)
   - PE includes reduced_tracking OR slow_responses, no diaphoresis →
     exclude KRAKEN (Kraken lean-against)
   - PE includes unsteady_gait or ataxia → exclude KRAKEN AND TRITON
     (Coral lean-yes)
   - All peak labs near-normal (lactate<1.5, CPK<200, troponin<0.05)
     → STRONG exclusion of KRAKEN
   - peak_lactate > 5.0 OR peak_CPK > 1000 → STRONG exclusion of
     TRITON and CORAL
   - Explicit alternative medical diagnosis named in MDM (after
     stripping boilerplate) — CAD, UTI, sprain, etc. → exclude all
     three drug classes ⇒ p_none surges
3. Multiplicative penalty per exclusion firing:
   `p_class *= max(0.1, 1 - 0.25 * exclusion_strength)`
   where exclusion_strength is 1.0 for STRONG, 0.5 for soft.
4. Renormalize.
5. Confidence ceiling: max(p_class) ≤ 0.80. If your distribution
   would otherwise exceed it, smooth toward the flat prior.

This produces wider, less-confident distributions than the
evidence-additive agents — that's the point. You're modeling
"evidence absent" rather than "evidence present", and the v6
discriminator table gives explicit lean-against directions that
this agent operationalizes.

[Same implementation + reporting requirements as Agent 1, output
to probs_8.csv. Plus: count of rows where max_prob > 0.5
(low-uncertainty rows).]
```

---

## Agent 9 — Recovery-pattern pharmacokinetics

```text
[Same Pre-read + Implementation + Output as Agent 1, swap to probs_9.csv.]

# YOUR INDEPENDENT HYPOTHESIS: RECOVERY-PATTERN PHARMACOKINETICS

Classify based on the **temporal arc of the encounter** derived from
`clinical_course`, `hpi`, and (after boilerplate strip) `mdm`.

v6-recalibrated PK profiles:

- **Sympathomimetic (Kraken)** PK: rapid onset (minutes after
  ingestion), peak symptoms 1–2 h pre-hospital, **resolves within
  4–8 h** as drug metabolizes — BUT may leave rhabdomyolysis labs
  (CPK/lactate/troponin) elevated at 4h. Watch for: vitals
  normalized by hour 2, cooling effective, agitation subsides
  spontaneously, but peak_CPK > 1000 persists.

- **CNS depressant with cardiac awareness (Triton)** PK: gradual
  onset over 30–90 min, **persistent psychomotor slowing through 4h**,
  patient remains aware of own racing heartbeat and ringing
  throughout. Vitals largely normal, labs clean. Watch for: "remained
  withdrawn", "tracking deficits persisted at hour 3", "patient
  reported continued tinnitus", "palpitations persisted but vitals
  reassuring".

- **Hallucinogen (Coral)** PK: variable onset (38–51 min for
  insufflated route), **gradual self-resolution** over 2–6 h with
  supportive care, perceptual symptoms wane. Watch for: "perceptual
  symptoms improving", "tolerated PO at hour 3", "remained calm with
  quiet room", waxing/waning that DAMPENS over time.

- **None**: trajectory describes targeted workup/treatment for a
  specific non-toxidrome diagnosis. Improvement is
  condition-specific (antibiotics, sprain RICE, cardiac
  intervention, etc.).

Probability proportional to how well each PK profile matches the
recorded trajectory. Boost Kraken on "rapid behavioral resolution
+ elevated peak labs at 4h". Boost Triton on "persistent slowing,
clean labs". Boost Coral on "perceptual symptoms wax/wane and
dampen, clean labs". Boost None on "specific medical
treatment given and effective".

Do NOT use the benzo-boilerplate MDM phrase as a treatment-response
signal.

[Same implementation + reporting requirements as Agent 1, output
to probs_9.csv.]
```

---

## Agent 10 — v6 PE-token + peak-lab cluster matching

```text
[Same Pre-read + Implementation + Output as Agent 1, swap to probs_10.csv.]

# YOUR INDEPENDENT HYPOTHESIS: v6 CLUSTER MATCHING (DATA-DRIVEN)

Treat `physical_exam_pertinent_positives` as a bag-of-tokens
(semicolon-delimited) and add a peak-lab cluster derived from
`clinical_course` (or any numeric peak-lab field if present).

1. **Reference clusters (v6, 14-finding vocabulary)**:
   - **Cluster K (Kraken / sympathomimetic)**: {diaphoretic,
     mild_tremor, tachycardic, agitated, restless,
     fatigued_appearance}
   - **Cluster T (Triton / depressant + cardiac awareness)**:
     {reduced_tracking, slow_responses, distractible}
   - **Cluster C (Coral / hallucinogen)**: {unsteady_gait, ataxia,
     intermittent_disorientation}
   - (Tokens not in the 14-finding vocabulary are ignored — do not
     fabricate tokens like `mydriasis`, `miosis`, `hyperreflexic`,
     `hypertensive`, `flushed`. They are not in this dataset.)

2. **Peak-lab cluster** (parse numbers from clinical_course or any
   `peak_*` / `lts_*_max` field in the record):
   - peak_lactate > 5.0 → +1.0 Kraken
   - peak_CPK > 1000   → +1.0 Kraken
   - peak_troponin > 0.15 → +0.5 Kraken
   - peak_HR > 150 OR peak_temp > 38.5°C → +0.5 Kraken
   - all of {peak_CPK < 200, peak_lactate < 1.5, peak_troponin < 0.05}
     → +0.4 Triton AND +0.4 Coral
   - (no numeric peak labs parseable: skip this section)

3. **Token overlap score** per PE-cluster: count of tokens from the
   record present in cluster ÷ cluster size. Combine with peak-lab
   bonuses additively.

4. **None signal**: any pertinent-positive token NOT in K/T/C plus an
   explicit medical-diagnosis keyword in MDM (after boilerplate
   strip) — chest_pain_cardiac, focal_neuro, abd_tenderness,
   dyspnea_pulmonary, fever_localized_source, etc. → boost p_none.

5. **Softmax** with a soft uniform prior of 0.1 per drug class, 0.15
   for None.

6. **Fallback**: when `physical_exam_pertinent_positives` is empty
   AND no peak-lab numerics parseable, fall back to
   (0.20, 0.20, 0.20, 0.40) — defer to None when symptom evidence
   is absent.

This is a **bottom-up token-statistics approach** rather than
top-down toxidrome reasoning — different inductive bias from
Agents 1–9.

[Same implementation + reporting requirements as Agent 1, output
to probs_10.csv.]
```

---

## Why this design

**v6-anchored fixed mapping across all ten agents.** The v6 toxidrome
report (n=157 manually annotated cases) supersedes the pre-v6
textbook mapping. Triton in this dataset is *not* the classical
bradycardic/miotic sedative — it's a CNS depressant with subjective
cardiac awareness and tinnitus. Encoding the v6 mapping up front
prevents agents from re-deriving the wrong textbook signature.

**Two-axis diversity preserved**:

- Agents 1–5 vary **which fields the agent looks at** (equal /
  toxidrome-led / HPI-led / MDM-minus-boilerplate /
  Bayesian-conservative). They share the substrate (the v6 PRE-READ)
  but differ in evidence source.
- Agents 6–10 vary **what reasoning paradigm the agent applies**
  (treatment-response trajectory, autonomic-only decision tree,
  counterfactual / negative-evidence, recovery-pattern PK, v6
  cluster matching). They all see the same evidence but reason
  differently about it.

This two-axis diversity guards against systematic blind spots in any
single hypothesis. **Agents 4 and 7 are repurposed** from their
pre-v6 forms: Agent 4 strips the MDM boilerplate (which contaminates
44% of MDMs); Agent 7 removes the pupil branch (absent from the
dataset) in favor of autonomic-skin-motor branches.

**Aggregation**: simple per-class mean across all ten agents, then
renormalize. Per-class cross-agent standard deviation retained in
`probs_avg.csv` so high-disagreement rows can be surfaced.

**Calibration target**: the manual annotations in
`data/Task1_Two_Tier_Input_Data.csv` (n=157 drug-positive cases). The
v6 hierarchy was derived from those annotations, so agent consensus
that fires the v6 rules cleanly should align with the manual
`ground_truth_drug_name`.
