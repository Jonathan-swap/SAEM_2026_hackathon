# Subagent Prompts — Probability Extraction (10 Agents)

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
| 2 | Toxidrome-led — PE + meds + course | `probs_2.csv` |
| 3 | HPI-led — brief_hpi + hpi + brief note | `probs_3.csv` |
| 4 | MDM-led — clinician's reasoning | `probs_4.csv` |
| 5 | Conservative / Bayesian — quantified uncertainty | `probs_5.csv` |
| 6 | Treatment-response trajectory | `probs_6.csv` |
| 7 | Classical pupil + autonomic decision tree | `probs_7.csv` |
| 8 | Counterfactual / negative-evidence | `probs_8.csv` |
| 9 | Recovery-pattern pharmacokinetics | `probs_9.csv` |
| 10 | Symptom-cluster prevalence (token bag) | `probs_10.csv` |

Each agent emits a CSV with columns `encounter_id, p_kraken,
p_triton, p_coral, p_none` for all 261 records; every row sums to
1.0 (±0.005).

`src/labels/merge_probabilities.py` auto-discovers all
`probs_<N>.csv` files, averages them, and writes the consensus to
`derived/probs_avg.csv`.

---

## Agent 1 — Equal weighting

```text
You are estimating probability distributions over 4 classes for 261
synthetic ED encounters from the SAEM 2026 Hackathon. The data is
fully synthetic — no PHI concerns.

# Classes (THIS MAPPING IS FIXED across all 5 agents in this batch)

Based on the hackathon brief's drug descriptions, the inferred
toxidrome mappings are:

- **Kraken Candy** ("strong and chaotic, unpredictable intensity")
  → **SYMPATHOMIMETIC / stimulant**. Signs: tachycardia,
  hypertension, hyperthermia, diaphoresis, agitation, tremor,
  mydriasis. Treatments: benzodiazepines for agitation, IV fluids,
  antipyretics, cooling.
- **Triton Tabs** ("small tabs, ocean-deep") → **SEDATIVE-HYPNOTIC
  / depressant**. Signs: bradycardia, hypotension, hypoventilation,
  low GCS, miosis, slow responses. Treatments: airway support,
  NIPPV, intubation, reversal agents (flumazenil, naloxone).
- **Coral Dust** ("vivid wave-like") → **HALLUCINOGENIC /
  serotonergic**. Signs: perceptual changes, mild tachycardia,
  ataxia, unsteady gait, hyperreflexia. Treatments: supportive
  care, benzodiazepines if agitated, IV fluids, dim/quiet room.
- **None** → typical medical pathology unrelated to festival drugs.

# Ground rules

- No polypharmacy (organizer-stated).
- Drug brand names DO NOT appear in the notes — infer from clinical
  signature.
- Output four probabilities per encounter that MUST sum to 1.0
  (±0.005 tolerance after rounding to 3 decimals).
- A maximally uncertain encounter ≈ (0.25, 0.25, 0.25, 0.25).
- A clearly non-festival case → high p_none (≥0.7).
- A clearly stimulant-toxidrome case → high p_kraken.

# Input

`C:\Users\rs3te\Work\Claude-safe\SAEM-Hackathon\derived\narratives.jsonl`
(261 records, one JSON per line). Use ONLY note fields: brief_hpi,
hpi, physical_exam_pertinent_positives, mdm, clinical_course,
ed_meds_procedures, triage_brief_note, chief_complaint.

# Output

`C:\Users\rs3te\Work\Claude-safe\SAEM-Hackathon\derived\probs_1.csv`

Columns: `encounter_id,p_kraken,p_triton,p_coral,p_none`

All 261 records in the same order as the JSONL. Each row: floats
rounded to 3 decimals; sum must equal 1.0 (±0.005). If your rounding
pushes the sum off, renormalize so the sum is exactly 1.0.

# Your reasoning emphasis (AGENT 1 — EQUAL WEIGHTING)

Treat all narrative fields as equally informative. Score evidence
for each toxidrome class from each field, then combine into a
posterior. Do not over-weight any single field. When fields
conflict, distribute probability mass between the conflicting
candidates rather than picking one.

# Implementation

Use the venv:
`C:/Users/rs3te/Work/Claude-safe/SAEM-Hackathon/.venv/Scripts/python.exe`.
Write a Python script that reads the JSONL and, for each record,
applies your reasoning and emits the 4 probabilities. After writing
the CSV, verify that every row sums to 1.0 ±0.005; print the count
of rows that fail validation (should be 0).

Do NOT read any of: `probs_2.csv`, `probs_3.csv`, `probs_4.csv`,
`probs_5.csv`, `labels_*.csv`, `derived_labels.csv`,
`ground_truth_labels*.csv`, files in `.claude/plans/`.

Report back: (1) path, (2) record count, (3) sum-validation count
(should be 261/261), (4) marginal mean prob per class (should sum
to 1.0), (5) any rows where p_none > 0.5. Keep final message under
200 words.
```

---

## Agent 2 — Toxidrome-led

```text
You are estimating probability distributions over 4 classes for 261
synthetic ED encounters from the SAEM 2026 Hackathon. The data is
fully synthetic — no PHI concerns.

# Classes (THIS MAPPING IS FIXED across all 5 agents in this batch)

- **Kraken Candy** → **SYMPATHOMIMETIC**: tachycardia, hypertension,
  hyperthermia, diaphoresis, agitation, tremor, mydriasis.
  Treatments: benzodiazepines for agitation, IV fluids,
  antipyretics, cooling.
- **Triton Tabs** → **SEDATIVE-HYPNOTIC**: bradycardia, hypotension,
  hypoventilation, low GCS, miosis, slow responses. Treatments:
  airway support, NIPPV, intubation, reversal agents.
- **Coral Dust** → **HALLUCINOGENIC / serotonergic**: perceptual
  changes, mild tachycardia, ataxia, unsteady gait, hyperreflexia.
  Treatments: supportive care, benzodiazepines if agitated, IV
  fluids.
- **None** → typical medical pathology unrelated to festival drugs.

# Ground rules

- No polypharmacy.
- Drug names absent from notes — infer from clinical signature.
- 4 probabilities sum to 1.0 (±0.005 after 3-decimal rounding).
- Uncertain → (0.25, 0.25, 0.25, 0.25); non-festival → p_none ≥0.7.

# Input

`...\derived\narratives.jsonl` (261 records). Use ONLY note fields.

# Output

`...\derived\probs_2.csv`. Columns:
`encounter_id,p_kraken,p_triton,p_coral,p_none`. All 261 records,
same order as JSONL. Renormalize if rounding throws off the sum.

# Your reasoning emphasis (AGENT 2 — TOXIDROME-LED)

Weight `physical_exam_pertinent_positives` and `ed_meds_procedures`
heaviest. The physical exam tokens (e.g. `diaphoretic`,
`tachycardic`, `ataxia`, `mild_tremor`, `dry_mucosa`,
`slow_responses`, `reduced_tracking`) and treatment patterns
(intubation/NIPPV/reversal → sedative class; benzo + antipyretic →
stimulant class; supportive only → hallucinogen) are your strongest
signal. Use HPI/MDM/clinical_course only as tie-breakers.

# Implementation

Use the venv. Write a Python script. Verify all rows sum to 1.0 ±0.005.

Do NOT read other agents' outputs.

Report back: (1) path, (2) record count, (3) sum-validation count,
(4) marginal mean prob per class, (5) p_none > 0.5 count.
<200 words.
```

---

## Agent 3 — HPI-led

```text
[Same Classes block + Ground rules + Input as Agent 2.]

# Output

`...\derived\probs_3.csv`. Columns:
`encounter_id,p_kraken,p_triton,p_coral,p_none`. All 261 records,
same order as JSONL.

# Your reasoning emphasis (AGENT 3 — HPI-LED)

Weight `brief_hpi` and `hpi` heaviest — these capture symptom onset,
festival exposure, and the patient's presenting story. Look for:
festival/main stage/campground/medical tent mentions; symptom onset
window; nature of perceptual or autonomic symptoms; collateral
history. Use physical findings and meds as confirmation/contradiction,
not as primary signal.

# Implementation

[Same as Agent 2.]
```

---

## Agent 4 — MDM-led

```text
[Same Classes block + Ground rules + Input as Agents 2-3.]

# Output

`...\derived\probs_4.csv`. Same columns and order.

# Your reasoning emphasis (AGENT 4 — MDM-LED)

Weight `mdm` and `clinical_course` heaviest — these capture the
clinician's working impression, severity tier, and what actually
happened. Look for: differential diagnosis statements;
severity/vulnerability language; treatment-rationale clauses;
trajectory descriptions. Use HPI/exam as confirmation only. The
clinician's reasoning is the closest thing you have to expert
ground truth.

# Implementation

[Same as Agent 2.]
```

---

## Agent 5 — Conservative / Bayesian

```text
[Same Classes block + Ground rules + Input as Agents 2-4.]

# Output

`...\derived\probs_5.csv`. Same columns and order.

# Your reasoning emphasis (AGENT 5 — CONSERVATIVE / BAYESIAN)

Think like a Bayesian who is honest about uncertainty. Avoid
overconfident probabilities. Specifically:

- **Reserve p > 0.85** only for cases where ≥2 narrative fields
  independently and strongly point to the same class.
- **For ambiguous cases** (signature could plausibly match two
  toxidromes), split probability between those classes (e.g.,
  0.45/0.45/0.05/0.05). Do not collapse to a single class.
- **For clear non-festival presentations**, p_none ≥ 0.85.
- **For festival cases with thin clinical signal**, default to a
  wider distribution like (0.25, 0.25, 0.25, 0.25) or
  (0.33, 0.33, 0.33, 0.01) rather than guessing.

Your output should have a higher proportion of "soft" distributions
than the other agents — that's the point.

# Implementation

Use the venv. Write a Python script. Verify all rows sum to 1.0
±0.005.

Do NOT read other agents' outputs.

Report back: (1) path, (2) record count, (3) sum-validation count,
(4) marginal mean prob per class, (5) p_none > 0.5 count, (6) count
of rows where max probability < 0.5 (indicates "soft" rows).
<200 words.
```

---

## Agent 6 — Treatment-response trajectory

```text
Estimate probability distributions over 4 classes for 261 synthetic
ED encounters (SAEM 2026 Hackathon). Data is fully synthetic.

# Fixed mapping (same as prior batches — consistent name binding)

- **Kraken Candy** → SYMPATHOMIMETIC (tachy, HTN, hyperthermia,
  diaphoresis, agitation, tremor, mydriasis). Treatment: benzos,
  IVF, antipyretics, cooling.
- **Triton Tabs** → SEDATIVE-HYPNOTIC (bradycardia, hypotension,
  hypoventilation, low GCS, miosis, slow responses). Treatment:
  airway support, NIPPV, intubation, reversal.
- **Coral Dust** → HALLUCINOGENIC / serotonergic (perceptual
  changes, mild tachy, ataxia, hyperreflexia). Treatment:
  supportive care, mild benzo, IVF.
- **None** → typical medical pathology, no festival drug.

# Output

`...\derived\probs_6.csv`. Columns:
`encounter_id,p_kraken,p_triton,p_coral,p_none`. 261 rows, each
summing to 1.0 ±0.005 (renormalize if rounding drifts).

# Input

`...\derived\narratives.jsonl`. Notes only. Use the venv
`.venv/Scripts/python.exe`.

# YOUR INDEPENDENT HYPOTHESIS: TREATMENT-RESPONSE TRAJECTORY

Reason about each case by asking: **what was tried, did it work,
did the patient escalate?** Treatment-response trajectory is more
diagnostic of drug class than initial presentation:

- **Benzodiazepine GIVEN and worked** (HR/agitation normalized, no
  escalation) → strongly Kraken (sympathomimetic responds to GABA-A
  agonism)
- **Benzodiazepine FAILED, escalated to airway support
  (NIPPV/intubation)** → strongly Triton (sedative caused
  respiratory depression that benzo worsens; required
  reversal/airway)
- **Reversal agent (naloxone/flumazenil) given** → strongly Triton
  (only sedatives have reversals)
- **Cooling/antipyretics required** → strongly Kraken (hyperthermia)
- **Supportive care only, no pharmacologic intervention** →
  strongly Coral (mild hallucinogen self-resolves)
- **Targeted medical treatment (antibiotics, NSAIDs for sprain, IV
  steroids for asthma, etc.)** → strongly None
- **Patient deteriorated despite initial supportive care** →
  escalation to ICU pattern, lean Triton or severe Kraken

Use `ed_meds_procedures`, `clinical_course`, and `mdm` for
treatment-response. Confidence high (≥0.7) when response pattern
is clear; spread mass for ambiguous response.

Do NOT read other probs_*.csv files. Report under 200 words: path,
record count, sum-validation, marginal means, p_none>0.5 count.
```

---

## Agent 7 — Classical toxidrome decision tree

```text
[Same Fixed Mapping + Output spec as Agent 6, paths swap to
probs_7.csv.]

# YOUR INDEPENDENT HYPOTHESIS: CLASSICAL TOXIDROME DECISION TREE

Apply the textbook toxicology decision tree — pupils + autonomic +
AMS pattern:

    1. Mental status:
       - Agitated/restless/anxious → likely sympathomimetic OR
         hallucinogen
       - Sedated/lethargic/somnolent/low GCS → likely sedative
       - Hallucinating/perceptual distortion → likely hallucinogen
       - Confused-but-no-toxidrome → consider None
    2. Autonomic / vitals direction:
       - HR ↑ + BP ↑ + Temp ↑ + diaphoretic → SYMPATHOMIMETIC (Kraken)
       - HR ↓ + BP ↓ + RR ↓ + cool/dry → SEDATIVE (Triton)
       - HR mildly ↑ + BP normal + temp normal + dry mucosa →
         HALLUCINOGEN (Coral)
       - Vitals stable + focal complaint → None
    3. Pupil pattern (when mentioned):
       - Mydriasis (large) → stimulant or hallucinogen
       - Miosis (small) → sedative/opioid-like
    4. Motor / coordination:
       - Tremor + clonus → sympathomimetic / serotonergic
       - Ataxia + unsteady_gait → hallucinogen
       - Flaccid / hyporeflexic → sedative

Walk the tree explicitly per record. Encode "fields-with-evidence"
— every branch that the record matches contributes additive
evidence. Distribute probability proportional to total tree-branch
evidence per class.

Do NOT read other probs_*.csv files. Report under 200 words: path,
record count, sum-validation, marginal means, p_none>0.5 count.
```

---

## Agent 8 — Counterfactual / negative-evidence

```text
[Same Fixed Mapping + Output spec as Agent 6, paths swap to
probs_8.csv.]

# YOUR INDEPENDENT HYPOTHESIS: COUNTERFACTUAL / NEGATIVE-EVIDENCE
# REASONING

For each record, instead of asking "what supports class X?", ask
"**what makes this NOT class X?**" and subtract:

1. Start with a flat prior: (0.25, 0.25, 0.25, 0.25).
2. For each class, scan the notes for **exclusionary evidence** —
   findings that *argue against* that class.
   - Bradycardic with normal/high BP → exclude Triton (sedatives
     don't preserve perfusion this way)
   - Calm/awake with stable vitals → exclude Kraken (no
     sympathomimetic without autonomic findings)
   - No perceptual symptoms mentioned AND no ataxia → exclude
     Coral (hallucinogen requires perceptual signal)
   - Explicit alternative medical diagnosis named in MDM (CAD, UTI,
     sprain, etc.) → exclude all three drug classes
3. Apply a multiplicative penalty proportional to exclusionary
   evidence count: `p_class *= max(0.1, 1 - 0.2 * exclusion_count)`.
4. Renormalize.
5. If no class has exclusionary evidence, output the flat prior.

This produces wider, less-confident distributions than
evidence-additive methods — that's the point. You're explicitly
modeling "evidence absent" rather than "evidence present".

Confidence ceiling: max(p_class) ≤ 0.75. If your distribution would
otherwise exceed it, smooth with the prior.

Do NOT read other probs_*.csv files. Report under 200 words: path,
record count, sum-validation, marginal means, count where max_prob
> 0.5 (low-uncertainty rows).
```

---

## Agent 9 — Recovery-pattern pharmacokinetics

```text
[Same Fixed Mapping + Output spec as Agent 6, paths swap to
probs_9.csv.]

# YOUR INDEPENDENT HYPOTHESIS: RECOVERY-PATTERN PHARMACOKINETICS

Classify based on the **temporal arc of the encounter** — onset,
peak, and recovery dynamics — derived from `clinical_course`,
`mdm`, and `hpi`:

- **Sympathomimetic (Kraken)** pharmacokinetics: rapid onset
  (~minutes after ingestion), peak symptoms within 1–2 hours, often
  pre-hospital, **resolves within 4–8 hours** as drug metabolizes.
  Watch for "settled with benzo", "vitals normalized by hour 2",
  "rapidly responsive to cooling".
- **Sedative (Triton)** pharmacokinetics: delayed peak (drug
  redistribution, "re-sedation"), **prolonged recovery** (need
  monitoring beyond 4h), often plateau or worsening trajectory at
  4h. Watch for "remained somnolent", "required prolonged airway
  support", "minimal improvement despite reversal".
- **Hallucinogen (Coral)** pharmacokinetics: variable onset
  depending on dose, **gradual self-resolution** with supportive
  care, perceptual symptoms wane over 2–6 hours. Watch for
  "perceptual symptoms improving", "remained calm with quiet room",
  "tolerated PO at hour 3".
- **None**: clinical course describes targeted workup/treatment for
  a specific non-toxidrome diagnosis; trajectory is
  condition-specific (improved with antibiotics, stable post-CT,
  etc.).

Probability proportional to how well each PK pattern matches the
recorded trajectory. Mention of "rapid improvement" or "fully
resolved within 4h" → boost Kraken/Coral. "Prolonged" or
"persistent" → boost Triton or severe None. "Specific medical
treatment given" → boost None.

Do NOT read other probs_*.csv files. Report under 200 words: path,
record count, sum-validation, marginal means, p_none>0.5 count.
```

---

## Agent 10 — Symptom-cluster prevalence (token bag)

```text
[Same Fixed Mapping + Output spec as Agent 6, paths swap to
probs_10.csv.]

# YOUR INDEPENDENT HYPOTHESIS: SYMPTOM-CLUSTER PREVALENCE
# (DATA-DRIVEN)

Treat `physical_exam_pertinent_positives` as a bag-of-tokens
(semicolon-delimited finding flags) and classify by **cluster
matching** rather than evidence-counting:

1. **Pre-define three reference clusters**:
   - **Cluster K (Kraken / stimulant)**: {tachycardic, diaphoretic,
     mild_tremor, restless, agitated, hyperthermic, mydriasis,
     anxious, hypertensive, flushed}
   - **Cluster T (Triton / sedative)**: {slow_responses,
     reduced_tracking, intermittent_disorientation, distractible,
     somnolent, bradypneic, hypotonic, miosis, low_gcs, lethargic,
     fatigued_appearance}
   - **Cluster C (Coral / hallucinogen)**: {ataxia, unsteady_gait,
     dry_mucosa, hyperreflexic, perceptual_distortion, tremor_fine,
     intermittent_hallucinations}
2. For each record, **count token overlap** with each cluster.
   Compute Jaccard-like similarity per class.
3. **None token cluster**: any pertinent-positive token NOT in
   K/T/C plus explicit medical-diagnosis keywords in MDM/HPI
   (chest_pain_cardiac, focal_neuro, abd_tenderness,
   fever_localized_source, dyspnea_pulmonary, etc.) → boost p_none.
4. Probability = softmax(similarity scores) with a uniform prior of
   0.1 per drug class, 0.15 for None.
5. When `physical_exam_pertinent_positives` is empty or vague, fall
   back to (0.20, 0.20, 0.20, 0.40) — defer to None when symptom
   evidence is absent.

This is a **bottom-up token-statistics approach** rather than
top-down toxidrome reasoning — different inductive bias.

Do NOT read other probs_*.csv files. Report under 200 words: path,
record count, sum-validation, marginal means, p_none>0.5 count.
```

---

## Why this design

**Fixed mapping across all ten agents.** The hackathon brief's
drug descriptions ("strong and chaotic", "ocean-deep",
"vivid wave-like") map cleanly to canonical toxidrome classes
(sympathomimetic / sedative / hallucinogenic). Locking the
Kraken=sympathomimetic / Triton=sedative / Coral=hallucinogen
assignment up front makes the averaged consensus meaningful —
otherwise agents would swap name bindings arbitrarily (which two
earlier agents did, exposing the data's inability to anchor the
names on its own).

**Diversity along two axes**:

- Agents 1–5 vary **which fields the agent looks at** (equal /
  toxidrome-led / HPI-led / MDM-led / Bayesian-conservative). They
  share the reasoning style but differ in evidence source.
- Agents 6–10 vary **what reasoning paradigm the agent applies**
  (treatment-response trajectory, classical decision tree,
  counterfactual / negative-evidence, recovery-pattern PK,
  symptom-cluster Jaccard). They all see the same evidence but
  reason differently about it.

This two-axis diversity guards against systematic blind spots in
any single hypothesis.

**Aggregation**: simple per-class mean across all ten agents, then
renormalize so the row sums to exactly 1.0. Per-class cross-agent
standard deviation is also retained in `probs_avg.csv` so high-
disagreement rows can be surfaced for review.

**Honest caveat**: the drug names (Kraken / Triton / Coral) never
appear in any of the released notes. Two independent agent batches
swapped Triton ↔ Coral assignments, confirming that the names are
unrooted in the data — the toxidrome classes are recoverable, but
the name binding requires external ground truth (organizer's
`ground_truth_labels_v4.csv`).
