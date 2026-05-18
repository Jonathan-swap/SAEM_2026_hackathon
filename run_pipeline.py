"""Phase-2 retrain orchestrator.

Runs the full feature + training pipeline end-to-end with wall-clock
timing per step. Designed to fit inside the 2.5h hackathon day-of
adaptation window with plenty of margin.

Usage:
    python run_pipeline.py                # full pipeline
    python run_pipeline.py --skip-agents  # skip LLM-agent steps (use
                                          # existing probs_*.csv files)

The 10 LLM-agent step requires the Claude Code harness to spawn
subagents — it is NOT runnable from this script. Either:
  (a) run the agents manually before invoking this script and rely
      on the existing probs_<N>.csv files in derived/, OR
  (b) regenerate probs only when the day-of dataset arrives.

Order:
  1. extract_narratives        (xlsx -> narratives.jsonl)
  2. extract_structured        (xlsx -> features_*.csv, outcomes left out)
  3. extract_time_features     (Groups A-G time features)
  4. extract_differentials     (triage<->4h paired deltas)
  5. extract_note_features     (onset minutes + location, triage notes)
  6. extract_note_4h_features  (HPI/MDM word counts + severity tier,
                                4h-only — never lands in features_triage)
  7. extract_v6_features       (PE binaries + peak-lab thresholds +
                                triage keywords from toxidrome v6)
  8. load_ground_truth         (drug labels -> derived/ground_truth.csv)
  9. build_outcomes            (drug + disposition merged -> outcomes.csv)
 10. [optional] 10 LLM agents  (spawn via harness — manual step)
 11. merge_probabilities       (consensus across agents)
 12. cleanup_features          (drop constants, merge candidates)
 13. train Task 1 baseline     (direct 4-class — §7a comparison)
 14. task1_cascade_b           (DEFAULT — Cascade-B with rforest +
                                 macro-F1-picked thresholds + prevalence
                                 Bernoulli stage-3; writes the canonical
                                 derived/task1_drug_predictions.csv)
 15. train Task 2              (deterioration at 4h)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # use whatever venv invoked us

STEPS = [
    ("extract_narratives",   "src/features/extract_narratives.py"),
    ("extract_structured",   "src/features/extract_structured.py"),
    ("extract_time_features","src/features/extract_time_features.py"),
    ("extract_differentials","src/features/extract_differentials.py"),
    ("extract_note_features","src/features/extract_note_features.py"),
    ("extract_note_4h_features","src/features/extract_note_4h_features.py"),
    ("extract_v6_features",  "src/features/extract_v6_features.py"),
    ("load_ground_truth",    "src/labels/load_ground_truth.py"),
    ("build_outcomes",       "src/labels/build_outcomes.py"),
]

POST_AGENT_STEPS = [
    ("merge_probabilities",  "src/labels/merge_probabilities.py"),
    ("eda_descriptive",      "src/eda/eda_descriptive.py"),
    ("cleanup_features",     "src/features/cleanup_features.py"),
    ("train_task1_baseline", "src/task1_drug_id/train_baseline.py"),
    # DEFAULT Task-1 model: Cascade-B (rforest, threshold-tuned, prevalence-
    # Bernoulli stage-3). Produces the canonical per-encounter predictions
    # at derived/task1_drug_predictions.csv (encounter_id, drug_class 0-3).
    ("task1_cascade_b",      "src/task1_drug_id/threshold_cascade_b.py"),
    ("train_task2",          "src/task2_deterioration/train_baseline.py"),
]


def run_step(name: str, script: str) -> float:
    """Run a script via the current Python; return elapsed seconds."""
    print(f"\n[{name}] -> {script}")
    print("-" * 72)
    start = time.perf_counter()
    rc = subprocess.run([PY, str(ROOT / script)], cwd=str(ROOT)).returncode
    elapsed = time.perf_counter() - start
    status = "OK" if rc == 0 else f"FAIL (rc={rc})"
    print(f"  -> {status}  ({elapsed:.1f}s)")
    if rc != 0:
        raise SystemExit(f"Pipeline halted at step '{name}' (rc={rc})")
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-agents", action="store_true",
                    help="Skip the 10-agent LLM step; reuse existing "
                         "probs_<N>.csv files in derived/")
    ap.add_argument("--features-only", action="store_true",
                    help="Stop after the feature pipeline; do not train")
    args = ap.parse_args()

    total_start = time.perf_counter()
    timings: list[tuple[str, float]] = []

    # Phase A: feature extraction + ground-truth load
    for name, script in STEPS:
        timings.append((name, run_step(name, script)))

    # Phase B: LLM agents (manual step in the harness)
    if not args.skip_agents:
        print("\n" + "=" * 72)
        print("STEP: 10 LLM agents (MANUAL — spawn via Claude Code harness)")
        print("=" * 72)
        print("This step is not automated from a plain Python script.")
        print("Spawn 10 subagents in parallel with the prompts in")
        print("`src/labels/agents/PROMPTS.md`. Each writes a probs_<N>.csv.")
        print("If you've already done this, re-run with --skip-agents.")
        print("Continuing assuming probs_1..10.csv already exist...")

    if args.features_only:
        print(f"\n--features-only: stopping before training.")
    else:
        for name, script in POST_AGENT_STEPS:
            timings.append((name, run_step(name, script)))

    total = time.perf_counter() - total_start
    print("\n" + "=" * 72)
    print(f"PIPELINE COMPLETE — total wall clock: {total:.1f}s ({total/60:.1f} min)")
    print("=" * 72)
    print(f"{'Step':<28s} {'Time (s)':>10s}")
    for name, t in timings:
        print(f"  {name:<26s} {t:>10.1f}")
    print(f"  {'TOTAL':<26s} {total:>10.1f}")

    print("\nArtifacts (derived/):")
    print(f"  features_triage.csv / features_fourh.csv")
    print(f"  ground_truth.csv / probs_avg.csv / derived_labels.csv")
    print(f"  task1_baseline_summary.csv / task1_oof_predictions.csv "
          f"(direct 4-class baseline, §7a)")
    print(f"  task1_drug_predictions.csv "
          f"(DEFAULT — Cascade-B canonical predictions)")
    print(f"  task1_cascade_b_threshold_grid.csv / _picked.csv / "
          f"_labels.csv / _report.md")
    print(f"  task2_baseline_summary.csv / task2_oof_predictions.csv")
    print(f"  eda_descriptive_report.md + eda_plots/")


if __name__ == "__main__":
    main()
