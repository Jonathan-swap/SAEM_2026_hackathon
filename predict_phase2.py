"""End-to-end Phase-2 inference using the production models.

Generates Task-2 disposition predictions for the Phase-2 dataset
(`data2/Hackathon_Data_Release_2_SHARE.xlsx`). The Phase-2 file is
missing the `Disposition` sheet — that IS the prediction target.

Workflow:
  1. Snapshot derived/ (full copy) and back up the Phase-1 xlsx so we
     can restore both at the end.
  2. Stage the Phase-2 xlsx at the path the feature extractors expect
     (`data/Hackathon_Data_Release_1_SHARE.xlsx`).
  3. Run only the feature extracts that don't need the Disposition
     sheet:
       extract_structured -> features_triage.csv, features_fourh.csv
       extract_time_features
       extract_differentials
       extract_note_features
       extract_note_4h_features
       extract_v6_features
  4. Run production/task1/predict.py against features_triage.csv to
     get per-encounter drug-class probabilities.
  5. Reshape those into probs_avg.csv (encounter_id + p_kraken/p_triton/
     p_coral/p_none) — the exact format the task2 production model
     expects in lieu of the 10-agent LLM consensus.
  6. Run production/task2/predict.py against features_fourh.csv +
     probs_avg.csv to get disposition predictions.
  7. Move Phase-2 outputs to derived/phase2/.
  8. Restore derived/ snapshot + xlsx, regardless of success/failure.

Run:
    .venv/Scripts/python.exe predict_phase2.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA2 = ROOT / "data2"
DERIVED = ROOT / "derived"

PHASE1_XLSX_NAME = "Hackathon_Data_Release_1_SHARE.xlsx"
PHASE2_XLSX_NAME = "Hackathon_Data_Release_2_SHARE.xlsx"

# A scratch path outside DERIVED — sibling so the snapshot survives
# even if cleanup_features etc. blow away derived/.
SNAPSHOT = ROOT / "_phase1_derived_snapshot"
XLSX_BAK = DATA / f"{PHASE1_XLSX_NAME}.phase1_bak"

# Feature extraction order. extract_narratives and build_outcomes need
# the Disposition sheet (absent in Phase-2), so they're skipped — we
# stub outcomes.csv + probs_avg.csv ourselves so eda_descriptive +
# cleanup_features (which produce the cand_* features that the
# production models require) can run.
FEATURE_SCRIPTS = [
    "src/features/extract_structured.py",
    "src/features/extract_time_features.py",
    "src/features/extract_differentials.py",
    "src/features/extract_note_features.py",
    "src/features/extract_note_4h_features.py",
    "src/features/extract_v6_features.py",
]

# Post-feature scripts that produce the cand_* family.
# We don't run eda_descriptive.py here — it does correlation/heatmap
# analysis against outcomes that don't exist for Phase-2. Instead we
# call its self-contained build_candidates() function directly.
POST_FEATURE_SCRIPTS = [
    "src/features/cleanup_features.py",  # merges cand_* + sentinel
]

# Phase-2 artifacts we want to keep aside under derived/phase2/.
PHASE2_KEEP = [
    "features_triage.csv",
    "features_fourh.csv",
    "exploratory_features.csv",
    "task1_drug_predictions_phase2.csv",
    "probs_avg.csv",
    "task2_disposition_predictions_phase2.csv",
]


def write_exploratory_features() -> None:
    """Compute cand_* features via eda_descriptive.build_candidates()
    and write derived/exploratory_features.csv. Self-contained: needs
    only features_triage.csv + features_fourh.csv (no outcomes / probs).
    """
    sys.path.insert(0, str(ROOT / "src"))
    from eda.eda_descriptive import build_candidates  # type: ignore
    triage = pd.read_csv(DERIVED / "features_triage.csv")
    fourh = pd.read_csv(DERIVED / "features_fourh.csv")
    cand = build_candidates(triage, fourh)
    out = DERIVED / "exploratory_features.csv"
    cand.to_csv(out, index=False)
    n_cand = sum(1 for c in cand.columns if c.startswith("cand_"))
    print(f"  build_candidates: wrote {n_cand} cand_* features "
          f"for {len(cand)} encounters -> {out.name}")


def write_stub_outcomes_and_probs() -> None:
    """Create placeholder outcomes.csv and probs_avg.csv so that
    eda_descriptive + cleanup_features can run. The cand_* features
    they produce are derived from triage features only and don't use
    these stub values."""
    triage = pd.read_csv(DERIVED / "features_triage.csv")[["encounter_id"]]
    stub_outcomes = triage.copy()
    stub_outcomes["ground_truth_drug"] = 0
    stub_outcomes["ground_truth_drug_name"] = "None"
    stub_outcomes["encounter_disposition_label"] = "Discharge"
    stub_outcomes.to_csv(DERIVED / "outcomes.csv", index=False)
    stub_probs = triage.copy()
    for c in ("p_kraken", "p_triton", "p_coral", "p_none"):
        stub_probs[c] = 0.25
    stub_probs.to_csv(DERIVED / "probs_avg.csv", index=False)
    # eda_descriptive also reads ground_truth.csv; write a minimal stub.
    stub_gt = triage.copy()
    stub_gt["ground_truth_drug"] = 0
    stub_gt["ground_truth_drug_name"] = "None"
    stub_gt.to_csv(DERIVED / "ground_truth.csv", index=False)
    print(f"  wrote stub outcomes.csv, probs_avg.csv, ground_truth.csv "
          f"for {len(triage)} encounters")


def run(script_or_args, check: bool = True) -> int:
    """Run a script (string path) or a list of CLI args."""
    if isinstance(script_or_args, str):
        cmd = [sys.executable, str(ROOT / script_or_args)]
        label = script_or_args
    else:
        cmd = [sys.executable] + script_or_args
        label = " ".join(script_or_args)
    print(f"\n[run] {label}")
    print("-" * 72)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"[run] rc={rc}")
    if check and rc != 0:
        raise RuntimeError(f"Step failed: {label} (rc={rc})")
    return rc


def snapshot_derived() -> None:
    if SNAPSHOT.exists():
        print(f"Snapshot already at {SNAPSHOT}; removing first")
        shutil.rmtree(SNAPSHOT)
    print(f"Snapshotting derived/ -> {SNAPSHOT}")
    shutil.copytree(DERIVED, SNAPSHOT)


def restore_derived() -> None:
    """Restore Phase-1 derived/ from snapshot, surviving Windows file
    locks. Strategy: rather than rmtree(derived) (which fails if
    anything's holding a file handle), we move derived/ aside to a
    timestamped folder, copytree the snapshot back, and only then try
    to clean up the moved-aside Phase-2 derived (best-effort)."""
    import time
    print(f"\nRestoring derived/ from {SNAPSHOT}")
    # Move Phase-2 outputs we want to keep aside FIRST.
    keep_tmp = ROOT / "_phase2_outputs_tmp"
    if keep_tmp.exists():
        shutil.rmtree(keep_tmp, ignore_errors=True)
    keep_tmp.mkdir(exist_ok=True)
    for fn in PHASE2_KEEP:
        src = DERIVED / fn
        if src.exists():
            shutil.copy2(src, keep_tmp / fn)

    # Move-then-copy instead of rmtree-then-copy: tolerates file locks.
    parked = ROOT / f"_phase2_derived_parked_{int(time.time())}"
    try:
        shutil.move(str(DERIVED), str(parked))
    except (PermissionError, OSError) as e:
        print(f"  WARN: could not move derived/ aside ({e}); "
              f"trying contents-only swap")
        # Fall back: empty derived, then copytree snapshot contents in
        for child in DERIVED.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
            except Exception:
                pass
        for child in SNAPSHOT.iterdir():
            dst = DERIVED / child.name
            if child.is_dir():
                shutil.copytree(child, dst)
            else:
                shutil.copy2(child, dst)
        shutil.rmtree(SNAPSHOT, ignore_errors=True)
    else:
        shutil.copytree(SNAPSHOT, DERIVED)
        shutil.rmtree(SNAPSHOT, ignore_errors=True)
        shutil.rmtree(parked, ignore_errors=True)

    # Move the kept Phase-2 outputs to derived/phase2/.
    phase2_dir = DERIVED / "phase2"
    phase2_dir.mkdir(exist_ok=True)
    for fn in PHASE2_KEEP:
        src = keep_tmp / fn
        if src.exists():
            shutil.move(str(src), str(phase2_dir / fn))
    shutil.rmtree(keep_tmp, ignore_errors=True)
    print(f"  derived/phase2/ contains: "
          f"{sorted(p.name for p in phase2_dir.iterdir())}")


def stage_phase2_xlsx() -> None:
    src = DATA2 / PHASE2_XLSX_NAME
    dst = DATA / PHASE1_XLSX_NAME
    if not src.exists():
        raise FileNotFoundError(f"Phase-2 xlsx not found: {src}")
    print(f"Backing up Phase-1 xlsx -> {XLSX_BAK}")
    if XLSX_BAK.exists():
        XLSX_BAK.unlink()
    shutil.move(str(dst), str(XLSX_BAK))
    print(f"Staging Phase-2 xlsx as {dst.name}")
    shutil.copy2(src, dst)


def restore_xlsx() -> None:
    dst = DATA / PHASE1_XLSX_NAME
    if not XLSX_BAK.exists():
        print(f"No xlsx backup to restore at {XLSX_BAK}")
        return
    if dst.exists():
        dst.unlink()
    shutil.move(str(XLSX_BAK), str(dst))
    print(f"Restored xlsx -> {dst}")


def build_probs_avg(task1_pred_csv: Path, out_csv: Path) -> None:
    """Convert the task1 production predict output into probs_avg.csv
    schema (encounter_id + p_kraken/p_triton/p_coral/p_none).

    The task2 production model expects the four columns to sum to 1
    per row; the cascade soft probabilities already do.
    """
    t1 = pd.read_csv(task1_pred_csv)
    needed = ["encounter_id", "p_kraken", "p_triton", "p_coral", "p_none"]
    missing = [c for c in needed if c not in t1.columns]
    if missing:
        raise ValueError(f"Task-1 predictions missing columns {missing}")
    out = t1[needed].copy()
    out.to_csv(out_csv, index=False)
    print(f"  probs_avg.csv: {len(out)} rows, "
          f"sum-check max-deviation = "
          f"{(out[['p_kraken','p_triton','p_coral','p_none']].sum(axis=1) - 1).abs().max():.2e}")


def main() -> None:
    if not (DATA2 / PHASE2_XLSX_NAME).exists():
        sys.exit(f"Phase-2 xlsx not found: {DATA2 / PHASE2_XLSX_NAME}")

    snapshot_derived()
    stage_phase2_xlsx()
    try:
        for script in FEATURE_SCRIPTS:
            run(script)

        # Produce cand_* features inline (replaces eda_descriptive's
        # build_candidates call) + stub outcomes for sentinel-only
        # checks in cleanup_features.
        print("\n[cand] computing cand_* features via "
              "eda_descriptive.build_candidates()")
        write_exploratory_features()
        print("[stub] writing placeholder outcomes for the "
              "leakage sentinel in cleanup_features")
        write_stub_outcomes_and_probs()
        for script in POST_FEATURE_SCRIPTS:
            run(script)

        # Task-1: features_triage -> drug predictions (per-encounter
        # probability columns we'll reshape into probs_avg).
        task1_pred = DERIVED / "task1_drug_predictions_phase2.csv"
        run([
            "production/task1/predict.py",
            str(DERIVED / "features_triage.csv"),
            str(task1_pred),
        ])

        # probs_avg.csv built from task1 outputs (substitute for the
        # LLM-agent consensus the original Task-2 pipeline expects).
        build_probs_avg(task1_pred, DERIVED / "probs_avg.csv")

        # Task-2: features_fourh + probs_avg -> disposition predictions.
        task2_pred = DERIVED / "task2_disposition_predictions_phase2.csv"
        run([
            "production/task2/predict.py",
            str(DERIVED / "features_fourh.csv"),
            str(DERIVED / "probs_avg.csv"),
            str(task2_pred),
        ])

        # Quick summary
        t1 = pd.read_csv(task1_pred)
        t2 = pd.read_csv(task2_pred)
        print("\n" + "=" * 72)
        print("Phase-2 prediction summary")
        print("=" * 72)
        print(f"Encounters scored: {len(t1)} (task1)   {len(t2)} (task2)")
        print(f"Task-1 drug_class distribution:")
        print(t1["drug_class"].value_counts().sort_index()
              .rename(index={0:"None",1:"Kraken",2:"Triton",3:"Coral"})
              .to_string())
        print(f"\nTask-2 disposition_class distribution:")
        print(t2["disposition_label"].value_counts().to_string())
    finally:
        restore_derived()
        restore_xlsx()


if __name__ == "__main__":
    main()
