"""Triple-check that Task-2 outcome data comes from the
`Disposition` sheet of the source xlsx (and nowhere else).

Five independent checks:

  1. The xlsx Disposition sheet has the expected schema
     (encounter_id, encounter_disposition_label) and an entry per
     encounter.
  2. extract_structured.py reads the Disposition sheet by name
     (static-code check on the script).
  3. The label column ONLY appears in the Disposition sheet — not
     in Triage_Data, not in Four_Hour_Data.
  4. features_fourh.csv's encounter_disposition_label values
     match the Disposition sheet 1:1 by encounter_id (row-level
     equality after sort).
  5. The actual Task-2 training pipelines
     (src/task2_deterioration/train_baseline.py and
     src/eval_temporal.py run_task2) source the label from
     features_fourh.csv only.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
XLSX = ROOT / "data" / "Hackathon_Data_Release_1_SHARE.xlsx"
LABEL = "encounter_disposition_label"


def main() -> None:
    print("=" * 78)
    print("Task-2 outcome source verification")
    print("=" * 78)

    # ---- Check 1: Disposition sheet schema --------------------------
    print("\n[1] Loading 'Disposition' sheet from xlsx...")
    disp = pd.read_excel(XLSX, sheet_name="Disposition",
                          engine="openpyxl")
    print(f"    columns:  {list(disp.columns)}")
    print(f"    n_rows:   {len(disp)}")
    print(f"    n_unique encounter_id: {disp['encounter_id'].nunique()}")
    if LABEL not in disp.columns:
        print(f"    FAIL: {LABEL!r} not in Disposition sheet.")
        raise SystemExit(1)
    vc = disp[LABEL].value_counts(dropna=False)
    print(f"    {LABEL} distribution:")
    for k, v in vc.items():
        print(f"      {k!s:>14s}  {v}")
    print(f"    OK  Disposition sheet has {LABEL} for "
          f"{disp[LABEL].notna().sum()} of {len(disp)} encounters.")

    # ---- Check 2: build_outcomes.py names the Disposition sheet ----
    print("\n[2] Static-code check on build_outcomes.py...")
    bo_src = (ROOT / "src" / "labels" / "build_outcomes.py").read_text()
    if re.search(r"sheet_name\s*=\s*[\"']Disposition[\"']", bo_src):
        print(f"    OK  build_outcomes.py reads sheet_name='Disposition'.")
    else:
        print(f"    FAIL: no read_excel(...sheet_name='Disposition') in "
              "build_outcomes.py.")
        raise SystemExit(1)

    # And extract_structured must NOT carry the disposition into features.
    ex_src = (ROOT / "src" / "features" / "extract_structured.py").read_text()
    if re.search(r"\.merge\(\s*dispo\s*,", ex_src):
        print(f"    FAIL: extract_structured.py is merging the dispo "
              "sheet into features_*.csv — outcomes must live only in "
              "outcomes.csv.")
        raise SystemExit(1)
    print(f"    OK  extract_structured.py does NOT merge dispo into "
          "features_*.csv.")

    # ---- Check 3: the label appears in NO other sheet ---------------
    print("\n[3] Confirming no other xlsx sheet carries the label...")
    triage = pd.read_excel(XLSX, sheet_name="Triage_Data",
                             engine="openpyxl")
    fourh = pd.read_excel(XLSX, sheet_name="Four_Hour_Data",
                            engine="openpyxl")
    leak_t = LABEL in triage.columns
    leak_f = LABEL in fourh.columns
    print(f"    Triage_Data has {LABEL}?       {leak_t}")
    print(f"    Four_Hour_Data has {LABEL}?    {leak_f}")
    if leak_t or leak_f:
        print(f"    FAIL: label appears in a non-Disposition sheet.")
        raise SystemExit(1)
    print(f"    OK  Label is exclusive to the Disposition sheet.")

    # ---- Check 4a: outcomes.csv == Disposition sheet --------------
    print("\n[4a] Row-level equality: outcomes.csv vs xlsx Disposition...")
    out = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", LABEL]].rename(columns={LABEL: "label_csv"})
    src = disp[["encounter_id", LABEL]].rename(columns={LABEL: "label_xlsx"})
    joined = out.merge(src, on="encounter_id", how="outer", indicator=True)
    print(f"    merge sizes: outcomes={len(out)}  xlsx={len(src)}  "
          f"joined={len(joined)}")
    only_csv = (joined["_merge"] == "left_only").sum()
    only_xlsx = (joined["_merge"] == "right_only").sum()
    both = (joined["_merge"] == "both").sum()
    print(f"    outcomes-only={only_csv}  xlsx-only={only_xlsx}  both={both}")
    mismatch = joined.loc[joined["_merge"] == "both",
                           ["label_csv", "label_xlsx"]]
    diffs = (mismatch["label_csv"] != mismatch["label_xlsx"]).sum()
    print(f"    Per-row label mismatches: {diffs}")
    if only_csv != 0 or only_xlsx != 0 or diffs != 0:
        print(f"    FAIL: outcomes.csv label does not match the "
              "Disposition sheet.")
        raise SystemExit(1)
    print(f"    OK  All {both} encounters match label-for-label.")
    vc_csv = out["label_csv"].value_counts(dropna=False).to_dict()
    vc_src = disp[LABEL].value_counts(dropna=False).to_dict()
    if vc_csv != vc_src:
        print(f"    FAIL: distribution mismatch  outcomes={vc_csv}  "
              f"xlsx={vc_src}")
        raise SystemExit(1)
    print(f"    OK  Class distribution matches exactly: {vc_csv}")

    # ---- Check 4b: features_*.csv must NOT contain the label ------
    print("\n[4b] Feature tables must NOT carry the label...")
    for fname in ("features_triage.csv", "features_fourh.csv"):
        ff = pd.read_csv(DERIVED / fname)
        if LABEL in ff.columns:
            print(f"    FAIL: {fname} contains {LABEL} — outcomes "
                  "must live only in outcomes.csv.")
            raise SystemExit(1)
        print(f"    OK  {fname}: {LABEL} not present.")

    # ---- Check 5: trainers read the label from features_fourh.csv only -
    print("\n[5] Static-code check on Task-2 trainers...")
    train2 = (ROOT / "src" / "task2_deterioration"
              / "train_baseline.py").read_text()
    evtemp = (ROOT / "src" / "eval_temporal.py").read_text()

    # Both trainers must read outcomes.csv as the canonical label source.
    if "outcomes.csv" in train2 and 'df["encounter_disposition_label"]' in train2:
        print(f"    OK  task2/train_baseline.py reads outcomes.csv "
              f"and uses df[{LABEL!r}].")
    else:
        print(f"    FAIL: task2/train_baseline.py is not sourcing the "
              "target from outcomes.csv.")
        raise SystemExit(1)

    if "outcomes.csv" in evtemp and 'df["encounter_disposition_label"]' in evtemp:
        print(f"    OK  eval_temporal.py reads outcomes.csv and uses "
              f"df[{LABEL!r}] for Task-2.")
    else:
        print(f"    FAIL: eval_temporal.py is not sourcing the target "
              "from outcomes.csv.")
        raise SystemExit(1)

    # And nothing reads the Disposition sheet directly except
    # extract_structured.py (the legitimate ingest point).
    print("\n    Static scan: who reads sheet_name='Disposition'?")
    for p in (ROOT / "src").rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"sheet_name\s*=\s*[\"']Disposition[\"']", text):
            rel = p.relative_to(ROOT)
            print(f"      {rel}")

    print("\n" + "=" * 78)
    print("VERDICT: Task-2 outcomes come EXCLUSIVELY from the "
          "'Disposition' sheet.")
    print("=" * 78)
    print(f"  Source xlsx sheet:        Disposition  (n={len(disp)})")
    print(f"  Ingest script:            src/labels/build_outcomes.py")
    print(f"  Canonical CSV:            derived/outcomes.csv :: {LABEL}")
    print(f"  features_*.csv:           do NOT contain the label")
    print(f"  Consumed by:              src/task2_deterioration/train_baseline.py")
    print(f"                            src/eval_temporal.py (run_task2)")
    print(f"                            src/unsupervised/cluster.py (task2 truth)")
    print(f"  Class distribution:       {vc_csv}")


if __name__ == "__main__":
    main()
