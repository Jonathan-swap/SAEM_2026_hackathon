"""Cascade-B extended to 5 classes (binary-cascade with prevalence
terminal stage).

Stack (assumes class indices 0=None, 1=Kraken, 2=Triton, 3=Coral,
4=NEW — rename via CLASS_NAMES / EXTRA_CLASS_NAME once the manual
labels arrive):

  Tier 1 (drug-vs-no-drug):     P(class != 0 | X)         ← τ_drug
  Tier 2 (Kraken-vs-rest):      P(class == 1 | drug+, X)  ← τ_kraken
  Tier 3 (Triton-vs-rest):      P(class == 2 |
                                 non-Kraken drug+, X)      ← τ_triton
  Stage 4 (Coral vs NEW):       per-encounter Bernoulli matching
                                training prevalence of class 3
                                within {3, 4} (deterministic via md5
                                of encounter_id — same idea as the
                                4-class Cascade-B Triton-vs-Coral
                                terminal stage).

If you'd rather peel a *different* class at Tier 3 (e.g. if class 4
turns out to be a more discriminable "Polydrug" that should beat T-vs-C
splitting), change the variable TIER3_TARGET_CLASS below and the
terminal stage will pair the remaining two classes by prevalence.

Inputs:
  derived/features_triage.csv
  derived/outcomes.csv  with a 5-level `ground_truth_drug`  OR
  derived/ground_truth_5class.csv  (pass --labels 5class)

Outputs (per run):
  derived/task1_cascade_5class_threshold_grid.csv      every cell
  derived/task1_cascade_5class_threshold_picked.csv    3 picked points
  derived/task1_cascade_5class_threshold_labels.csv    per-encounter probs
  derived/task1_cascade_5class_threshold_report.md     human-readable report

Threshold grid: τ_drug, τ_kraken, τ_triton each on [0.05, 0.95] step
0.05 by default → 19^3 = 6859 cells per (cv, holdout) split (fast).
Step can be tightened with --step.

Time-leakage: identical to Cascade-B (features_triage.csv only, no
4h, no narrative). Same 5-fold CV (random_state=42) and temporal
holdout (last day) as the rest of Task-1.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

# ---- Class indices and names. Rename EXTRA_CLASS_NAME once labels land.
NONE_IDX, KRAKEN_IDX, TRITON_IDX, CORAL_IDX, NEW_IDX = 0, 1, 2, 3, 4
EXTRA_CLASS_NAME = "Class 4"
CLASS_NAMES = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust",
               EXTRA_CLASS_NAME]
TIER3_TARGET_CLASS = TRITON_IDX  # class peeled at Tier 3
TERMINAL_PAIR = (CORAL_IDX, NEW_IDX)  # classes left for the prevalence stage

TEXT_COL = "triage_brief_note"


# ---------- Data + preprocessor (mirrors threshold_cascade_b.py) -------

def load_features_and_y(labels_source: str = "auto"
                         ) -> tuple[pd.DataFrame, np.ndarray, pd.Series,
                                     np.ndarray]:
    X = pd.read_csv(DERIVED / "features_triage.csv")

    if labels_source == "5class":
        labels_path = DERIVED / "ground_truth_5class.csv"
        if not labels_path.exists():
            raise SystemExit(
                f"--labels 5class requested but {labels_path} not present. "
                "Drop the manual 5-class labels there with columns "
                "[encounter_id, ground_truth_drug_5class, ...].")
        outcomes = pd.read_csv(labels_path)[
            ["encounter_id", "ground_truth_drug_5class"]]
        outcomes = outcomes.rename(columns={
            "ground_truth_drug_5class": "ground_truth_drug"})
    else:
        outcomes = pd.read_csv(DERIVED / "outcomes.csv")[
            ["encounter_id", "ground_truth_drug"]]

    for c in ("encounter_disposition_label", "ground_truth_drug",
               "ground_truth_drug_name", "ground_truth_drug_5class",
               "ground_truth_drug_5class_name"):
        if c in X.columns:
            X = X.drop(columns=[c])
    df = X.merge(outcomes, on="encounter_id", how="inner")
    drop = ["encounter_id", "encounter_arrival_date", "ground_truth_drug"]
    drop = [c for c in drop if c in df.columns]
    y = df["ground_truth_drug"].astype(int).to_numpy()
    arrival = df.get("encounter_arrival_date", pd.Series([None] * len(df)))
    ids = df["encounter_id"].to_numpy()
    return df.drop(columns=drop), y, arrival, ids


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    text_col = TEXT_COL if TEXT_COL in X.columns else None
    obj_cols = [c for c in X.select_dtypes(include=["object", "string"]).columns
                if c != text_col]
    num_cols = (X.select_dtypes(include="number").columns.tolist()
                + X.select_dtypes(include="bool").columns.tolist())
    num_cols = list(set(num_cols))
    transformers = []
    if text_col:
        transformers.append(("text", TfidfVectorizer(
            max_features=50, stop_words="english", ngram_range=(1, 2),
            min_df=3), text_col))
    if obj_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant",
                                     fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False)),
        ]), obj_cols))
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols))
    return ColumnTransformer(transformers, remainder="drop")


def make_rforest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


# ---------- Tiered binary fits ----------------------------------------

def _fit_binary(X_tr: np.ndarray, y_tr_5cls: np.ndarray, X_te: np.ndarray,
                positive_class: int, restrict_mask: np.ndarray | None = None
                ) -> np.ndarray:
    """Fit a binary classifier and return P(positive_class) on X_te.

    If restrict_mask is given, training is filtered to the subset where
    the mask is True (this is how we condition tier-K on the previous
    tiers' negatives, e.g. "Kraken given drug+")."""
    if restrict_mask is None:
        X_use = X_tr
        y_use = (y_tr_5cls == positive_class).astype(int)
    else:
        X_use = X_tr[restrict_mask]
        y_use = (y_tr_5cls[restrict_mask] == positive_class).astype(int)
    mdl = make_rforest()
    mdl.fit(X_use, y_use)
    return mdl.predict_proba(X_te)[:, 1]


def fit_three_tiers(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_drug, p_kraken_given_drug, p_t3_given_non_kraken_drug)
    each shaped (n_te,). Triton vs Coral split happens at Stage 4."""
    p_drug = _fit_binary(X_tr, y_tr, X_te, positive_class=NONE_IDX) * -1 + 1
    # ^ predict the positive class directly: trick is unnecessary —
    # let's just rebuild with positive_class = "any non-None":
    # cleaner with an inline approach:
    y_drug = (y_tr != NONE_IDX).astype(int)
    mdl = make_rforest()
    mdl.fit(X_tr, y_drug)
    p_drug = mdl.predict_proba(X_te)[:, 1]

    drug_mask = (y_tr != NONE_IDX)
    p_kraken = _fit_binary(X_tr, y_tr, X_te,
                            positive_class=KRAKEN_IDX,
                            restrict_mask=drug_mask)

    non_k_drug_mask = drug_mask & (y_tr != KRAKEN_IDX)
    p_tier3 = _fit_binary(X_tr, y_tr, X_te,
                           positive_class=TIER3_TARGET_CLASS,
                           restrict_mask=non_k_drug_mask)
    return p_drug, p_kraken, p_tier3


# ---------- Terminal stage 4: prevalence-Bernoulli for the remaining pair

def _bernoulli_label(eid: str, p_first: float) -> int:
    """Deterministic per-encounter Bernoulli: md5(eid) -> uniform.
    Returns TERMINAL_PAIR[0] with probability p_first, else
    TERMINAL_PAIR[1]."""
    h = hashlib.md5(str(eid).encode()).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    return TERMINAL_PAIR[0] if u < p_first else TERMINAL_PAIR[1]


# ---------- Hard-cascade label assembly --------------------------------

def cascade_labels(p_drug: np.ndarray, p_kraken: np.ndarray,
                    p_tier3: np.ndarray, encounter_ids: np.ndarray,
                    tau_drug: float, tau_k: float, tau_t3: float,
                    prev_first_terminal: float) -> np.ndarray:
    """Walk the cascade per encounter and return integer labels 0..4."""
    out = np.empty(len(p_drug), dtype=int)
    for i in range(len(p_drug)):
        if p_drug[i] < tau_drug:
            out[i] = NONE_IDX
            continue
        if p_kraken[i] >= tau_k:
            out[i] = KRAKEN_IDX
            continue
        if p_tier3[i] >= tau_t3:
            out[i] = TIER3_TARGET_CLASS
            continue
        out[i] = _bernoulli_label(encounter_ids[i], prev_first_terminal)
    return out


# ---------- 5-fold OOF probabilities + terminal prevalence -------------

def cv_three_tiers(X_df: pd.DataFrame, y: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """5-fold OOF: stack tier-1, tier-2, tier-3 probabilities. Returns
    (p_drug_oof, p_kraken_oof, p_tier3_oof, prev_first_terminal)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_drug = np.zeros(len(y))
    p_kraken = np.zeros(len(y))
    p_tier3 = np.zeros(len(y))
    for tr, te in skf.split(X_df, y):
        pre = make_preprocessor(X_df)
        pre.fit(X_df.iloc[tr])
        X_tr = np.asarray(pre.transform(X_df.iloc[tr]), dtype=float)
        X_te = np.asarray(pre.transform(X_df.iloc[te]), dtype=float)
        a, b, c = fit_three_tiers(X_tr, y[tr], X_te)
        p_drug[te] = a
        p_kraken[te] = b
        p_tier3[te] = c
    # Prevalence of the first terminal class within the terminal pair,
    # computed on the full training cohort (a stable constant for a
    # fixed label set).
    terminal_mask = np.isin(y, TERMINAL_PAIR)
    if terminal_mask.sum() == 0:
        prev_first = 0.5
    else:
        prev_first = float((y[terminal_mask] == TERMINAL_PAIR[0]).mean())
    return p_drug, p_kraken, p_tier3, prev_first


def temporal_three_tiers(X_df: pd.DataFrame, y: np.ndarray,
                          arrival: pd.Series
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray, float]:
    """Train on every day except the last; predict on the last day.
    Returns (p_drug_te, p_kraken_te, p_tier3_te, y_te, ids_te,
    prev_first)."""
    dates = pd.to_datetime(arrival)
    last = dates.dt.date.max()
    is_test = (dates.dt.date == last).to_numpy()
    is_train = ~is_test
    pre = make_preprocessor(X_df)
    pre.fit(X_df.iloc[is_train])
    X_tr = np.asarray(pre.transform(X_df.iloc[is_train]), dtype=float)
    X_te = np.asarray(pre.transform(X_df.iloc[is_test]), dtype=float)
    a, b, c = fit_three_tiers(X_tr, y[is_train], X_te)
    terminal_mask = np.isin(y[is_train], TERMINAL_PAIR)
    if terminal_mask.sum() == 0:
        prev_first = 0.5
    else:
        prev_first = float((y[is_train][terminal_mask]
                              == TERMINAL_PAIR[0]).mean())
    ids = X_df.iloc[is_test].index.to_numpy()  # positional fallback
    return a, b, c, y[is_test], ids, prev_first


# ---------- Grid search over the three thresholds ---------------------

def grid_score(p_drug: np.ndarray, p_k: np.ndarray, p_t3: np.ndarray,
               y_true: np.ndarray, ids: np.ndarray,
               prev_first: float, taus: np.ndarray,
               split_name: str, K: int) -> pd.DataFrame:
    rows = []
    labels_idx = list(range(K))
    for td in taus:
        for tk in taus:
            for tt3 in taus:
                yp = cascade_labels(p_drug, p_k, p_t3, ids,
                                     tau_drug=td, tau_k=tk, tau_t3=tt3,
                                     prev_first_terminal=prev_first)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, yp, labels=labels_idx, average=None,
                    zero_division=0)
                macro_f1 = float(np.nanmean(f1))
                min_f1 = float(np.nanmin(f1)) if len(f1) else float("nan")
                acc = float(accuracy_score(y_true, yp))
                rows.append({
                    "split": split_name,
                    "tau_drug": td, "tau_kraken": tk, "tau_tier3": tt3,
                    "accuracy": acc, "macro_f1": macro_f1,
                    "min_class_f1": min_f1,
                    **{f"f1_class_{i}": float(f1[i]) for i in range(K)},
                    **{f"prec_class_{i}": float(prec[i]) for i in range(K)},
                    **{f"rec_class_{i}": float(rec[i]) for i in range(K)},
                })
    return pd.DataFrame(rows)


def pick(df: pd.DataFrame) -> pd.DataFrame:
    picks = []
    for crit in ("macro_f1", "accuracy", "min_class_f1"):
        row = df.iloc[df[crit].idxmax()].copy()
        row["pick_criterion"] = crit
        picks.append(row)
    return pd.DataFrame(picks)


# ---------- Main -------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", choices=["auto", "5class"], default="auto",
                    help="auto: read outcomes.csv. 5class: read "
                         "derived/ground_truth_5class.csv.")
    ap.add_argument("--step", type=float, default=0.05,
                    help="Threshold-grid step (default 0.05 — 19^3 cells).")
    args = ap.parse_args()

    X_df, y, arrival, ids = load_features_and_y(args.labels)
    K = int(y.max() + 1)
    if K != 5:
        print(f"WARN: ground_truth_drug has {K} levels (expected 5). "
              "Running anyway; the cascade Tier-3 / Stage-4 will still "
              "work if K >= 4 — but the report's 5th-class numbers will "
              "be empty.")

    classes_used = CLASS_NAMES[:K]
    print(f"Loaded {len(y)} encounters, K={K} classes -> {classes_used}")
    print(f"Class distribution:")
    for k in range(K):
        n = int((y == k).sum())
        print(f"  {k} ({classes_used[k]:<14s}): {n} ({n/len(y)*100:.1f}%)")

    # ---- 5-fold OOF tier probabilities ----
    print("\n=== 5-fold OOF: tier-1, tier-2, tier-3 ===")
    p_drug_cv, p_k_cv, p_t3_cv, prev_cv = cv_three_tiers(X_df, y)
    print(f"Training prevalence of {CLASS_NAMES[TERMINAL_PAIR[0]]} within "
          f"terminal pair {TERMINAL_PAIR}: {prev_cv:.3f}")

    # ---- Temporal holdout ----
    print("\n=== Temporal holdout: tier-1, tier-2, tier-3 ===")
    p_drug_te, p_k_te, p_t3_te, y_te, ids_te, prev_te = (
        temporal_three_tiers(X_df, y, arrival))
    is_test = pd.to_datetime(arrival).dt.date == \
              pd.to_datetime(arrival).dt.date.max()
    eids_te = pd.Series(ids).iloc[is_test.to_numpy()].to_numpy()
    print(f"Holdout n={len(y_te)}. Terminal prevalence on train fold: "
          f"{prev_te:.3f}")

    # ---- Grid search ----
    taus = np.round(np.arange(0.05, 0.96, args.step), 4)
    print(f"\nThreshold grid: {len(taus)}^3 = {len(taus)**3} cells "
          f"per split (step={args.step}).")

    grid_cv = grid_score(p_drug_cv, p_k_cv, p_t3_cv, y, ids, prev_cv,
                          taus, "cv", K)
    grid_te = grid_score(p_drug_te, p_k_te, p_t3_te, y_te, eids_te,
                          prev_te, taus, "temporal", K)

    # Pick on CV, apply same triplet to holdout for an honest read.
    picks_cv = pick(grid_cv)
    print("\n=== Picked triplets (chosen on CV) ===")
    print(picks_cv[["pick_criterion", "tau_drug", "tau_kraken",
                     "tau_tier3", "accuracy", "macro_f1",
                     "min_class_f1"]].to_string(index=False))

    # Reapply CV picks to holdout
    holdout_for_picks = []
    for _, p in picks_cv.iterrows():
        yp = cascade_labels(p_drug_te, p_k_te, p_t3_te, eids_te,
                             tau_drug=p["tau_drug"], tau_k=p["tau_kraken"],
                             tau_t3=p["tau_tier3"],
                             prev_first_terminal=prev_te)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, yp, labels=list(range(K)), average=None, zero_division=0)
        holdout_for_picks.append({
            "split": "temporal_applied_from_cv_pick",
            "pick_criterion": p["pick_criterion"],
            "tau_drug": p["tau_drug"], "tau_kraken": p["tau_kraken"],
            "tau_tier3": p["tau_tier3"],
            "accuracy": float(accuracy_score(y_te, yp)),
            "macro_f1": float(np.nanmean(f1)),
            "min_class_f1": float(np.nanmin(f1)),
            **{f"f1_class_{i}": float(f1[i]) for i in range(K)},
        })
    picks_applied = pd.DataFrame(holdout_for_picks)
    print("\n=== Same triplets applied to temporal holdout ===")
    print(picks_applied[["pick_criterion", "accuracy", "macro_f1",
                           "min_class_f1"]].to_string(index=False))

    # ---- Save artifacts ----
    out_grid = DERIVED / "task1_cascade_5class_threshold_grid.csv"
    out_picked = DERIVED / "task1_cascade_5class_threshold_picked.csv"
    out_labels = DERIVED / "task1_cascade_5class_threshold_labels.csv"
    out_report = DERIVED / "task1_cascade_5class_threshold_report.md"

    pd.concat([grid_cv, grid_te], ignore_index=True).to_csv(out_grid,
                                                              index=False)
    pd.concat([picks_cv.assign(split="cv"), picks_applied],
              ignore_index=True).to_csv(out_picked, index=False)

    # Per-encounter labels using the macro-F1 pick on CV (the headline)
    headline = picks_cv.iloc[picks_cv["macro_f1"].idxmax()]
    yp_oof = cascade_labels(p_drug_cv, p_k_cv, p_t3_cv, ids, prev_cv,
                              tau_drug=headline["tau_drug"],
                              tau_k=headline["tau_kraken"],
                              tau_t3=headline["tau_tier3"])
    labels_df = pd.DataFrame({
        "encounter_id": ids,
        "true_label": y,
        "p_drug_oof": p_drug_cv,
        "p_kraken_given_drug_oof": p_k_cv,
        "p_tier3_given_non_kraken_drug_oof": p_t3_cv,
        "prev_first_terminal": prev_cv,
        "cascade_label_macroF1_pick": yp_oof,
    })
    labels_df.to_csv(out_labels, index=False)

    with open(out_report, "w") as f:
        f.write("# Task-1 5-class binary cascade — threshold sweep\n\n")
        f.write(f"Hierarchy:\n\n")
        f.write(f"- Tier 1 — drug-vs-no-drug "
                 f"(positive = class != {NONE_IDX}/None).\n")
        f.write(f"- Tier 2 — Kraken-vs-non-Kraken-drug.\n")
        f.write(f"- Tier 3 — `{CLASS_NAMES[TIER3_TARGET_CLASS]}` "
                 f"vs rest within non-Kraken drug-positive.\n")
        f.write(f"- Stage 4 — `{CLASS_NAMES[TERMINAL_PAIR[0]]}` vs "
                 f"`{CLASS_NAMES[TERMINAL_PAIR[1]]}` split by training "
                 f"prevalence ({prev_cv:.3f}/"
                 f"{1 - prev_cv:.3f}) per-encounter Bernoulli.\n\n")
        f.write(f"Class distribution (n={len(y)}):\n\n")
        for k in range(K):
            n = int((y == k).sum())
            f.write(f"- {k} ({classes_used[k]}): {n} "
                     f"({n / len(y) * 100:.1f}%)\n")
        f.write("\n## Picked thresholds (chosen on 5-fold OOF)\n\n")
        f.write(picks_cv[["pick_criterion", "tau_drug", "tau_kraken",
                           "tau_tier3", "accuracy", "macro_f1",
                           "min_class_f1"]].to_markdown(index=False))
        f.write("\n\n## Same thresholds applied to temporal holdout\n\n")
        f.write(picks_applied[["pick_criterion", "accuracy", "macro_f1",
                                "min_class_f1"]].to_markdown(index=False))
        f.write("\n")
    print(f"\nSaved:")
    print(f"  {out_grid}")
    print(f"  {out_picked}")
    print(f"  {out_labels}")
    print(f"  {out_report}")


if __name__ == "__main__":
    main()
