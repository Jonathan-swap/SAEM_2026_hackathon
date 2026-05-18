"""Cascade-B threshold tradeoff — grid-search the best (τ_drug, τ_kraken)
pair, retain every 4-class probability per encounter, emit a final
outcome candidate label.

Cascade-B (RUNBOOK §7h deployment champion):
  Stage 1 (tier-1):    P(drug-positive | X)            ← THRESHOLD τ_drug
  Stage 2 (K-vs-rest): P(Kraken | drug-positive, X)    ← THRESHOLD τ_kraken
  Stage 3 (T-vs-C):    training-set prevalence         ← PER-ENCOUNTER
                                                          BERNOULLI MATCHING
                                                          THE PREVALENCE

Stage-3 has no discriminating model — that's the §7g ceiling, Triton
vs Coral cannot be told apart from triage data. Instead of collapsing
to a single majority class, each non-Kraken drug-positive encounter is
assigned Triton with probability = training prevalence and Coral
otherwise. The draw is deterministic per encounter (md5 hash of
encounter_id → uniform → Bernoulli), so:

  - Marginal Triton/Coral output distribution matches training prevalence.
  - Same encounter always gets the same T/C label across re-runs and
    across grid cells.
  - No RNG state to manage during the grid search.

Approach:
  1. 5-fold OOF for tier-1 + K-vs-rest (rforest, the best Cascade-B
     model per §7h, holdout macro AUC 0.721).
  2. Compute Triton prevalence on the training-OOF non-Kraken drug+
     cohort.
  3. Grid-search (τ_drug, τ_kraken) ∈ [0.05, 0.95]² in 0.02 steps.
     For every pair, assemble hard-cascade labels and score macro F1,
     accuracy, and per-class F1 against ground truth.
  4. Pick three optima:
       - best macro F1   (the headline tradeoff point)
       - best accuracy   (overall correctness)
       - best min-class F1 (fairest — worst class is best)
  5. Apply each picked pair unchanged to the temporal holdout.
  6. Emit per-encounter rows with ALL probabilities retained:
       p_drug, p_kraken_given_drug, p_triton_prev,
       p_none, p_kraken, p_triton, p_coral   (the 4 soft probabilities,
                                              normalised to sum to 1)
       final_label                            (the hard-cascade pick)

Outputs:
  derived/task1_cascade_b_threshold_grid.csv    every (τ_drug, τ_kraken) cell
  derived/task1_cascade_b_threshold_picked.csv  3 picked pairs + per-cell metrics
  derived/task1_cascade_b_threshold_labels.csv  per-encounter probs + final label
  derived/task1_cascade_b_threshold_report.md   full markdown report
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
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

CLASS_NAMES = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]
NONE_IDX, KRAKEN_IDX, TRITON_IDX, CORAL_IDX = 0, 1, 2, 3

TEXT_COL = "triage_brief_note"

GRID_STEP = 0.02
GRID_LO, GRID_HI = 0.05, 0.95


# ---------- Data + preprocessor (matches compare_cascades.py) -----------

def load_features_and_y() -> tuple[pd.DataFrame, np.ndarray, pd.Series,
                                    np.ndarray]:
    X = pd.read_csv(DERIVED / "features_triage.csv")
    outcomes = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "ground_truth_drug"]]
    for c in ("encounter_disposition_label", "ground_truth_drug",
              "ground_truth_drug_name"):
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
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    num_cols = list(set(num_cols + bool_cols))
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


# ---------- Binary fits --------------------------------------------------

def _fit_tier1(X_tr, y_tr_4cls, X_te) -> np.ndarray:
    y = (y_tr_4cls != NONE_IDX).astype(int)
    mdl = make_rforest()
    mdl.fit(X_tr, y)
    return mdl.predict_proba(X_te)[:, 1]


def _fit_kraken_vs_rest(X_tr, y_tr_4cls, X_te) -> np.ndarray:
    mask = (y_tr_4cls != NONE_IDX)
    y = (y_tr_4cls[mask] == KRAKEN_IDX).astype(int)
    mdl = make_rforest()
    mdl.fit(X_tr[mask], y)
    return mdl.predict_proba(X_te)[:, 1]


def _triton_prev(y_tr_4cls) -> float:
    """Training-set fraction of Triton among non-Kraken drug+ rows."""
    mask = (y_tr_4cls == TRITON_IDX) | (y_tr_4cls == CORAL_IDX)
    if mask.sum() == 0:
        return 0.5
    return float((y_tr_4cls[mask] == TRITON_IDX).mean())


def stable_uniforms(encounter_ids: np.ndarray) -> np.ndarray:
    """Stable uniform[0,1) draws keyed on encounter_id (md5 of id).

    Same encounter_id always yields the same draw — independent of
    array order, grid cell, run, or platform. Used by the stage-3
    prevalence-based Bernoulli so each non-Kraken drug+ encounter
    gets a deterministic T/C label that respects the prevalence.
    """
    us = np.zeros(len(encounter_ids), dtype=np.float64)
    for i, eid in enumerate(encounter_ids):
        h = hashlib.md5(str(eid).encode("utf-8")).digest()
        us[i] = int.from_bytes(h[:8], "big") / float(1 << 64)
    return us


# ---------- OOF + holdout pipelines -------------------------------------

def run_oof(X_df, y) -> tuple[np.ndarray, np.ndarray, float]:
    """5-fold OOF tier-1 + K-vs-rest probabilities.

    Triton-prevalence is averaged across the 5 training folds (each
    fold computes its own training-set prevalence; we use the mean as
    the deployment scalar, which mirrors the CV-honest spirit).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_drug = np.zeros(len(y))
    p_kraken = np.zeros(len(y))
    prevs = []
    for fold, (tr, te) in enumerate(skf.split(X_df, y), start=1):
        pre = make_preprocessor(X_df)
        pre.fit(X_df.iloc[tr])
        X_tr = np.asarray(pre.transform(X_df.iloc[tr]), dtype=float)
        X_te = np.asarray(pre.transform(X_df.iloc[te]), dtype=float)
        p_drug[te] = _fit_tier1(X_tr, y[tr], X_te)
        p_kraken[te] = _fit_kraken_vs_rest(X_tr, y[tr], X_te)
        prevs.append(_triton_prev(y[tr]))
        print(f"  fold {fold}: train n={len(tr)}, test n={len(te)}, "
              f"triton_prev_in_train={prevs[-1]:.3f}")
    return p_drug, p_kraken, float(np.mean(prevs))


def run_holdout(X_df, y, arrival
                ) -> tuple[np.ndarray, np.ndarray, float,
                            np.ndarray, np.ndarray]:
    dates = pd.to_datetime(arrival)
    last_day = dates.dt.date.max()
    is_test = (dates.dt.date == last_day).to_numpy()
    is_train = ~is_test
    pre = make_preprocessor(X_df)
    pre.fit(X_df.iloc[is_train])
    X_tr = np.asarray(pre.transform(X_df.iloc[is_train]), dtype=float)
    X_te = np.asarray(pre.transform(X_df.iloc[is_test]), dtype=float)
    y_tr = y[is_train]
    y_te = y[is_test]
    p_drug = _fit_tier1(X_tr, y_tr, X_te)
    p_kraken = _fit_kraken_vs_rest(X_tr, y_tr, X_te)
    prev = _triton_prev(y_tr)
    print(f"  holdout: train n={len(y_tr)}, test n={len(y_te)}, "
          f"triton_prev_in_train={prev:.3f} (test = {last_day})")
    return p_drug, p_kraken, prev, y_te, is_test


# ---------- Cascade-B label + soft probability assembly -----------------

def soft_probs_4class(p_drug: np.ndarray, p_kraken: np.ndarray,
                       triton_prev: float) -> np.ndarray:
    """Chain-product soft probabilities (always sums to 1 by construction)."""
    n = len(p_drug)
    p = np.zeros((n, 4))
    p[:, NONE_IDX] = 1.0 - p_drug
    p[:, KRAKEN_IDX] = p_drug * p_kraken
    p[:, TRITON_IDX] = p_drug * (1.0 - p_kraken) * triton_prev
    p[:, CORAL_IDX] = p_drug * (1.0 - p_kraken) * (1.0 - triton_prev)
    return p


def hard_labels(p_drug: np.ndarray, p_kraken: np.ndarray,
                triton_prev: float,
                t_drug: float, t_kraken: float,
                tc_uniforms: np.ndarray) -> np.ndarray:
    """Cascade-B hard rule. Stage 1 + 2 use thresholds; stage 3 assigns
    Triton/Coral per-encounter so the marginal output distribution
    matches the training prevalence.

    Each row's `tc_uniforms[i]` is a deterministic uniform[0,1) draw
    keyed on its encounter_id. A row that falls into the non-Kraken
    drug-positive bucket gets Triton if its draw < triton_prev, else
    Coral. The same encounter therefore always gets the same T/C
    label across grid cells and re-runs.
    """
    labels = np.full(len(p_drug), NONE_IDX, dtype=int)
    is_drug = p_drug >= t_drug
    labels[is_drug & (p_kraken >= t_kraken)] = KRAKEN_IDX
    is_non_k = is_drug & (p_kraken < t_kraken)
    labels[is_non_k & (tc_uniforms < triton_prev)] = TRITON_IDX
    labels[is_non_k & (tc_uniforms >= triton_prev)] = CORAL_IDX
    return labels


# ---------- Grid search -------------------------------------------------

def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), zero_division=0,
    )
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f)),
        "min_class_f1": float(np.min(f)),
        "f1_none": float(f[NONE_IDX]),
        "f1_kraken": float(f[KRAKEN_IDX]),
        "f1_triton": float(f[TRITON_IDX]),
        "f1_coral": float(f[CORAL_IDX]),
        "precision_none": float(p[NONE_IDX]),
        "precision_kraken": float(p[KRAKEN_IDX]),
        "recall_none": float(r[NONE_IDX]),
        "recall_kraken": float(r[KRAKEN_IDX]),
    }


def grid_search(p_drug: np.ndarray, p_kraken: np.ndarray,
                 triton_prev: float, y_true: np.ndarray,
                 tc_uniforms: np.ndarray) -> pd.DataFrame:
    tau_grid = np.arange(GRID_LO, GRID_HI + 1e-9, GRID_STEP)
    rows = []
    for td in tau_grid:
        for tk in tau_grid:
            preds = hard_labels(p_drug, p_kraken, triton_prev,
                                  td, tk, tc_uniforms)
            m = metric_pack(y_true, preds)
            rows.append({"tau_drug": round(float(td), 3),
                          "tau_kraken": round(float(tk), 3),
                          **m})
    return pd.DataFrame(rows)


@dataclass
class PickedPoint:
    criterion: str
    tau_drug: float
    tau_kraken: float
    metrics: dict


def pick_optima(grid: pd.DataFrame) -> dict[str, PickedPoint]:
    picks = {}
    for crit in ("macro_f1", "accuracy", "min_class_f1"):
        idx = int(grid[crit].idxmax())
        row = grid.iloc[idx]
        m = {c: float(row[c]) for c in grid.columns
             if c not in ("tau_drug", "tau_kraken")}
        picks[crit] = PickedPoint(
            criterion=crit,
            tau_drug=float(row["tau_drug"]),
            tau_kraken=float(row["tau_kraken"]),
            metrics=m,
        )
    return picks


# ---------- Reporting helpers -------------------------------------------

def md_picked_table(picks: dict[str, PickedPoint]) -> str:
    lines = [
        "| Criterion | τ_drug | τ_kraken | Macro F1 | Accuracy | Min-class F1 | F1(N) | F1(K) | F1(T) | F1(C) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for crit, p in picks.items():
        m = p.metrics
        lines.append(
            f"| {crit} | {p.tau_drug:.2f} | {p.tau_kraken:.2f} | "
            f"{m['macro_f1']:.3f} | {m['accuracy']:.3f} | "
            f"{m['min_class_f1']:.3f} | "
            f"{m['f1_none']:.3f} | {m['f1_kraken']:.3f} | "
            f"{m['f1_triton']:.3f} | {m['f1_coral']:.3f} |"
        )
    return "\n".join(lines)


def md_confusion(cm) -> str:
    rows = ["| true \\ pred | None | Kraken | Triton | Coral |",
            "|---|---:|---:|---:|---:|"]
    for i, c in enumerate(CLASS_NAMES):
        rows.append(f"| {c} | {cm[i][0]} | {cm[i][1]} | "
                    f"{cm[i][2]} | {cm[i][3]} |")
    return "\n".join(rows)


# ---------- Orchestration -----------------------------------------------

def main() -> None:
    print("=" * 78)
    print("Cascade-B threshold tradeoff (rforest, 5-fold OOF grid + holdout)")
    print("=" * 78)
    X_df, y, arrival, ids = load_features_and_y()
    print(f"Features: {X_df.shape}   y class counts: "
          f"{np.bincount(y, minlength=4).tolist()}")

    print("\n--- 5-fold OOF (threshold-picking data) ---")
    p_drug_oof, p_kraken_oof, prev_oof = run_oof(X_df, y)
    print(f"Mean training Triton-prev across 5 folds: {prev_oof:.3f}")

    # Deterministic per-encounter uniform draws for stage-3 Bernoulli.
    tc_uniforms_oof = stable_uniforms(ids)
    triton_share_oof = float((tc_uniforms_oof < prev_oof).mean())
    print(f"Stage-3 stable hash: {triton_share_oof:.3f} of all rows "
          f"would draw Triton (target prevalence {prev_oof:.3f})")

    print(f"\n--- Grid-search (t_drug x t_kraken)  "
          f"[{GRID_LO:.2f}..{GRID_HI:.2f}] step {GRID_STEP:.2f} "
          f"= {int(round((GRID_HI-GRID_LO)/GRID_STEP))+1}^2 cells ---")
    grid = grid_search(p_drug_oof, p_kraken_oof, prev_oof, y,
                        tc_uniforms_oof)
    grid_path = DERIVED / "task1_cascade_b_threshold_grid.csv"
    grid.to_csv(grid_path, index=False)
    print(f"  grid cells: {len(grid)}    written: {grid_path}")

    picks = pick_optima(grid)
    print("\nPicked points (OOF):")
    for crit, p in picks.items():
        m = p.metrics
        print(f"  {crit:14s}  t_drug={p.tau_drug:.2f}  "
              f"t_kraken={p.tau_kraken:.2f}  "
              f"macroF1={m['macro_f1']:.3f}  acc={m['accuracy']:.3f}  "
              f"minF1={m['min_class_f1']:.3f}")

    pick_rows = []
    for crit, p in picks.items():
        pick_rows.append({"split": "oof", "criterion": crit,
                          "tau_drug": p.tau_drug,
                          "tau_kraken": p.tau_kraken,
                          "triton_prev": prev_oof,
                          **p.metrics})

    print("\n--- Temporal holdout — apply picked thresholds ---")
    p_drug_h, p_kraken_h, prev_h, y_h, is_test = run_holdout(
        X_df, y, arrival,
    )
    ids_h = ids[is_test]
    tc_uniforms_h = stable_uniforms(ids_h)

    holdout_metrics = {}
    for crit, p in picks.items():
        preds = hard_labels(p_drug_h, p_kraken_h, prev_h,
                             p.tau_drug, p.tau_kraken,
                             tc_uniforms_h)
        m = metric_pack(y_h, preds)
        cm = confusion_matrix(y_h, preds, labels=list(range(4))).tolist()
        holdout_metrics[crit] = (m, cm, preds)
        pick_rows.append({"split": "holdout", "criterion": crit,
                          "tau_drug": p.tau_drug,
                          "tau_kraken": p.tau_kraken,
                          "triton_prev": prev_h,
                          **m})

    pd.DataFrame(pick_rows).to_csv(
        DERIVED / "task1_cascade_b_threshold_picked.csv", index=False,
    )

    # --- Per-encounter label CSV: retain ALL probabilities + final label
    # Use the macro_f1-optimal pair for the "final outcome candidate".
    best = picks["macro_f1"]
    label_rows = []

    soft_oof = soft_probs_4class(p_drug_oof, p_kraken_oof, prev_oof)
    final_oof = hard_labels(p_drug_oof, p_kraken_oof, prev_oof,
                              best.tau_drug, best.tau_kraken,
                              tc_uniforms_oof)
    for i, eid in enumerate(ids):
        label_rows.append({
            "split": "oof",
            "encounter_id": eid,
            "p_drug": float(p_drug_oof[i]),
            "p_kraken_given_drug": float(p_kraken_oof[i]),
            "triton_prev": prev_oof,
            "tc_uniform_draw": float(tc_uniforms_oof[i]),
            "p_none": float(soft_oof[i, NONE_IDX]),
            "p_kraken": float(soft_oof[i, KRAKEN_IDX]),
            "p_triton": float(soft_oof[i, TRITON_IDX]),
            "p_coral": float(soft_oof[i, CORAL_IDX]),
            "true_label": CLASS_NAMES[y[i]],
            "final_label": CLASS_NAMES[final_oof[i]],
            "tau_drug": best.tau_drug,
            "tau_kraken": best.tau_kraken,
            "picked_by": "macro_f1",
        })

    soft_h = soft_probs_4class(p_drug_h, p_kraken_h, prev_h)
    final_h = hard_labels(p_drug_h, p_kraken_h, prev_h,
                            best.tau_drug, best.tau_kraken,
                            tc_uniforms_h)
    for j, eid in enumerate(ids_h):
        label_rows.append({
            "split": "holdout",
            "encounter_id": eid,
            "p_drug": float(p_drug_h[j]),
            "p_kraken_given_drug": float(p_kraken_h[j]),
            "triton_prev": prev_h,
            "tc_uniform_draw": float(tc_uniforms_h[j]),
            "p_none": float(soft_h[j, NONE_IDX]),
            "p_kraken": float(soft_h[j, KRAKEN_IDX]),
            "p_triton": float(soft_h[j, TRITON_IDX]),
            "p_coral": float(soft_h[j, CORAL_IDX]),
            "true_label": CLASS_NAMES[y_h[j]],
            "final_label": CLASS_NAMES[final_h[j]],
            "tau_drug": best.tau_drug,
            "tau_kraken": best.tau_kraken,
            "picked_by": "macro_f1",
        })

    labels_path = DERIVED / "task1_cascade_b_threshold_labels.csv"
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)

    # Minimal one-row-per-encounter export:
    #   encounter_id, drug_class  ∈ {0=None, 1=Kraken, 2=Triton, 3=Coral}
    # Drug class is the OOF (CV-honest) cascade prediction using the
    # macro-F1-picked thresholds and the prevalence-Bernoulli T/C split.
    drug_predictions = pd.DataFrame({
        "encounter_id": ids,
        "drug_class": final_oof.astype(int),
    })
    drug_pred_path = DERIVED / "task1_drug_predictions.csv"
    drug_predictions.to_csv(drug_pred_path, index=False)

    # --- Markdown report ----------------------------------------------------
    md = [
        "# Cascade-B threshold-tradeoff results",
        "",
        f"Single model family: **rforest** (best Cascade-B model per "
        f"RUNBOOK §7h, holdout macro AUC 0.721). Thresholds picked on "
        f"5-fold OOF (n={len(y)}), applied unchanged to the temporal "
        f"holdout (test = last day, n={int(is_test.sum())}).",
        "",
        f"Triton prevalence (training mean across 5 OOF folds): "
        f"**{prev_oof:.3f}** → stage-3 assigns Triton with that probability "
        f"and Coral with **{1-prev_oof:.3f}**, per encounter.",
        "",
        f"Grid: τ_drug × τ_kraken ∈ "
        f"[{GRID_LO:.2f}, {GRID_HI:.2f}]² step {GRID_STEP:.2f} "
        f"= {len(grid)} cells.",
        "",
        "## Picked points — OOF (where thresholds were chosen)",
        "",
        md_picked_table(picks),
        "",
        "## Picked points — temporal holdout (deployment metric)",
        "",
    ]
    holdout_picks_for_table = {}
    for crit, (m, _cm, _preds) in holdout_metrics.items():
        holdout_picks_for_table[crit] = PickedPoint(
            criterion=crit,
            tau_drug=picks[crit].tau_drug,
            tau_kraken=picks[crit].tau_kraken,
            metrics=m,
        )
    md.append(md_picked_table(holdout_picks_for_table))
    md.append("")
    md.append("### Confusion matrices — holdout")
    md.append("")
    for crit, (m, cm, _preds) in holdout_metrics.items():
        md.append(
            f"**{crit}** (τ_drug={picks[crit].tau_drug:.2f}, "
            f"τ_kraken={picks[crit].tau_kraken:.2f}) — "
            f"accuracy {m['accuracy']:.3f}, macro F1 {m['macro_f1']:.3f}"
        )
        md.append("")
        md.append(md_confusion(cm))
        md.append("")

    md += [
        "## Files",
        "",
        f"- `derived/task1_cascade_b_threshold_grid.csv` — every "
        f"(τ_drug, τ_kraken) cell with metrics",
        f"- `derived/task1_cascade_b_threshold_picked.csv` — three "
        f"picked pairs evaluated on OOF and holdout",
        f"- `derived/task1_cascade_b_threshold_labels.csv` — per-encounter "
        f"output: p_drug, p_kraken_given_drug, triton_prev, the four "
        f"soft probabilities (p_none/p_kraken/p_triton/p_coral), true "
        f"label, and final candidate label (picked by macro F1).",
        "",
        "### Stage-3 behavior",
        "",
        f"Cascade-B has no T-vs-C model — the §7g ceiling. Instead of "
        f"collapsing every non-Kraken drug-positive encounter to a "
        f"single majority class, stage 3 now does a per-encounter "
        f"Bernoulli draw against the training prevalence "
        f"({prev_oof:.3f} Triton among non-K drug+). The draw is "
        f"deterministic (md5 hash of encounter_id → uniform → "
        f"compare to prevalence), so:",
        "",
        f"- the marginal Triton/Coral output distribution matches "
        f"the training prevalence (~{prev_oof:.1%} Triton, "
        f"~{1-prev_oof:.1%} Coral among non-K drug+ predictions);",
        "- the same encounter always gets the same T/C label across "
        "re-runs and across grid cells;",
        "- no RNG state is needed during the grid search.",
        "",
        "T vs C discrimination still cannot exceed chance at triage. "
        "The prevalence-Bernoulli simply preserves the marginal class "
        "balance instead of zeroing one class. For real T-vs-C "
        "discrimination, move to the 4-hour-horizon Task-2 features.",
        "",
    ]
    report_path = DERIVED / "task1_cascade_b_threshold_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\nWrote: {grid_path}")
    print(f"Wrote: {DERIVED / 'task1_cascade_b_threshold_picked.csv'}")
    print(f"Wrote: {labels_path}")
    print(f"Wrote: {drug_pred_path}")
    print(f"Wrote: {report_path}")

    print("\n--- Holdout summary (with OOF-picked thresholds applied) ---")
    for crit, (m, _cm, _preds) in holdout_metrics.items():
        p = picks[crit]
        print(f"  {crit:14s}  t_drug={p.tau_drug:.2f} "
              f"t_kraken={p.tau_kraken:.2f}  "
              f"acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}  "
              f"minF1={m['min_class_f1']:.3f}")


if __name__ == "__main__":
    main()
