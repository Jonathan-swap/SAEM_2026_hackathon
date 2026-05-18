"""Pick a threshold for each binary stage of Cascade-C and assemble
the final 4-class outcome label per encounter.

Cascade-C composition (per RUNBOOK §7h):
  Stage 1 (tier-1):    P(drug-positive | X)
  Stage 2 (K-vs-rest): P(Kraken | drug-positive, X)
  Stage 3 (T-vs-C):    P(Triton | non-Kraken drug-positive, X)

Each stage gets a hard threshold picked under THREE criteria, reported
side-by-side:
  - Youden's J     argmax (sensitivity + specificity − 1)
  - Max F1         argmax F1 of the positive class
  - Sens ≥ 0.90    lowest threshold where sensitivity stays ≥ 0.90

Thresholds are picked on 5-fold OOF predictions (CV-honest, all 261
patients contribute). They are then applied unchanged to the temporal
holdout to produce the deployment-relevant 4-class label.

Model family: rforest — the best Cascade-C model per RUNBOOK §7h
(holdout macro AUC 0.707, ahead of hgb 0.687 and logreg 0.608).

Final label assembly (hard cascade):
  if P(drug)        <  τ_drug:    label = None
  elif P(K|drug)    >= τ_kraken:  label = Kraken Candy
  elif P(T|non-K)   >= τ_triton:  label = Triton Tabs
  else:                            label = Coral Dust

Outputs:
  derived/task1_cascade_thresholds.csv         picked thresholds + diagnostics
  derived/task1_cascade_threshold_labels.csv   per-encounter labels (every criterion × split)
  derived/task1_cascade_threshold_report.md    full metric report
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

# Label indices match outcomes.csv :: ground_truth_drug
CLASS_NAMES = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]
NONE_IDX, KRAKEN_IDX, TRITON_IDX, CORAL_IDX = 0, 1, 2, 3

TEXT_COL = "triage_brief_note"

CRITERIA = ("youden", "max_f1", "sens_at_least_90")
SENS_FLOOR = 0.90


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
    """Same hyperparameters as compare_cascades.py."""
    return RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


# ---------- Binary fits (each filters y to its cohort, predicts on all rows) ----

def _fit_tier1(X_tr, y_tr_4cls, X_te) -> np.ndarray:
    """P(drug-positive)."""
    y = (y_tr_4cls != NONE_IDX).astype(int)
    mdl = make_rforest()
    mdl.fit(X_tr, y)
    return mdl.predict_proba(X_te)[:, 1]


def _fit_kraken_vs_rest(X_tr, y_tr_4cls, X_te) -> np.ndarray:
    """P(Kraken | drug-positive). Trained on drug-positive only."""
    mask = (y_tr_4cls != NONE_IDX)
    y = (y_tr_4cls[mask] == KRAKEN_IDX).astype(int)
    mdl = make_rforest()
    mdl.fit(X_tr[mask], y)
    return mdl.predict_proba(X_te)[:, 1]


def _fit_triton_vs_coral(X_tr, y_tr_4cls, X_te) -> np.ndarray:
    """P(Triton | non-Kraken drug-positive). Trained on T/C only.

    Falls back to training-set Triton prevalence among non-K drug
    if the fold has too few non-K drug cases (<10). Same fallback as
    compare_cascades.py.
    """
    mask = (y_tr_4cls == TRITON_IDX) | (y_tr_4cls == CORAL_IDX)
    if mask.sum() < 10:
        prev = float((y_tr_4cls[mask] == TRITON_IDX).mean()) if mask.sum() else 0.5
        return np.full(len(X_te), prev)
    y = (y_tr_4cls[mask] == TRITON_IDX).astype(int)
    mdl = make_rforest()
    mdl.fit(X_tr[mask], y)
    return mdl.predict_proba(X_te)[:, 1]


# ---------- Threshold pickers -------------------------------------------

@dataclass
class ThresholdPick:
    criterion: str
    threshold: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float


def pick_youden(y_bin: np.ndarray, p: np.ndarray) -> ThresholdPick:
    fpr, tpr, thr = roc_curve(y_bin, p)
    # Discard the artificial threshold at +inf returned by roc_curve.
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    j = tpr - fpr
    idx = int(np.argmax(j))
    return _pack_pick("youden", float(thr[idx]), y_bin, p)


def pick_max_f1(y_bin: np.ndarray, p: np.ndarray) -> ThresholdPick:
    # Sweep over the unique probability values; this is exact for the
    # piecewise-constant F1 curve.
    candidates = np.unique(p)
    best_f1, best_thr = -1.0, 0.5
    for t in candidates:
        pred = (p >= t).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y_bin, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return _pack_pick("max_f1", best_thr, y_bin, p)


def pick_sens_floor(y_bin: np.ndarray, p: np.ndarray,
                    floor: float = SENS_FLOOR) -> ThresholdPick:
    """Highest threshold whose sensitivity is still ≥ floor.

    Higher τ → fewer positives flagged → lower sens. We want the
    largest τ that still preserves sens ≥ floor (so spec is as good
    as possible while honoring the screening floor).
    """
    fpr, tpr, thr = roc_curve(y_bin, p)
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    ok = tpr >= floor
    if not ok.any():
        # No threshold satisfies the floor; fall back to the lowest threshold
        # (which gives the highest possible sens for this set of probs).
        idx = int(np.argmax(tpr))
    else:
        # Among thresholds meeting the floor, pick the most stringent
        # (highest τ, which minimises false positives).
        ok_idx = np.flatnonzero(ok)
        idx = int(ok_idx[np.argmax(thr[ok_idx])])
    return _pack_pick("sens_at_least_90", float(thr[idx]), y_bin, p)


def _pack_pick(criterion: str, t: float,
               y_bin: np.ndarray, p: np.ndarray) -> ThresholdPick:
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y_bin == 1)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())
    tn = int(((pred == 0) & (y_bin == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) else float("nan")
    return ThresholdPick(criterion=criterion, threshold=t,
                          sensitivity=sens, specificity=spec,
                          precision=prec, f1=f1)


# ---------- Cascade label assembly --------------------------------------

def assemble_cascade_labels(p_drug: np.ndarray,
                             p_kraken: np.ndarray,
                             p_triton: np.ndarray,
                             t_drug: float,
                             t_kraken: float,
                             t_triton: float) -> np.ndarray:
    """Hard threshold cascade → 4-class integer label per row."""
    labels = np.full(len(p_drug), NONE_IDX, dtype=int)
    is_drug = p_drug >= t_drug
    labels[is_drug & (p_kraken >= t_kraken)] = KRAKEN_IDX
    is_non_k = is_drug & (p_kraken < t_kraken)
    labels[is_non_k & (p_triton >= t_triton)] = TRITON_IDX
    labels[is_non_k & (p_triton < t_triton)] = CORAL_IDX
    return labels


# ---------- OOF + holdout probability pipelines -------------------------

def run_oof(X_df: pd.DataFrame, y: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5-fold OOF probabilities for tier-1, K-vs-rest, T-vs-C.

    Returns three (n,) arrays aligned with the rows of X_df.
    Every row has a prediction from every binary (each binary trained
    on its cohort within the fold, but predicts on the held-out test
    fold regardless of cohort).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_drug = np.zeros(len(y))
    p_kraken = np.zeros(len(y))
    p_triton = np.zeros(len(y))
    for fold, (tr, te) in enumerate(skf.split(X_df, y), start=1):
        pre = make_preprocessor(X_df)
        pre.fit(X_df.iloc[tr])
        X_tr = np.asarray(pre.transform(X_df.iloc[tr]), dtype=float)
        X_te = np.asarray(pre.transform(X_df.iloc[te]), dtype=float)
        p_drug[te] = _fit_tier1(X_tr, y[tr], X_te)
        p_kraken[te] = _fit_kraken_vs_rest(X_tr, y[tr], X_te)
        p_triton[te] = _fit_triton_vs_coral(X_tr, y[tr], X_te)
        print(f"  fold {fold}: trained binaries on n={len(tr)}, "
              f"predicted on n={len(te)}")
    return p_drug, p_kraken, p_triton


def run_holdout(X_df: pd.DataFrame, y: np.ndarray, arrival: pd.Series
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train on all-but-last-day, predict the last day."""
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
    print(f"  holdout train n={len(y_tr)}, test n={len(y_te)} "
          f"(test = {last_day})")
    p_drug = _fit_tier1(X_tr, y_tr, X_te)
    p_kraken = _fit_kraken_vs_rest(X_tr, y_tr, X_te)
    p_triton = _fit_triton_vs_coral(X_tr, y_tr, X_te)
    return p_drug, p_kraken, p_triton, y_te, is_test


# ---------- Metrics on the assembled 4-class predictions ----------------

def metric_pack_4class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    out = {"accuracy": acc, "macro_f1": float(np.mean(f))}
    for k, c in enumerate(CLASS_NAMES):
        short = c.split()[0].lower()
        out[f"precision_{short}"] = float(p[k])
        out[f"recall_{short}"] = float(r[k])
        out[f"f1_{short}"] = float(f[k])
        out[f"support_{short}"] = int(s[k])
    out["confusion_matrix"] = cm.tolist()
    return out


# ---------- Reporting helpers -------------------------------------------

def md_thresholds_table(picks: dict[str, dict[str, ThresholdPick]]) -> str:
    lines = [
        "| Criterion | Stage | τ | Sensitivity | Specificity | Precision | F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    stage_labels = {
        "drug": "Tier-1 (drug vs no-drug)",
        "kraken": "K-vs-rest (Kraken vs Triton/Coral)",
        "triton": "T-vs-C (Triton vs Coral)",
    }
    for crit in CRITERIA:
        for stage in ("drug", "kraken", "triton"):
            pk = picks[crit][stage]
            lines.append(
                f"| {crit} | {stage_labels[stage]} | "
                f"{pk.threshold:.3f} | {pk.sensitivity:.3f} | "
                f"{pk.specificity:.3f} | {pk.precision:.3f} | "
                f"{pk.f1:.3f} |"
            )
    return "\n".join(lines)


def md_metrics_table(per_crit_metrics: dict[str, dict]) -> str:
    lines = [
        "| Criterion | Accuracy | Macro F1 | F1 None | F1 Kraken | F1 Triton | F1 Coral |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for crit in CRITERIA:
        m = per_crit_metrics[crit]
        lines.append(
            f"| {crit} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
            f"{m['f1_none']:.3f} | {m['f1_kraken']:.3f} | "
            f"{m['f1_triton']:.3f} | {m['f1_coral']:.3f} |"
        )
    return "\n".join(lines)


def md_confusion(cm: list[list[int]]) -> str:
    rows = ["| true \\ pred | None | Kraken | Triton | Coral |",
            "|---|---:|---:|---:|---:|"]
    for i, c in enumerate(CLASS_NAMES):
        rows.append(f"| {c} | {cm[i][0]} | {cm[i][1]} | "
                    f"{cm[i][2]} | {cm[i][3]} |")
    return "\n".join(rows)


# ---------- Orchestration -----------------------------------------------

def pick_all_thresholds(p_drug, p_kraken, p_triton, y
                         ) -> dict[str, dict[str, ThresholdPick]]:
    """Pick thresholds for each stage under each criterion."""
    y_drug = (y != NONE_IDX).astype(int)
    drug_mask = y != NONE_IDX
    y_kraken = (y[drug_mask] == KRAKEN_IDX).astype(int)
    non_k_mask = (y == TRITON_IDX) | (y == CORAL_IDX)
    y_triton = (y[non_k_mask] == TRITON_IDX).astype(int)

    pickers = {
        "youden": pick_youden,
        "max_f1": pick_max_f1,
        "sens_at_least_90": pick_sens_floor,
    }
    out: dict[str, dict[str, ThresholdPick]] = {}
    for crit, fn in pickers.items():
        out[crit] = {
            "drug":   fn(y_drug,   p_drug),
            "kraken": fn(y_kraken, p_kraken[drug_mask]),
            "triton": fn(y_triton, p_triton[non_k_mask]),
        }
    return out


def main() -> None:
    print("=" * 78)
    print("Cascade-C threshold picking (rforest, 5-fold OOF, then holdout)")
    print("=" * 78)
    X_df, y, arrival, ids = load_features_and_y()
    print(f"Features: {X_df.shape}   y: {y.shape}   "
          f"class counts: {np.bincount(y, minlength=4).tolist()}")

    print("\n--- 5-fold OOF (threshold-picking data) ---")
    p_drug_oof, p_kraken_oof, p_triton_oof = run_oof(X_df, y)

    print("\nPicking thresholds on OOF probabilities...")
    picks = pick_all_thresholds(p_drug_oof, p_kraken_oof, p_triton_oof, y)
    for crit in CRITERIA:
        ps = picks[crit]
        print(f"  {crit:18s}  t_drug={ps['drug'].threshold:.3f}  "
              f"t_kraken={ps['kraken'].threshold:.3f}  "
              f"t_triton={ps['triton'].threshold:.3f}")

    print("\n--- Temporal holdout (apply thresholds, get final labels) ---")
    p_drug_h, p_kraken_h, p_triton_h, y_h, is_test = run_holdout(
        X_df, y, arrival,
    )
    ids_h = ids[is_test]

    # Assemble per-encounter labels for every criterion × split
    rows = []
    oof_metrics: dict[str, dict] = {}
    hld_metrics: dict[str, dict] = {}
    for crit in CRITERIA:
        td = picks[crit]["drug"].threshold
        tk = picks[crit]["kraken"].threshold
        tt = picks[crit]["triton"].threshold
        oof_pred = assemble_cascade_labels(
            p_drug_oof, p_kraken_oof, p_triton_oof, td, tk, tt,
        )
        hld_pred = assemble_cascade_labels(
            p_drug_h, p_kraken_h, p_triton_h, td, tk, tt,
        )
        oof_metrics[crit] = metric_pack_4class(y, oof_pred)
        hld_metrics[crit] = metric_pack_4class(y_h, hld_pred)
        for i, eid in enumerate(ids):
            rows.append({
                "split": "oof",
                "criterion": crit,
                "encounter_id": eid,
                "p_drug": p_drug_oof[i],
                "p_kraken": p_kraken_oof[i],
                "p_triton": p_triton_oof[i],
                "true_label": CLASS_NAMES[y[i]],
                "pred_label": CLASS_NAMES[oof_pred[i]],
                "tau_drug": td, "tau_kraken": tk, "tau_triton": tt,
            })
        for j, eid in enumerate(ids_h):
            rows.append({
                "split": "holdout",
                "criterion": crit,
                "encounter_id": eid,
                "p_drug": p_drug_h[j],
                "p_kraken": p_kraken_h[j],
                "p_triton": p_triton_h[j],
                "true_label": CLASS_NAMES[y_h[j]],
                "pred_label": CLASS_NAMES[hld_pred[j]],
                "tau_drug": td, "tau_kraken": tk, "tau_triton": tt,
            })

    # --- Write threshold table ---
    pick_rows = []
    for crit in CRITERIA:
        for stage in ("drug", "kraken", "triton"):
            pk = picks[crit][stage]
            pick_rows.append({
                "criterion": crit,
                "stage": stage,
                "threshold": pk.threshold,
                "sensitivity": pk.sensitivity,
                "specificity": pk.specificity,
                "precision": pk.precision,
                "f1": pk.f1,
            })
    thr_path = DERIVED / "task1_cascade_thresholds.csv"
    pd.DataFrame(pick_rows).to_csv(thr_path, index=False)

    # --- Write per-encounter label CSV ---
    labels_path = DERIVED / "task1_cascade_threshold_labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)

    # --- Write markdown report ---
    md = [
        "# Cascade-C threshold-based 4-class predictions",
        "",
        "Single model family: **rforest** (best Cascade-C model per "
        "RUNBOOK §7h). Thresholds picked on 5-fold OOF (n=261), then "
        "applied unchanged to the temporal holdout (test = last day).",
        "",
        "## Picked thresholds",
        "",
        md_thresholds_table(picks),
        "",
        "## 4-class metrics (5-fold OOF, n=261)",
        "",
        md_metrics_table(oof_metrics),
        "",
        "### Confusion matrices — OOF",
        "",
    ]
    for crit in CRITERIA:
        md.append(f"**{crit}** — accuracy {oof_metrics[crit]['accuracy']:.3f}, "
                  f"macro F1 {oof_metrics[crit]['macro_f1']:.3f}")
        md.append("")
        md.append(md_confusion(oof_metrics[crit]["confusion_matrix"]))
        md.append("")

    md += [
        f"## 4-class metrics (temporal holdout, n={int(is_test.sum())})",
        "",
        md_metrics_table(hld_metrics),
        "",
        "### Confusion matrices — holdout",
        "",
    ]
    for crit in CRITERIA:
        md.append(f"**{crit}** — accuracy {hld_metrics[crit]['accuracy']:.3f}, "
                  f"macro F1 {hld_metrics[crit]['macro_f1']:.3f}")
        md.append("")
        md.append(md_confusion(hld_metrics[crit]["confusion_matrix"]))
        md.append("")

    md += [
        "## Files",
        "",
        f"- `{thr_path.relative_to(ROOT).as_posix()}` — picked thresholds + diagnostics",
        f"- `{labels_path.relative_to(ROOT).as_posix()}` — per-encounter labels (every criterion × split)",
        "",
    ]
    report_path = DERIVED / "task1_cascade_threshold_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\nWrote: {thr_path}")
    print(f"Wrote: {labels_path}")
    print(f"Wrote: {report_path}")

    print("\n--- Summary (holdout) ---")
    for crit in CRITERIA:
        m = hld_metrics[crit]
        print(f"  {crit:18s}  acc={m['accuracy']:.3f}  "
              f"macroF1={m['macro_f1']:.3f}  "
              f"F1(K)={m['f1_kraken']:.3f}  "
              f"F1(T)={m['f1_triton']:.3f}  "
              f"F1(C)={m['f1_coral']:.3f}")


if __name__ == "__main__":
    main()
