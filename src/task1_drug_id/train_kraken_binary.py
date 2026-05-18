"""Task 1 — Kraken vs (Triton + Coral) binary classifier.

Sub-problem of tier 2: given a patient is drug-positive, is it
**Kraken** (sympathomimetic, severity-enriched) or one of the
non-Kraken drugs (Triton depressant / Coral hallucinogen)?

Clinically the most consequential split — Kraken is enriched among
high-severity cases (75% of high-severity = Kraken per v6) and has
the only triage-visible toxidrome (diaphoretic+tachycardic+tremor,
elevated AG, hyperthermia). A Kraken-vs-rest decision drives a very
different ED workup (rhabdomyolysis labs, aggressive fluids,
cooling) than Triton/Coral (supportive care).

Cohort: ``ground_truth_drug != 0`` (drug-positive only, n=157).
Class encoding:
  1 = Kraken Candy           (positive, n≈58, prevalence ~37%)
  0 = Other drug (T or C)    (negative, n≈99)

Runs BOTH 5-fold stratified CV and temporal holdout, same model
lineup as the other Task-1 scripts (logreg, rforest, hgb).

Outputs:
  derived/task1_kraken_binary_baseline_summary.csv       (CV)
  derived/task1_kraken_binary_oof_predictions.csv        (CV OOF)
  derived/task1_kraken_binary_temporal_summary.csv       (holdout)
  derived/task1_kraken_binary_temporal_predictions.csv   (holdout preds)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              classification_report, confusion_matrix,
                              log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

# Source ground_truth_drug: 0=None, 1=Kraken, 2=Triton, 3=Coral.
# This script's positive = Kraken (1); negative = Triton (2) OR Coral (3).
CLASSES = ["Other drug (Triton/Coral)", "Kraken Candy"]
TEXT_COL = "triage_brief_note"


# ---------- Data loading ----------------------------------------------

def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.Series, np.ndarray]:
    X = pd.read_csv(DERIVED / "features_triage.csv")
    outcomes = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "ground_truth_drug"]]
    for c in ("encounter_disposition_label", "ground_truth_drug",
               "ground_truth_drug_name"):
        if c in X.columns:
            X = X.drop(columns=[c])
    df = X.merge(outcomes, on="encounter_id", how="inner")
    # Cohort: drug-positive only
    n_before = len(df)
    df = df[df["ground_truth_drug"] != 0].reset_index(drop=True)
    print(f"Cohort filter (drug-positive): {n_before} -> {len(df)} patients")

    drop = ["encounter_id", "encounter_arrival_date", "ground_truth_drug"]
    drop = [c for c in drop if c in df.columns]
    y = (df["ground_truth_drug"] == 1).astype(int).to_numpy()
    arrival = df.get("encounter_arrival_date",
                      pd.Series([None] * len(df)))
    ids = df["encounter_id"].to_numpy()
    X_df = df.drop(columns=drop)
    assert "ground_truth_drug" not in X_df.columns
    return X_df, y, arrival, ids


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


def model_zoo() -> dict:
    return {
        "logreg":  LogisticRegression(max_iter=3000, C=0.5,
                                       class_weight="balanced",
                                       solver="lbfgs"),
        "rforest": RandomForestClassifier(n_estimators=400, max_depth=8,
                                           min_samples_leaf=4,
                                           class_weight="balanced",
                                           random_state=42, n_jobs=-1),
        "hgb":     HistGradientBoostingClassifier(max_iter=300,
                                                   max_depth=6,
                                                   learning_rate=0.05,
                                                   l2_regularization=0.5,
                                                   class_weight="balanced",
                                                   random_state=42),
    }


# ---------- Binary metric helper -------------------------------------

def binary_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> dict:
    pred = (p_pos >= 0.5).astype(int)
    prev = float(y_true.mean())
    brier = float(brier_score_loss(y_true, p_pos))
    denom = prev * (1.0 - prev)
    bss = (1.0 - brier / denom) if denom > 0 else float("nan")
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    try:
        auc = float(roc_auc_score(y_true, p_pos))
    except ValueError:
        auc = float("nan")
    try:
        prauc = float(average_precision_score(y_true, p_pos))
    except ValueError:
        prauc = float("nan")
    return {
        "prevalence": prev,
        "accuracy": float((pred == y_true).mean()),
        "roc_auc": auc,
        "pr_auc": prauc,
        "log_loss": float(log_loss(y_true,
                                     np.clip(p_pos, 1e-12, 1 - 1e-12),
                                     labels=[0, 1])),
        "brier": brier,
        "bss": bss,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def print_metrics(name: str, m: dict) -> None:
    print(f"  {name}:")
    print(f"    log-loss        {m['log_loss']:.4f}")
    print(f"    accuracy        {m['accuracy']:.4f}")
    print(f"    ROC-AUC         {m['roc_auc']:.4f}")
    print(f"    PR-AUC          {m['pr_auc']:.4f}")
    print(f"    prevalence      {m['prevalence']:.3f}")
    print(f"    Brier           {m['brier']:.4f}")
    print(f"    Brier Skill     {m['bss']:+.4f}")
    print(f"    Sensitivity     {m['sensitivity']:.4f}")
    print(f"    Specificity     {m['specificity']:.4f}")
    print(f"    PPV             {m['ppv']:.4f}")
    print(f"    NPV             {m['npv']:.4f}")
    print(f"    Confusion: TP={m['tp']}  TN={m['tn']}  "
          f"FP={m['fp']}  FN={m['fn']}")


# ---------- 5-fold CV --------------------------------------------------

def run_cv() -> None:
    print("=" * 78)
    print("Task 1 — Kraken vs (Triton + Coral), 5-fold stratified CV")
    print("=" * 78)
    X_df, y, _, ids = load_data()
    print(f"X: {X_df.shape}   y: {y.shape}")
    print(f"Class distribution:  Other drug = {int((y == 0).sum())}  "
          f"Kraken = {int((y == 1).sum())}")
    print(f"Prevalence (Kraken): {y.mean():.3f}")
    maj = int(y.mean() >= 0.5)
    print(f"Majority-class baseline accuracy: {(y == maj).mean():.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    oof_proba = {name: np.zeros(len(y)) for name in model_zoo()}

    for name in model_zoo():
        fold_metrics: list[dict] = []
        for fold, (tr, te) in enumerate(skf.split(X_df, y)):
            pre = make_preprocessor(X_df)
            pre.fit(X_df.iloc[tr])
            X_tr = np.asarray(pre.transform(X_df.iloc[tr]), dtype=float)
            X_te = np.asarray(pre.transform(X_df.iloc[te]), dtype=float)
            mdl = model_zoo()[name]
            mdl.fit(X_tr, y[tr])
            p = mdl.predict_proba(X_te)[:, 1]
            oof_proba[name][te] = p
            fold_metrics.append(binary_metrics(y[te], p))

        keys = ("log_loss", "accuracy", "roc_auc", "pr_auc",
                 "brier", "bss", "sensitivity", "specificity",
                 "ppv", "npv")
        m_mean = {k: float(np.mean([m[k] for m in fold_metrics]))
                  for k in keys}
        m_std = {k: float(np.std([m[k] for m in fold_metrics]))
                 for k in keys}
        print(f"\n--- {name} (5-fold OOF) ---")
        for k in keys:
            print(f"  {k:14s}  {m_mean[k]:.4f}  (+/- {m_std[k]:.4f})")
        row = {"model": name, "prevalence": float(y.mean()),
                **{f"{k}_mean": m_mean[k] for k in keys},
                **{f"{k}_std": m_std[k] for k in keys}}
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("model")
    best = summary["roc_auc_mean"].idxmax()
    print(f"\nBest model by mean ROC-AUC: {best}")

    best_p = oof_proba[best]
    best_pred = (best_p >= 0.5).astype(int)
    cm = confusion_matrix(y, best_pred, labels=[0, 1])
    print(f"\nConfusion matrix ({best}, OOF):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    print(f"\nClassification report ({best}, OOF):")
    print(classification_report(y, best_pred, target_names=CLASSES,
                                  digits=3, zero_division=0))

    summary.to_csv(DERIVED / "task1_kraken_binary_baseline_summary.csv")
    oof = pd.DataFrame({
        "encounter_id": ids,
        "true_label": [CLASSES[int(v)] for v in y],
    })
    for name in model_zoo():
        oof[f"p_kraken_{name}"] = oof_proba[name]
    oof[f"pred_label_{best}"] = [CLASSES[int(v)] for v in
                                   (oof_proba[best] >= 0.5).astype(int)]
    oof.to_csv(DERIVED / "task1_kraken_binary_oof_predictions.csv",
                index=False)
    print(f"\nSaved: derived/task1_kraken_binary_baseline_summary.csv")
    print(f"       derived/task1_kraken_binary_oof_predictions.csv")


# ---------- Temporal holdout ------------------------------------------

def run_temporal() -> None:
    print("\n" + "=" * 78)
    print("Task 1 — Kraken vs rest, temporal holdout")
    print("=" * 78)
    X_df, y, arrival, ids = load_data()
    dates = pd.to_datetime(arrival)
    last_day = dates.dt.date.max()
    is_test = (dates.dt.date == last_day).to_numpy()
    is_train = ~is_test
    print(f"Train: {int(is_train.sum())} (dates < {last_day})")
    print(f"Test:  {int(is_test.sum())} (date = {last_day})")
    print(f"Train Kraken prevalence: {y[is_train].mean():.3f}")
    print(f"Test  Kraken prevalence: {y[is_test].mean():.3f}")

    pre = make_preprocessor(X_df)
    pre.fit(X_df.iloc[is_train])
    X_tr = np.asarray(pre.transform(X_df.iloc[is_train]), dtype=float)
    X_te = np.asarray(pre.transform(X_df.iloc[is_test]), dtype=float)
    rows = []
    pred_rows = []
    test_ids = ids[is_test]
    y_te = y[is_test]
    for name in model_zoo():
        mdl = model_zoo()[name]
        mdl.fit(X_tr, y[is_train])
        p = mdl.predict_proba(X_te)[:, 1]
        m = binary_metrics(y_te, p)
        print_metrics(name, m)
        rows.append({"model": name, **m})
        for i, eid in enumerate(test_ids):
            pred_rows.append({
                "model": name, "encounter_id": eid,
                "true_label": CLASSES[int(y_te[i])],
                "p_kraken": float(p[i]),
                "pred_label": CLASSES[int(p[i] >= 0.5)],
            })

    summary = pd.DataFrame(rows).set_index("model")
    best = summary["roc_auc"].idxmax()
    print(f"\nBest model by ROC-AUC: {best}")

    summary.to_csv(DERIVED / "task1_kraken_binary_temporal_summary.csv")
    pd.DataFrame(pred_rows).to_csv(
        DERIVED / "task1_kraken_binary_temporal_predictions.csv",
        index=False)
    print(f"\nSaved: derived/task1_kraken_binary_temporal_summary.csv")
    print(f"       derived/task1_kraken_binary_temporal_predictions.csv")


def main() -> None:
    run_cv()
    run_temporal()


if __name__ == "__main__":
    main()
