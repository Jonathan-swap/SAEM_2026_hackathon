"""Task 1 tier-2 baseline — which drug? (Kraken / Triton / Coral).

The natural pair to ``train_binary.py``: given that a patient has
been identified as drug-positive (tier 1), classify which of the
three festival drugs they're on.

Cohort filter: ``ground_truth_drug != 0`` (drug-positive only,
n=157 of 261). Mirrors the organisers'
``Task1_Two_Tier_Input_Data.csv`` two-tier structure.

Predicts a 3-class outcome:
  - Kraken Candy (sympathomimetic)
  - Triton Tabs  (CNS depressant + cardiac awareness)
  - Coral Dust   (hallucinogen)

Runs BOTH 5-fold stratified cross-validation and temporal holdout
(train on days < last, test on the last day).

Source of truth: ``derived/outcomes.csv``.

Outputs:
  derived/task1_tier2_baseline_summary.csv      (CV)
  derived/task1_tier2_oof_predictions.csv       (CV OOF)
  derived/task1_tier2_temporal_summary.csv      (holdout)
  derived/task1_tier2_temporal_predictions.csv  (holdout preds)
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

# Tier-2 classes — int codes mapped to local 0/1/2 for sklearn.
# Source `ground_truth_drug`: 0=None, 1=Kraken, 2=Triton, 3=Coral.
CLASSES = ["Kraken Candy", "Triton Tabs", "Coral Dust"]
SOURCE_TO_LOCAL = {1: 0, 2: 1, 3: 2}
LOCAL_TO_NAME = {0: "Kraken Candy", 1: "Triton Tabs", 2: "Coral Dust"}
TEXT_COL = "triage_brief_note"


# ---------- Data loading ----------------------------------------------

def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.Series, np.ndarray]:
    """Return (X_df, y_local, arrival_date, encounter_ids).

    y_local is the local-indexed target (0=Kraken, 1=Triton, 2=Coral).
    Cohort already filtered to drug-positive (n=157).
    """
    X = pd.read_csv(DERIVED / "features_triage.csv")
    outcomes = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "ground_truth_drug"]]
    # Defensive: features files must not carry outcome columns
    for c in ("encounter_disposition_label", "ground_truth_drug",
               "ground_truth_drug_name"):
        if c in X.columns:
            X = X.drop(columns=[c])
    df = X.merge(outcomes, on="encounter_id", how="inner")

    # Cohort filter: drug-positive only
    n_before = len(df)
    df = df[df["ground_truth_drug"] != 0].reset_index(drop=True)
    print(f"Cohort filter (drug-positive): {n_before} -> {len(df)} patients")

    drop = ["encounter_id", "encounter_arrival_date", "ground_truth_drug"]
    drop = [c for c in drop if c in df.columns]
    y_local = df["ground_truth_drug"].astype(int).map(SOURCE_TO_LOCAL).to_numpy()
    arrival = df.get("encounter_arrival_date",
                      pd.Series([None] * len(df)))
    ids = df["encounter_id"].to_numpy()
    X_df = df.drop(columns=drop)
    assert "ground_truth_drug" not in X_df.columns
    return X_df, y_local, arrival, ids


# ---------- Preprocessor + models -------------------------------------

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


# ---------- Metric pack -----------------------------------------------

def fit_and_score(model_name, X_tr, X_te, y_tr, y_te) -> dict:
    mdl = model_zoo()[model_name]
    mdl.fit(X_tr, y_tr)
    p_full = np.zeros((len(X_te), len(CLASSES)))
    p_seen = mdl.predict_proba(X_te)
    for j, cls in enumerate(mdl.classes_):
        p_full[:, cls] = p_seen[:, j]
    pred = p_full.argmax(axis=1)
    classes_idx = list(range(len(CLASSES)))

    try:
        auc_macro = roc_auc_score(y_te, p_full, multi_class="ovr",
                                    average="macro", labels=classes_idx)
    except ValueError:
        auc_macro = float("nan")
    try:
        ll = log_loss(y_te, p_full, labels=classes_idx)
    except ValueError:
        ll = float("nan")
    acc = float((pred == y_te).mean())

    per_auc, per_prauc, per_brier, per_bss, prev = {}, {}, {}, {}, {}
    for k, c in enumerate(CLASSES):
        y_bin = (y_te == k).astype(int)
        prev[c] = float(y_bin.mean())
        try:
            per_auc[c] = float(roc_auc_score(y_bin, p_full[:, k]))
        except ValueError:
            per_auc[c] = float("nan")
        try:
            per_prauc[c] = float(average_precision_score(y_bin, p_full[:, k]))
        except ValueError:
            per_prauc[c] = float("nan")
        per_brier[c] = float(brier_score_loss(y_bin, p_full[:, k]))
        denom = prev[c] * (1 - prev[c])
        per_bss[c] = (1.0 - per_brier[c] / denom) if denom > 0 else float("nan")
    macro_prauc = float(np.nanmean(list(per_prauc.values())))

    return {
        "logloss": ll, "accuracy": acc,
        "macro_auc": auc_macro, "macro_prauc": macro_prauc,
        "prevalence": prev, "per_auc": per_auc, "per_prauc": per_prauc,
        "per_brier": per_brier, "per_bss": per_bss,
        "p_full": p_full, "pred": pred,
    }


def print_metric_block(name: str, r: dict) -> None:
    print(f"\n  --- {name} ---")
    print(f"    log-loss        {r['logloss']:.4f}")
    print(f"    accuracy        {r['accuracy']:.4f}")
    print(f"    macro ROC-AUC   {r['macro_auc']:.4f}")
    print(f"    macro PR-AUC    {r['macro_prauc']:.4f}")
    print(f"    prevalence      "
          f"K={r['prevalence']['Kraken Candy']:.2f}  "
          f"T={r['prevalence']['Triton Tabs']:.2f}  "
          f"C={r['prevalence']['Coral Dust']:.2f}")
    print(f"    OVR ROC-AUC     "
          f"K={r['per_auc']['Kraken Candy']:.3f}  "
          f"T={r['per_auc']['Triton Tabs']:.3f}  "
          f"C={r['per_auc']['Coral Dust']:.3f}")
    print(f"    OVR PR-AUC      "
          f"K={r['per_prauc']['Kraken Candy']:.3f}  "
          f"T={r['per_prauc']['Triton Tabs']:.3f}  "
          f"C={r['per_prauc']['Coral Dust']:.3f}")
    print(f"    Brier           "
          f"K={r['per_brier']['Kraken Candy']:.3f}  "
          f"T={r['per_brier']['Triton Tabs']:.3f}  "
          f"C={r['per_brier']['Coral Dust']:.3f}")
    print(f"    Brier Skill     "
          f"K={r['per_bss']['Kraken Candy']:+.3f}  "
          f"T={r['per_bss']['Triton Tabs']:+.3f}  "
          f"C={r['per_bss']['Coral Dust']:+.3f}")


# ---------- 5-fold CV --------------------------------------------------

def run_cv() -> None:
    print("=" * 78)
    print("Task 1 tier-2 — 5-fold stratified CV (Kraken / Triton / Coral)")
    print("=" * 78)
    X_df, y, _, ids = load_data()
    print(f"X: {X_df.shape}   y: {y.shape}")
    print(f"Class distribution:")
    for k, c in enumerate(CLASSES):
        n = int((y == k).sum())
        print(f"  {c:14s}  {n:>3d}  ({n / len(y) * 100:.1f}%)")
    maj = int(np.bincount(y).argmax())
    print(f"Majority-class baseline accuracy: {(y == maj).mean():.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    oof_proba = {name: np.zeros((len(y), len(CLASSES)))
                 for name in model_zoo()}

    for name in model_zoo():
        fold_metrics: list[dict] = []
        for fold, (tr, te) in enumerate(skf.split(X_df, y)):
            pre = make_preprocessor(X_df)
            pre.fit(X_df.iloc[tr])
            X_tr = np.asarray(pre.transform(X_df.iloc[tr]), dtype=float)
            X_te = np.asarray(pre.transform(X_df.iloc[te]), dtype=float)
            r = fit_and_score(name, X_tr, X_te, y[tr], y[te])
            oof_proba[name][te] = r["p_full"]
            fold_metrics.append(r)

        # Aggregate
        agg = {
            "logloss_mean": float(np.mean([m["logloss"] for m in fold_metrics])),
            "logloss_std":  float(np.std([m["logloss"] for m in fold_metrics])),
            "acc_mean":     float(np.mean([m["accuracy"] for m in fold_metrics])),
            "acc_std":      float(np.std([m["accuracy"] for m in fold_metrics])),
            "auc_macro_mean":   float(np.nanmean([m["macro_auc"] for m in fold_metrics])),
            "auc_macro_std":    float(np.nanstd([m["macro_auc"] for m in fold_metrics])),
            "prauc_macro_mean": float(np.nanmean([m["macro_prauc"] for m in fold_metrics])),
            "prauc_macro_std":  float(np.nanstd([m["macro_prauc"] for m in fold_metrics])),
        }
        for c in CLASSES:
            short = c.split()[0].lower()
            agg[f"auc_{short}"]   = float(np.nanmean([m["per_auc"][c] for m in fold_metrics]))
            agg[f"prauc_{short}"] = float(np.nanmean([m["per_prauc"][c] for m in fold_metrics]))
            agg[f"brier_{short}"] = float(np.mean([m["per_brier"][c] for m in fold_metrics]))
            agg[f"bss_{short}"]   = float(np.nanmean([m["per_bss"][c] for m in fold_metrics]))
            agg[f"prevalence_{short}"] = float(np.mean([m["prevalence"][c] for m in fold_metrics]))

        print(f"\n--- {name} (5-fold means) ---")
        print(f"  log-loss        {agg['logloss_mean']:.4f}  (+/- {agg['logloss_std']:.4f})")
        print(f"  accuracy        {agg['acc_mean']:.4f}  (+/- {agg['acc_std']:.4f})")
        print(f"  macro ROC-AUC   {agg['auc_macro_mean']:.4f}  (+/- {agg['auc_macro_std']:.4f})")
        print(f"  macro PR-AUC    {agg['prauc_macro_mean']:.4f}  (+/- {agg['prauc_macro_std']:.4f})")

        rows.append({"model": name, **agg})

    summary = pd.DataFrame(rows).set_index("model")
    best = summary["auc_macro_mean"].idxmax()
    print(f"\nBest model by mean macro ROC-AUC: {best}")

    # Confusion matrix + classification report on OOF predictions
    best_pred = oof_proba[best].argmax(axis=1)
    cm = confusion_matrix(y, best_pred, labels=list(range(len(CLASSES))))
    print(f"\nConfusion matrix ({best}, OOF):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    print(f"\nClassification report ({best}, OOF):")
    print(classification_report(y, best_pred, target_names=CLASSES,
                                  digits=3, zero_division=0))

    # Save
    summary.to_csv(DERIVED / "task1_tier2_baseline_summary.csv")
    oof = pd.DataFrame({
        "encounter_id": ids,
        "true_label":   [CLASSES[int(v)] for v in y],
    })
    for name in model_zoo():
        for k, c in enumerate(CLASSES):
            oof[f"p_{c.split()[0].lower()}_{name}"] = oof_proba[name][:, k]
    oof[f"pred_label_{best}"] = [CLASSES[int(v)] for v in oof_proba[best].argmax(axis=1)]
    oof.to_csv(DERIVED / "task1_tier2_oof_predictions.csv", index=False)
    print(f"\nSaved: derived/task1_tier2_baseline_summary.csv")
    print(f"       derived/task1_tier2_oof_predictions.csv")


# ---------- Temporal holdout ------------------------------------------

def run_temporal() -> None:
    print("\n" + "=" * 78)
    print("Task 1 tier-2 — temporal holdout "
          "(train days < last, test = last day)")
    print("=" * 78)

    X_df, y, arrival, ids = load_data()
    dates = pd.to_datetime(arrival)
    last_day = dates.dt.date.max()
    is_test = (dates.dt.date == last_day).to_numpy()
    is_train = ~is_test
    print(f"Train: {int(is_train.sum())} drug-positive encounters "
          f"(dates < {last_day})")
    print(f"Test:  {int(is_test.sum())} drug-positive encounters "
          f"(date = {last_day})")
    print(f"Train class counts:")
    for k, c in enumerate(CLASSES):
        n = int((y[is_train] == k).sum())
        print(f"  {c:14s}  {n}")
    print(f"Test class counts:")
    for k, c in enumerate(CLASSES):
        n = int((y[is_test] == k).sum())
        print(f"  {c:14s}  {n}")

    pre = make_preprocessor(X_df)
    pre.fit(X_df.iloc[is_train])
    X_tr = np.asarray(pre.transform(X_df.iloc[is_train]), dtype=float)
    X_te = np.asarray(pre.transform(X_df.iloc[is_test]), dtype=float)

    rows = []
    pred_rows = []
    test_ids = ids[is_test]
    y_te = y[is_test]

    for name in model_zoo():
        r = fit_and_score(name, X_tr, X_te, y[is_train], y_te)
        print_metric_block(name, r)
        row = {"model": name,
                "logloss": r["logloss"],
                "accuracy": r["accuracy"],
                "macro_auc": r["macro_auc"],
                "macro_prauc": r["macro_prauc"]}
        for c in CLASSES:
            short = c.split()[0].lower()
            row[f"prevalence_{short}"] = r["prevalence"][c]
            row[f"auc_{short}"] = r["per_auc"][c]
            row[f"prauc_{short}"] = r["per_prauc"][c]
            row[f"brier_{short}"] = r["per_brier"][c]
            row[f"bss_{short}"] = r["per_bss"][c]
        rows.append(row)

        for i, eid in enumerate(test_ids):
            pred_rows.append({
                "model": name, "encounter_id": eid,
                "true_label": CLASSES[int(y_te[i])],
                "pred_label": CLASSES[int(r["p_full"][i].argmax())],
                **{f"p_{c.split()[0].lower()}": float(r["p_full"][i, k])
                   for k, c in enumerate(CLASSES)},
            })

    summary = pd.DataFrame(rows).set_index("model")
    best = summary["macro_auc"].idxmax()
    print(f"\nBest model by macro ROC-AUC: {best}")

    best_preds = [r for r in pred_rows if r["model"] == best]
    y_pred_best = [CLASSES.index(r["pred_label"]) for r in best_preds]
    cm = confusion_matrix(y_te, y_pred_best, labels=list(range(len(CLASSES))))
    print(f"\nConfusion matrix ({best}, test = last day):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    summary.to_csv(DERIVED / "task1_tier2_temporal_summary.csv")
    pd.DataFrame(pred_rows).to_csv(
        DERIVED / "task1_tier2_temporal_predictions.csv", index=False)
    print(f"\nSaved: derived/task1_tier2_temporal_summary.csv")
    print(f"       derived/task1_tier2_temporal_predictions.csv")


def main() -> None:
    run_cv()
    run_temporal()


if __name__ == "__main__":
    main()
