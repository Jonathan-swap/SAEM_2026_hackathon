"""Temporal holdout evaluation.

Trains on all encounters EXCEPT the last day; tests on the last
day only. Mirrors a real Phase-2 deployment scenario where the model
trained on prior days must predict on a fresh wave of arrivals.

Runs both Task-1 (drug ID at triage) and Task-2 (deterioration at 4h)
with the same model lineup (logreg / rforest / hgb) and feature
plumbing as `src/task1_drug_id/train_baseline.py` and
`src/task2_deterioration/train_baseline.py`.

Writes:
  derived/task1_temporal_summary.csv
  derived/task1_temporal_predictions.csv
  derived/task2_temporal_summary.csv
  derived/task2_temporal_predictions.csv
  derived/temporal_holdout_report.md
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
from sklearn.metrics import (brier_score_loss, classification_report,
                              confusion_matrix, log_loss,
                              roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "derived"

DRUG_CLASSES = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]
DISPO_CLASSES = ["Discharge", "Floor", "ICU"]
TEXT_COL = "triage_brief_note"


# ---------- preprocessor (shared) ------------------------------------

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


def model_zoo_task1() -> dict:
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


def model_zoo_task2() -> dict:
    return {
        "logreg":  LogisticRegression(max_iter=3000, C=0.3,
                                       class_weight="balanced"),
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


# ---------- split helpers --------------------------------------------

def temporal_split(df: pd.DataFrame, date_col: str = "encounter_arrival_date"
                    ) -> tuple[pd.Series, pd.Series, pd.Timestamp, pd.Timestamp]:
    """Return (is_train, is_test, train_max_date, test_date)."""
    dates = pd.to_datetime(df[date_col])
    last_day = dates.dt.date.max()
    is_test = (dates.dt.date == last_day)
    is_train = ~is_test
    train_max = dates[is_train].max() if is_train.any() else None
    return is_train, is_test, train_max, pd.Timestamp(last_day)


# ---------- Task 1 ----------------------------------------------------

def run_task1() -> dict:
    print("\n" + "=" * 78)
    print("Task 1 — drug ID at triage (temporal holdout)")
    print("=" * 78)

    X_all = pd.read_csv(DERIVED / "features_triage.csv")
    gt = pd.read_csv(DERIVED / "ground_truth.csv")[
        ["encounter_id", "ground_truth_drug"]]
    df = X_all.merge(gt, on="encounter_id", how="inner")

    is_train, is_test, train_max, test_date = temporal_split(df)
    print(f"Train: {is_train.sum()} encounters, dates up to {train_max.date()}")
    print(f"Test:  {is_test.sum()} encounters, date = {test_date.date()}")
    print(f"Train class distribution:")
    for c, n in df.loc[is_train, "ground_truth_drug"].value_counts().sort_index().items():
        print(f"  {DRUG_CLASSES[c]:14s}  {n:>3d}")
    print(f"Test class distribution:")
    for c, n in df.loc[is_test, "ground_truth_drug"].value_counts().sort_index().items():
        print(f"  {DRUG_CLASSES[c]:14s}  {n:>3d}")

    drop = ["encounter_id", "encounter_arrival_date", "ground_truth_drug"]
    drop = [c for c in drop if c in df.columns]
    X_df = df.drop(columns=drop)
    y = df["ground_truth_drug"].astype(int).to_numpy()
    classes = list(range(len(DRUG_CLASSES)))

    X_tr_df = X_df.loc[is_train].reset_index(drop=True)
    X_te_df = X_df.loc[is_test].reset_index(drop=True)
    y_tr = y[is_train.to_numpy()]
    y_te = y[is_test.to_numpy()]

    pre = make_preprocessor(X_tr_df)
    pre.fit(X_tr_df)
    X_tr = np.asarray(pre.transform(X_tr_df), dtype=float)
    X_te = np.asarray(pre.transform(X_te_df), dtype=float)

    rows = []
    pred_rows = []
    for name, mdl in model_zoo_task1().items():
        mdl.fit(X_tr, y_tr)
        p_full = np.zeros((len(X_te), len(DRUG_CLASSES)))
        p_seen = mdl.predict_proba(X_te)
        for j, cls in enumerate(mdl.classes_):
            p_full[:, cls] = p_seen[:, j]
        pred = p_full.argmax(axis=1)

        try:
            auc = roc_auc_score(y_te, p_full, multi_class="ovr",
                                 average="macro", labels=classes)
        except ValueError:
            auc = float("nan")
        ll = log_loss(y_te, p_full, labels=classes)
        acc = float((pred == y_te).mean())
        brier = {DRUG_CLASSES[k]:
                  float(brier_score_loss((y_te == k).astype(int),
                                          p_full[:, k]))
                  for k in classes}

        print(f"\n--- {name} ---")
        print(f"  log-loss        {ll:.4f}")
        print(f"  accuracy        {acc:.4f}")
        print(f"  macro AUC       {auc:.4f}")
        print(f"  Brier           "
              f"n={brier['None']:.3f}  k={brier['Kraken Candy']:.3f}  "
              f"t={brier['Triton Tabs']:.3f}  c={brier['Coral Dust']:.3f}")

        rows.append({
            "model": name, "logloss": ll, "accuracy": acc,
            "macro_auc": auc,
            **{f"brier_{k.lower().split()[0]}": brier[k] for k in DRUG_CLASSES},
        })
        for i, eid in enumerate(df.loc[is_test, "encounter_id"].values):
            pred_rows.append({
                "model": name, "encounter_id": eid,
                "true_label": DRUG_CLASSES[int(y_te[i])],
                "pred_label": DRUG_CLASSES[int(pred[i])],
                **{f"p_{c.lower().split()[0]}": float(p_full[i, k])
                   for k, c in enumerate(DRUG_CLASSES)},
            })

    summary = pd.DataFrame(rows).set_index("model")
    print("\nSummary:")
    print(summary.to_string())

    best = summary["macro_auc"].idxmax()
    print(f"\nBest model by macro AUC: {best}")

    # Confusion matrix for the best model
    best_preds = [r for r in pred_rows if r["model"] == best]
    y_pred_best = [DRUG_CLASSES.index(r["pred_label"]) for r in best_preds]
    cm = confusion_matrix(y_te, y_pred_best,
                            labels=list(range(len(DRUG_CLASSES))))
    print(f"Confusion matrix ({best}, test = last day):")
    print(pd.DataFrame(cm, index=DRUG_CLASSES,
                        columns=DRUG_CLASSES).to_string())

    summary.to_csv(DERIVED / "task1_temporal_summary.csv")
    pd.DataFrame(pred_rows).to_csv(
        DERIVED / "task1_temporal_predictions.csv", index=False)
    print(f"\nSaved: derived/task1_temporal_summary.csv "
          f"+ task1_temporal_predictions.csv")
    return {"summary": summary, "best": best,
            "n_train": int(is_train.sum()), "n_test": int(is_test.sum()),
            "train_max": train_max, "test_date": test_date}


# ---------- Task 2 ----------------------------------------------------

def run_task2() -> dict:
    print("\n" + "=" * 78)
    print("Task 2 — deterioration at 4h (temporal holdout)")
    print("=" * 78)

    X_all = pd.read_csv(DERIVED / "features_fourh.csv")
    gt = pd.read_csv(DERIVED / "ground_truth.csv")[
        ["encounter_id", "ground_truth_drug"]]
    df = X_all.merge(gt, on="encounter_id", how="inner")
    n_before = len(df)
    df = df[df["ground_truth_drug"] != 0].reset_index(drop=True)
    print(f"Cohort filter (drug-positive): {n_before} -> {len(df)} patients")

    is_train, is_test, train_max, test_date = temporal_split(df)
    print(f"Train: {is_train.sum()} encounters, dates up to {train_max.date()}")
    print(f"Test:  {is_test.sum()} encounters, date = {test_date.date()}")

    print(f"Train disposition distribution:")
    for c, n in df.loc[is_train, "encounter_disposition_label"].value_counts().items():
        print(f"  {c:10s}  {n:>3d}")
    print(f"Test disposition distribution:")
    for c, n in df.loc[is_test, "encounter_disposition_label"].value_counts().items():
        print(f"  {c:10s}  {n:>3d}")

    drop = ["encounter_id", "encounter_arrival_date", "ground_truth_drug",
            "encounter_disposition_label"]
    if "ground_truth_drug_name" in df.columns:
        drop.append("ground_truth_drug_name")
    X_df = df.drop(columns=[c for c in drop if c in df.columns])
    label_map = {c: i for i, c in enumerate(DISPO_CLASSES)}
    y = df["encounter_disposition_label"].map(label_map).astype(int).to_numpy()
    classes = list(range(len(DISPO_CLASSES)))

    X_tr_df = X_df.loc[is_train].reset_index(drop=True)
    X_te_df = X_df.loc[is_test].reset_index(drop=True)
    y_tr = y[is_train.to_numpy()]
    y_te = y[is_test.to_numpy()]

    pre = make_preprocessor(X_tr_df)
    pre.fit(X_tr_df)
    X_tr = np.asarray(pre.transform(X_tr_df), dtype=float)
    X_te = np.asarray(pre.transform(X_te_df), dtype=float)

    rows = []
    pred_rows = []
    for name, mdl in model_zoo_task2().items():
        mdl.fit(X_tr, y_tr)
        p_full = np.zeros((len(X_te), len(DISPO_CLASSES)))
        p_seen = mdl.predict_proba(X_te)
        for j, cls in enumerate(mdl.classes_):
            p_full[:, cls] = p_seen[:, j]
        pred = p_full.argmax(axis=1)

        try:
            auc = roc_auc_score(y_te, p_full, multi_class="ovr",
                                 average="macro", labels=classes)
        except ValueError:
            auc = float("nan")
        ll = log_loss(y_te, p_full, labels=classes)
        acc = float((pred == y_te).mean())
        per_class_auc = {}
        for k, c in enumerate(DISPO_CLASSES):
            try:
                per_class_auc[c] = float(
                    roc_auc_score((y_te == k).astype(int), p_full[:, k]))
            except ValueError:
                per_class_auc[c] = float("nan")
        per_class_brier = {DISPO_CLASSES[k]:
                            float(brier_score_loss((y_te == k).astype(int),
                                                    p_full[:, k]))
                            for k in classes}

        print(f"\n--- {name} ---")
        print(f"  log-loss        {ll:.4f}")
        print(f"  accuracy        {acc:.4f}")
        print(f"  macro AUC       {auc:.4f}")
        print(f"  per-class AUC   "
              f"D={per_class_auc['Discharge']:.3f}  "
              f"F={per_class_auc['Floor']:.3f}  "
              f"ICU={per_class_auc['ICU']:.3f}")

        rows.append({
            "model": name, "logloss": ll, "accuracy": acc,
            "macro_auc": auc,
            **{f"auc_{c.lower()}": per_class_auc[c] for c in DISPO_CLASSES},
            **{f"brier_{c.lower()}": per_class_brier[c]
               for c in DISPO_CLASSES},
        })
        for i, eid in enumerate(df.loc[is_test, "encounter_id"].values):
            pred_rows.append({
                "model": name, "encounter_id": eid,
                "true_label": DISPO_CLASSES[int(y_te[i])],
                "pred_label": DISPO_CLASSES[int(pred[i])],
                **{f"p_{c.lower()}": float(p_full[i, k])
                   for k, c in enumerate(DISPO_CLASSES)},
            })

    summary = pd.DataFrame(rows).set_index("model")
    print("\nSummary:")
    print(summary.to_string())

    best = summary["macro_auc"].idxmax()
    print(f"\nBest model by macro AUC: {best}")

    best_preds = [r for r in pred_rows if r["model"] == best]
    y_pred_best = [DISPO_CLASSES.index(r["pred_label"]) for r in best_preds]
    cm = confusion_matrix(y_te, y_pred_best,
                            labels=list(range(len(DISPO_CLASSES))))
    print(f"Confusion matrix ({best}, test = last day):")
    print(pd.DataFrame(cm, index=DISPO_CLASSES,
                        columns=DISPO_CLASSES).to_string())

    summary.to_csv(DERIVED / "task2_temporal_summary.csv")
    pd.DataFrame(pred_rows).to_csv(
        DERIVED / "task2_temporal_predictions.csv", index=False)
    print(f"\nSaved: derived/task2_temporal_summary.csv "
          f"+ task2_temporal_predictions.csv")
    return {"summary": summary, "best": best,
            "n_train": int(is_train.sum()), "n_test": int(is_test.sum()),
            "train_max": train_max, "test_date": test_date}


def _df_to_md(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [
            f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            for v in row.values]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    t1 = run_task1()
    t2 = run_task2()

    lines = ["# Temporal holdout evaluation\n",
             f"Train on all encounters with `encounter_arrival_date "
             f"<= {t1['train_max'].date()}`; test on "
             f"`{t1['test_date'].date()}` only.\n",
             "## Task 1 — drug ID at triage (4 classes)\n",
             f"Train n = {t1['n_train']}, Test n = {t1['n_test']}.\n",
             _df_to_md(t1["summary"]),
             f"\nBest model: **{t1['best']}**\n",
             "## Task 2 — deterioration at 4h (drug-positive cohort, 3 classes)\n",
             f"Train n = {t2['n_train']}, Test n = {t2['n_test']}.\n",
             _df_to_md(t2["summary"]),
             f"\nBest model: **{t2['best']}**\n"]
    (DERIVED / "temporal_holdout_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\nReport: derived/temporal_holdout_report.md")


if __name__ == "__main__":
    main()
