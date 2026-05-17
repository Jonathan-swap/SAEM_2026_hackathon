"""Task 2 baseline — deterioration index (Discharge / Floor / ICU).

Per the hackathon brief: Task 2 predicts disposition for patients we
identified as drug-positive in Task 1 (we don't owe predictions for
non-festival typical-medical-pathology patients).

Cohort filter: probs_avg.csv argmax != "None" -> drug-positive subset.

Inputs:
  derived/features_fourh.csv  (Task 2 horizon — 4 hours)
  derived/probs_avg.csv       (Task 1 outputs — used both as a cohort
                               filter AND as additional features, since
                               at 4h the working-impression drug class
                               is part of clinical context.)

Reports:
  5-fold stratified CV on the drug-positive cohort
  - macro one-vs-rest AUC (Discharge vs rest, Floor vs rest, ICU vs rest)
  - log-loss (proper score)
  - per-class precision / recall / F1
  - confusion matrix
  - per-class Brier
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
                              confusion_matrix, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

CLASSES = ["Discharge", "Floor", "ICU"]
PROB_COLS = ["p_kraken", "p_triton", "p_coral", "p_none"]


def load_data(use_drug_probs_as_features: bool = True
              ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X = pd.read_csv(DERIVED / "features_fourh.csv")
    # keep_default_na=False so the string "None" survives as a value
    probs = pd.read_csv(DERIVED / "probs_avg.csv",
                         keep_default_na=False, na_values=[""])[
        ["encounter_id", "argmax_class", *PROB_COLS]]

    df = X.merge(probs, on="encounter_id", how="inner")

    # Cohort filter: drug-positive only (argmax != None)
    n_before = len(df)
    df = df[df["argmax_class"] != "None"].reset_index(drop=True)
    print(f"Cohort filter (argmax != 'None'): {n_before} -> {len(df)} patients")

    # Drop columns that should not be features
    drop = [
        "encounter_id",
        "encounter_arrival_date",
        "argmax_class",  # categorical version of the prob columns; redundant
    ]
    # Optionally remove drug-class probs from features
    if not use_drug_probs_as_features:
        drop += PROB_COLS

    y_label = df["encounter_disposition_label"].copy()
    drop.append("encounter_disposition_label")
    X_df = df.drop(columns=[c for c in drop if c in df.columns])

    # Encode y as 0=Discharge, 1=Floor, 2=ICU
    label_map = {c: i for i, c in enumerate(CLASSES)}
    y = y_label.map(label_map).to_numpy()
    return X_df, y, y_label.to_numpy()


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    text_col = "triage_brief_note" if "triage_brief_note" in X.columns else None
    obj_cols = [c for c in X.select_dtypes(include=["object", "string"]).columns
                if c != text_col]
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    num_cols = list(set(num_cols + bool_cols))

    transformers = []
    if text_col:
        transformers.append(("text", TfidfVectorizer(
            max_features=50, stop_words="english", ngram_range=(1, 2),
            min_df=3), text_col))
    if obj_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), obj_cols))
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols))
    return ColumnTransformer(transformers, remainder="drop")


def model_zoo() -> dict:
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


def evaluate(model_name: str, X: pd.DataFrame,
             y: np.ndarray) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preprocessor = make_preprocessor(X)

    fold_logloss = []
    fold_acc = []
    fold_auc_macro = []
    fold_auc_per_class = {c: [] for c in CLASSES}
    fold_brier = {c: [] for c in CLASSES}

    oof_proba = np.zeros((len(X), len(CLASSES)))
    oof_pred = np.zeros(len(X), dtype=int)

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        pre = make_preprocessor(X)
        pre.fit(X.iloc[tr])
        X_tr = pre.transform(X.iloc[tr])
        X_te = pre.transform(X.iloc[te])
        if hasattr(X_tr, "toarray"):
            X_tr = X_tr.toarray()
            X_te = X_te.toarray()
        X_tr = np.asarray(X_tr, dtype=float)
        X_te = np.asarray(X_te, dtype=float)

        model = model_zoo()[model_name]
        model.fit(X_tr, y[tr])
        p = model.predict_proba(X_te)
        col_order = list(model.classes_)
        if col_order != list(range(len(CLASSES))):
            order = [col_order.index(i) for i in range(len(CLASSES))]
            p = p[:, order]

        oof_proba[te] = p
        oof_pred[te] = p.argmax(axis=1)

        fold_logloss.append(log_loss(y[te], p, labels=list(range(len(CLASSES)))))
        fold_acc.append(float((p.argmax(axis=1) == y[te]).mean()))
        try:
            fold_auc_macro.append(roc_auc_score(y[te], p, multi_class="ovr",
                                                  average="macro",
                                                  labels=list(range(len(CLASSES)))))
        except ValueError:
            fold_auc_macro.append(float("nan"))

        for k, c in enumerate(CLASSES):
            try:
                fold_auc_per_class[c].append(
                    roc_auc_score((y[te] == k).astype(int), p[:, k]))
            except ValueError:
                fold_auc_per_class[c].append(float("nan"))
            fold_brier[c].append(
                brier_score_loss((y[te] == k).astype(int), p[:, k]))

    return {
        "model": model_name,
        "logloss_mean": float(np.mean(fold_logloss)),
        "logloss_std": float(np.std(fold_logloss)),
        "acc_mean": float(np.mean(fold_acc)),
        "acc_std": float(np.std(fold_acc)),
        "auc_macro_mean": float(np.nanmean(fold_auc_macro)),
        "auc_macro_std": float(np.nanstd(fold_auc_macro)),
        "auc_discharge": float(np.nanmean(fold_auc_per_class["Discharge"])),
        "auc_floor": float(np.nanmean(fold_auc_per_class["Floor"])),
        "auc_icu": float(np.nanmean(fold_auc_per_class["ICU"])),
        "brier_discharge": float(np.mean(fold_brier["Discharge"])),
        "brier_floor": float(np.mean(fold_brier["Floor"])),
        "brier_icu": float(np.mean(fold_brier["ICU"])),
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
    }


def main() -> None:
    print("="*78)
    print("Task 2 baseline — disposition prediction (Discharge / Floor / ICU)")
    print("="*78)

    # Two variants: with and without Task-1 drug-class probabilities
    for variant_name, use_drug_probs in [
        ("WITH drug-class probs (full 4h context)", True),
        ("WITHOUT drug-class probs (clinical features only)", False),
    ]:
        print(f"\n\n###### VARIANT: {variant_name} ######")
        X, y, y_label = load_data(use_drug_probs_as_features=use_drug_probs)

        print(f"X: {X.shape}, y: {y.shape}")
        print(f"Disposition distribution: "
              f"{pd.Series(y_label).value_counts().to_dict()}")

        baseline_class = np.bincount(y).argmax()
        print(f"Majority-class baseline accuracy: "
              f"{(y == baseline_class).mean():.4f} "
              f"(always predict '{CLASSES[baseline_class]}')")

        # Marginal log-loss
        marginal = np.bincount(y, minlength=len(CLASSES)) / len(y)
        marg_ll = -np.mean([np.log(np.clip(marginal[y[i]], 1e-12, 1.0))
                             for i in range(len(y))])
        print(f"Marginal-only log-loss: {marg_ll:.4f}  "
              f"(uniform = {np.log(len(CLASSES)):.4f})")

        results = []
        oof_store: dict[str, dict] = {}
        for name in model_zoo():
            print(f"\n--- {name} ---")
            r = evaluate(name, X, y)
            oof_store[name] = {"proba": r.pop("oof_proba"),
                                "pred": r.pop("oof_pred")}
            results.append(r)
            print(f"  log-loss:        {r['logloss_mean']:.4f} "
                  f"(+/- {r['logloss_std']:.4f})")
            print(f"  accuracy:        {r['acc_mean']:.4f} "
                  f"(+/- {r['acc_std']:.4f})")
            print(f"  macro AUC:       {r['auc_macro_mean']:.4f} "
                  f"(+/- {r['auc_macro_std']:.4f})")
            print(f"  per-class AUC:   "
                  f"D={r['auc_discharge']:.3f}  F={r['auc_floor']:.3f}  "
                  f"ICU={r['auc_icu']:.3f}")
            print(f"  per-class Brier: "
                  f"D={r['brier_discharge']:.3f}  F={r['brier_floor']:.3f}  "
                  f"ICU={r['brier_icu']:.3f}")

        print(f"\n--- SUMMARY ({variant_name}) ---")
        summary = pd.DataFrame(results).set_index("model")
        print(summary.to_string())

        best = summary["auc_macro_mean"].idxmax()
        print(f"\nBest model by macro AUC: {best}")

        # Confusion matrix for best model (OOF)
        best_pred = oof_store[best]["pred"]
        print(f"\nConfusion matrix ({best}, OOF, rows=true, cols=pred):")
        cm = confusion_matrix(y, best_pred, labels=list(range(len(CLASSES))))
        cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
        print(cm_df.to_string())

        print(f"\nClassification report ({best}, OOF):")
        print(classification_report(y, best_pred,
                                      labels=list(range(len(CLASSES))),
                                      target_names=CLASSES, digits=3,
                                      zero_division=0))

        # Save artifacts for the WITH-probs variant
        if use_drug_probs:
            ids = pd.read_csv(DERIVED / "features_fourh.csv")[
                ["encounter_id"]]
            argmax_class = pd.read_csv(DERIVED / "probs_avg.csv",
                                         keep_default_na=False,
                                         na_values=[""])[
                ["encounter_id", "argmax_class"]]
            ids = ids.merge(argmax_class, on="encounter_id")
            cohort = ids[ids["argmax_class"] != "None"].reset_index(drop=True)
            oof = pd.DataFrame(oof_store[best]["proba"],
                                columns=[f"p_{c}" for c in CLASSES])
            oof.insert(0, "encounter_id", cohort["encounter_id"].values)
            oof["pred_argmax"] = [CLASSES[i] for i in oof_store[best]["pred"]]
            oof["true_label"] = y_label
            oof_path = DERIVED / "task2_oof_predictions.csv"
            oof.to_csv(oof_path, index=False)
            summary_path = DERIVED / "task2_baseline_summary.csv"
            summary.to_csv(summary_path)
            print(f"\nSaved: {oof_path}")
            print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
