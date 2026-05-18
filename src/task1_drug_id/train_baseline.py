"""Task 1 baseline — drug-class classifier trained on triage-only features
against the **manually-annotated ground truth** from
derived/ground_truth.csv.

Switched from soft cross-entropy (10-agent LLM consensus) to hard
multinomial labels on 2026-05-17 — the manual labels are the gold
standard per the hackathon organizers and supersede the
note-derived soft labels.

Time-leakage: ONLY uses features in features_triage.csv. No 4-hour
data, no narrative HPI/MDM, no time-series.

Reports per model (5-fold stratified CV):
  - Log-loss
  - Top-1 accuracy
  - Macro one-vs-rest AUC
  - Per-class precision/recall/F1
  - Confusion matrix
  - Per-class Brier
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
                              log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

# Index follows the source file convention: 0=None, 1=Kraken,
# 2=Triton, 3=Coral (see src/labels/load_ground_truth.py).
CLASSES = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]


# ---------- Data loading ----------------------------------------------

def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_csv(DERIVED / "features_triage.csv")
    outcomes = pd.read_csv(DERIVED / "outcomes.csv")[
        ["encounter_id", "ground_truth_drug"]]
    # Defensive drop in case a stale features_triage.csv still
    # carries the outcome columns (it shouldn't, post-leakage-fix).
    for c in ("encounter_disposition_label", "ground_truth_drug",
               "ground_truth_drug_name"):
        if c in X.columns:
            X = X.drop(columns=[c])
    df = X.merge(outcomes, on="encounter_id", how="inner")
    assert len(df) == len(X) == len(outcomes), "Row count mismatch"

    # Drop columns that are not legal Task-1 inputs
    drop = [
        "encounter_id",
        "encounter_arrival_date",
        "encounter_disposition_label",  # Task 2 target, never a feature
    ]
    drop = [c for c in drop if c in df.columns]
    y = df["ground_truth_drug"].to_numpy(dtype=int)
    X_df = df.drop(columns=drop + ["ground_truth_drug"])
    # Guard: target must not leak into X
    assert "ground_truth_drug" not in X_df.columns
    assert "encounter_disposition_label" not in X_df.columns
    return X_df, y


# ---------- Feature pipeline ------------------------------------------

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    text_col = "triage_brief_note" if "triage_brief_note" in X.columns else None
    obj_cols = [c for c in X.select_dtypes(include=["object", "string"]).columns
                if c != text_col]
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

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


# ---------- Models -----------------------------------------------------

def model_zoo() -> dict[str, object]:
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


# ---------- Evaluation -------------------------------------------------

def evaluate(model_name: str, X: pd.DataFrame,
             y: np.ndarray) -> dict[str, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_logloss = []
    fold_acc = []
    fold_auc_macro = []
    fold_auc_per_class = {c: [] for c in CLASSES}
    fold_prauc_macro = []
    fold_prauc = {c: [] for c in CLASSES}
    fold_brier = {c: [] for c in CLASSES}
    fold_bss = {c: [] for c in CLASSES}
    fold_prev = {c: [] for c in CLASSES}

    oof_proba = np.zeros((len(X), len(CLASSES)))
    oof_pred = np.zeros(len(X), dtype=int)

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        pre = make_preprocessor(X)
        pre.fit(X.iloc[tr])
        X_tr_mat = pre.transform(X.iloc[tr])
        X_te_mat = pre.transform(X.iloc[te])
        if hasattr(X_tr_mat, "toarray"):
            X_tr_mat = X_tr_mat.toarray()
            X_te_mat = X_te_mat.toarray()
        X_tr_mat = np.asarray(X_tr_mat, dtype=float)
        X_te_mat = np.asarray(X_te_mat, dtype=float)

        model = model_zoo()[model_name]
        model.fit(X_tr_mat, y[tr])

        p_pred = model.predict_proba(X_te_mat)
        col_order = list(model.classes_)
        if col_order != list(range(len(CLASSES))):
            order = [col_order.index(i) for i in range(len(CLASSES))]
            p_pred = p_pred[:, order]
        oof_proba[te] = p_pred
        oof_pred[te] = p_pred.argmax(axis=1)

        eps = 1e-12
        # Log-loss vs hard target
        from sklearn.metrics import log_loss
        fold_logloss.append(log_loss(y[te], p_pred,
                                       labels=list(range(len(CLASSES)))))
        fold_acc.append(float((p_pred.argmax(axis=1) == y[te]).mean()))
        try:
            fold_auc_macro.append(
                roc_auc_score(y[te], p_pred, multi_class="ovr",
                              average="macro",
                              labels=list(range(len(CLASSES)))))
        except ValueError:
            fold_auc_macro.append(float("nan"))

        for k, c in enumerate(CLASSES):
            y_bin = (y[te] == k).astype(int)
            prev = float(y_bin.mean())
            fold_prev[c].append(prev)
            brier_val = float(brier_score_loss(y_bin, p_pred[:, k]))
            fold_brier[c].append(brier_val)
            denom = prev * (1 - prev)
            fold_bss[c].append((1.0 - brier_val / denom)
                                 if denom > 0 else float("nan"))
            try:
                fold_auc_per_class[c].append(
                    roc_auc_score(y_bin, p_pred[:, k]))
            except ValueError:
                fold_auc_per_class[c].append(float("nan"))
            try:
                fold_prauc[c].append(
                    average_precision_score(y_bin, p_pred[:, k]))
            except ValueError:
                fold_prauc[c].append(float("nan"))
        # macro PR-AUC = mean of per-class AP across the 4 classes
        fold_prauc_macro.append(
            float(np.nanmean([fold_prauc[c][-1] for c in CLASSES])))

    return {
        "model": model_name,
        "logloss_mean": float(np.mean(fold_logloss)),
        "logloss_std": float(np.std(fold_logloss)),
        "acc_mean": float(np.mean(fold_acc)),
        "acc_std": float(np.std(fold_acc)),
        "auc_macro_mean": float(np.nanmean(fold_auc_macro)),
        "auc_macro_std": float(np.nanstd(fold_auc_macro)),
        "auc_none": float(np.nanmean(fold_auc_per_class["None"])),
        "auc_kraken": float(np.nanmean(fold_auc_per_class["Kraken Candy"])),
        "auc_triton": float(np.nanmean(fold_auc_per_class["Triton Tabs"])),
        "auc_coral": float(np.nanmean(fold_auc_per_class["Coral Dust"])),
        "prauc_macro_mean": float(np.nanmean(fold_prauc_macro)),
        "prauc_macro_std": float(np.nanstd(fold_prauc_macro)),
        "prauc_none": float(np.nanmean(fold_prauc["None"])),
        "prauc_kraken": float(np.nanmean(fold_prauc["Kraken Candy"])),
        "prauc_triton": float(np.nanmean(fold_prauc["Triton Tabs"])),
        "prauc_coral": float(np.nanmean(fold_prauc["Coral Dust"])),
        "brier_none": float(np.mean(fold_brier["None"])),
        "brier_kraken": float(np.mean(fold_brier["Kraken Candy"])),
        "brier_triton": float(np.mean(fold_brier["Triton Tabs"])),
        "brier_coral": float(np.mean(fold_brier["Coral Dust"])),
        "bss_none": float(np.nanmean(fold_bss["None"])),
        "bss_kraken": float(np.nanmean(fold_bss["Kraken Candy"])),
        "bss_triton": float(np.nanmean(fold_bss["Triton Tabs"])),
        "bss_coral": float(np.nanmean(fold_bss["Coral Dust"])),
        "prevalence_none": float(np.mean(fold_prev["None"])),
        "prevalence_kraken": float(np.mean(fold_prev["Kraken Candy"])),
        "prevalence_triton": float(np.mean(fold_prev["Triton Tabs"])),
        "prevalence_coral": float(np.mean(fold_prev["Coral Dust"])),
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
    }


# ---------- Main ------------------------------------------------------

def main() -> None:
    X, y = load_data()
    label_names = [CLASSES[i] for i in y]
    print(f"Loaded: X={X.shape}, y={y.shape}")
    print(f"Ground-truth class distribution:")
    for cls in CLASSES:
        n = sum(1 for v in label_names if v == cls)
        print(f"  {cls:14s}  {n:>3d}  ({n/len(y)*100:5.1f}%)")

    majority_class = int(np.bincount(y).argmax())
    print(f"\nMajority-class baseline accuracy: "
          f"{(y == majority_class).mean():.4f} "
          f"(always predict '{CLASSES[majority_class]}')")
    marginal = np.bincount(y, minlength=len(CLASSES)) / len(y)
    marg_ll = -np.mean([np.log(np.clip(marginal[y[i]], 1e-12, 1.0))
                         for i in range(len(y))])
    print(f"Marginal-only log-loss: {marg_ll:.4f}  "
          f"(uniform = {np.log(len(CLASSES)):.4f})")

    results = []
    oof_store: dict[str, dict] = {}
    for name in model_zoo():
        print(f"\n--- Training {name} ---")
        r = evaluate(name, X, y)
        oof_store[name] = {"proba": r.pop("oof_proba"),
                            "pred": r.pop("oof_pred")}
        results.append(r)
        print(f"  log-loss:        {r['logloss_mean']:.4f} "
              f"(+/- {r['logloss_std']:.4f})")
        print(f"  accuracy:        {r['acc_mean']:.4f} "
              f"(+/- {r['acc_std']:.4f})")
        print(f"  macro ROC-AUC:   {r['auc_macro_mean']:.4f} "
              f"(+/- {r['auc_macro_std']:.4f})")
        print(f"  OVR ROC-AUC:     "
              f"n={r['auc_none']:.3f}  k={r['auc_kraken']:.3f}  "
              f"t={r['auc_triton']:.3f}  c={r['auc_coral']:.3f}")
        print(f"  macro PR-AUC:    {r['prauc_macro_mean']:.4f} "
              f"(+/- {r['prauc_macro_std']:.4f})")
        print(f"  OVR PR-AUC:      "
              f"n={r['prauc_none']:.3f}  k={r['prauc_kraken']:.3f}  "
              f"t={r['prauc_triton']:.3f}  c={r['prauc_coral']:.3f}")
        print(f"  prevalence:      "
              f"n={r['prevalence_none']:.3f}  k={r['prevalence_kraken']:.3f}  "
              f"t={r['prevalence_triton']:.3f}  c={r['prevalence_coral']:.3f}")
        print(f"  Brier per class: "
              f"n={r['brier_none']:.3f}  k={r['brier_kraken']:.3f}  "
              f"t={r['brier_triton']:.3f}  c={r['brier_coral']:.3f}")
        print(f"  Brier Skill Sc:  "
              f"n={r['bss_none']:+.3f}  k={r['bss_kraken']:+.3f}  "
              f"t={r['bss_triton']:+.3f}  c={r['bss_coral']:+.3f}")

    print("\n" + "=" * 78)
    print("SUMMARY (5-fold stratified CV, ground-truth labels)")
    print("=" * 78)
    summary = pd.DataFrame(results).set_index("model")
    print(summary.to_string())

    best = summary["auc_macro_mean"].idxmax()
    print(f"\nBest model by macro AUC: {best}")

    from sklearn.metrics import classification_report, confusion_matrix
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

    oof = pd.DataFrame(oof_store[best]["proba"],
                        columns=[f"pred_p_{c.lower().replace(' candy','').replace(' tabs','').replace(' dust','')}"
                                 for c in CLASSES])
    ids = pd.read_csv(DERIVED / "features_triage.csv")["encounter_id"]
    oof.insert(0, "encounter_id", ids.values)
    oof["pred_argmax"] = [CLASSES[i] for i in oof_store[best]["pred"]]
    oof["true_label"] = [CLASSES[i] for i in y]
    oof_path = DERIVED / "task1_oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    summary_path = DERIVED / "task1_baseline_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved: {oof_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
