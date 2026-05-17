"""Task 1 baseline — drug-class classifier trained on triage-only features
with soft cross-entropy targets from derived/probs_avg.csv.

Soft-label training trick: for each encounter with target
distribution (p_kraken, p_triton, p_coral, p_none), we expand the
row into 4 weighted copies (one per class) with sample_weight = p_i.
Sklearn classifiers minimize the weighted multinomial log-loss in
this form, which is exactly cross-entropy against the soft target.

Time-leakage: ONLY uses features in features_triage.csv (37 cols).
No 4-hour data, no narrative HPI/MDM, no time-series.

Reports per model:
  - 5-fold stratified CV (stratified on argmax of soft target)
  - Log-loss against soft target  (training objective)
  - Top-1 accuracy against argmax (intuition only — argmax is not
    ground truth, it's the agent consensus)
  - Macro one-vs-rest AUC (against argmax)
  - Per-class Brier score
  - Mean predicted entropy (how confident is the model?)
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
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

CLASSES = ["Kraken Candy", "Triton Tabs", "Coral Dust", "None"]
PROB_COLS = ["p_kraken", "p_triton", "p_coral", "p_none"]


# ---------- Data loading ----------------------------------------------

def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X = pd.read_csv(DERIVED / "features_triage.csv")
    y_soft_df = pd.read_csv(DERIVED / "probs_avg.csv")[
        ["encounter_id", *PROB_COLS]]
    df = X.merge(y_soft_df, on="encounter_id", how="inner")
    assert len(df) == len(X) == len(y_soft_df), "Row count mismatch"

    # Drop columns that are not legal Task-1 inputs
    drop = [
        "encounter_id",
        "encounter_arrival_date",
        "encounter_disposition_label",  # Task 2 target, not Task 1 input
    ]
    drop = [c for c in drop if c in df.columns]
    y_soft = df[PROB_COLS].to_numpy(dtype=float)
    X_df = df.drop(columns=drop + PROB_COLS)
    argmax = y_soft.argmax(axis=1)
    return X_df, y_soft, argmax


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


# ---------- Soft-label expansion --------------------------------------

def expand_soft(X: np.ndarray, y_soft: np.ndarray,
                min_weight: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand each row into K class-rows with sample_weight = p_k."""
    n, K = y_soft.shape
    rows = np.repeat(np.arange(n), K)
    classes = np.tile(np.arange(K), n)
    weights = y_soft.flatten()

    # Drop near-zero-weight expansions for speed
    keep = weights > min_weight
    return X[rows[keep]], classes[keep], weights[keep]


# ---------- Models -----------------------------------------------------

def model_zoo() -> dict[str, object]:
    return {
        "logreg":  LogisticRegression(max_iter=2000, C=0.5,
                                       class_weight=None,
                                       solver="lbfgs", n_jobs=1),
        "rforest": RandomForestClassifier(n_estimators=400, max_depth=8,
                                           min_samples_leaf=4,
                                           random_state=42, n_jobs=-1),
        "hgb":     HistGradientBoostingClassifier(max_iter=300,
                                                   max_depth=6,
                                                   learning_rate=0.05,
                                                   l2_regularization=0.5,
                                                   random_state=42),
    }


# ---------- Evaluation -------------------------------------------------

def evaluate(model_name: str, X: pd.DataFrame,
             y_soft: np.ndarray, argmax: np.ndarray) -> dict[str, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preprocessor = make_preprocessor(X)

    fold_logloss = []
    fold_acc = []
    fold_auc_macro = []
    fold_brier = {c: [] for c in CLASSES}
    fold_entropy = []

    oof_proba = np.zeros((len(X), len(CLASSES)))

    for fold, (tr, te) in enumerate(skf.split(X, argmax)):
        X_tr = X.iloc[tr]
        X_te = X.iloc[te]
        pre = preprocessor
        pre.fit(X_tr)
        X_tr_mat = pre.transform(X_tr)
        X_te_mat = pre.transform(X_te)
        if hasattr(X_tr_mat, "toarray"):
            X_tr_mat = X_tr_mat.toarray()
            X_te_mat = X_te_mat.toarray()
        X_tr_mat = np.asarray(X_tr_mat, dtype=float)
        X_te_mat = np.asarray(X_te_mat, dtype=float)

        X_exp, y_exp, w_exp = expand_soft(X_tr_mat, y_soft[tr])

        model = {**model_zoo()}[model_name]
        model.fit(X_exp, y_exp, sample_weight=w_exp)

        p_pred = model.predict_proba(X_te_mat)  # (n_te, K)
        # Ensure column order matches CLASSES (model uses 0..K-1 by class_)
        col_order = list(model.classes_)
        # Reorder if needed
        if col_order != list(range(len(CLASSES))):
            order = [col_order.index(i) for i in range(len(CLASSES))]
            p_pred = p_pred[:, order]
        oof_proba[te] = p_pred

        # Log-loss vs soft target (true cross-entropy)
        eps = 1e-12
        ll = -(y_soft[te] * np.log(np.clip(p_pred, eps, 1.0))).sum(axis=1).mean()
        fold_logloss.append(ll)

        # Top-1 accuracy vs argmax
        acc = float((p_pred.argmax(axis=1) == argmax[te]).mean())
        fold_acc.append(acc)

        # Macro one-vs-rest AUC against argmax
        try:
            auc = roc_auc_score(argmax[te], p_pred, multi_class="ovr",
                                  average="macro", labels=list(range(len(CLASSES))))
        except ValueError:
            auc = float("nan")
        fold_auc_macro.append(auc)

        # Per-class Brier (one-vs-rest)
        for k, c in enumerate(CLASSES):
            try:
                fold_brier[c].append(
                    brier_score_loss((argmax[te] == k).astype(int),
                                     p_pred[:, k]))
            except ValueError:
                fold_brier[c].append(float("nan"))

        # Entropy
        ent = -(p_pred * np.log(np.clip(p_pred, eps, 1.0))).sum(axis=1).mean()
        fold_entropy.append(ent)

    return {
        "model": model_name,
        "logloss_mean": float(np.mean(fold_logloss)),
        "logloss_std": float(np.std(fold_logloss)),
        "argmax_acc_mean": float(np.mean(fold_acc)),
        "argmax_acc_std": float(np.std(fold_acc)),
        "auc_macro_mean": float(np.nanmean(fold_auc_macro)),
        "auc_macro_std": float(np.nanstd(fold_auc_macro)),
        "brier_kraken": float(np.mean(fold_brier["Kraken Candy"])),
        "brier_triton": float(np.mean(fold_brier["Triton Tabs"])),
        "brier_coral": float(np.mean(fold_brier["Coral Dust"])),
        "brier_none": float(np.mean(fold_brier["None"])),
        "entropy_mean": float(np.mean(fold_entropy)),
        "oof_proba": oof_proba,
    }


# ---------- Main ------------------------------------------------------

def main() -> None:
    X, y_soft, argmax = load_data()
    print(f"Loaded: X={X.shape}, y_soft={y_soft.shape}")
    print(f"Argmax label distribution: "
          f"{pd.Series([CLASSES[i] for i in argmax]).value_counts().to_dict()}")

    results = []
    oof_store: dict[str, np.ndarray] = {}
    for name in model_zoo():
        print(f"\n--- Training {name} ---")
        r = evaluate(name, X, y_soft, argmax)
        oof_store[name] = r.pop("oof_proba")
        results.append(r)
        print(f"  log-loss vs soft target: {r['logloss_mean']:.4f} "
              f"(+/- {r['logloss_std']:.4f})")
        print(f"  argmax accuracy:         {r['argmax_acc_mean']:.4f} "
              f"(+/- {r['argmax_acc_std']:.4f})")
        print(f"  macro AUC (vs argmax):   {r['auc_macro_mean']:.4f} "
              f"(+/- {r['auc_macro_std']:.4f})")
        print(f"  Brier per class: "
              f"k={r['brier_kraken']:.3f}  t={r['brier_triton']:.3f}  "
              f"c={r['brier_coral']:.3f}  n={r['brier_none']:.3f}")
        print(f"  mean predicted entropy:  {r['entropy_mean']:.4f} "
              f"(max possible = {np.log(4):.4f})")

    # Baseline reference: predict the marginal (no features)
    marginal = y_soft.mean(axis=0)
    eps = 1e-12
    marginal_logloss = -(y_soft * np.log(np.clip(marginal, eps, 1.0))).sum(axis=1).mean()
    print(f"\nMarginal-only log-loss (predict mean(y) for everyone): "
          f"{marginal_logloss:.4f}")
    print(f"Uniform-prediction log-loss (0.25 each):                 "
          f"{np.log(4):.4f}")

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY (5-fold stratified CV, lower log-loss is better)")
    print("=" * 78)
    summary = pd.DataFrame(results).set_index("model")
    print(summary.to_string())

    # Save OOF predictions for the best model
    best = summary["logloss_mean"].idxmin()
    print(f"\nBest model by log-loss: {best}")
    oof = pd.DataFrame(oof_store[best], columns=[f"pred_{c}" for c in PROB_COLS])
    oof.insert(0, "encounter_id",
               pd.read_csv(DERIVED / "features_triage.csv")["encounter_id"])
    oof_path = DERIVED / "task1_oof_predictions.csv"
    oof.to_csv(oof_path, index=False)
    print(f"OOF predictions saved: {oof_path}")

    summary_path = DERIVED / "task1_baseline_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
