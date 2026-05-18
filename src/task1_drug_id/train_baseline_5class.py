"""Task 1 — direct 5-class drug-class classifier (triage features only).

Drop-in 5-class variant of `train_baseline.py`. The only structural
change is that CLASSES is auto-detected from the label column: if the
labels carry 5 levels, this script trains the 5-class problem; if it
sees 4 levels it falls back to the 4-class baseline behaviour (so it
is safe to run before the 5-class labels arrive — it just reproduces
the current §7a results until the labels are extended).

How to add the 5th class:
  Option A (preferred, no schema change): extend the existing
    `ground_truth_drug` column in `derived/outcomes.csv` so it carries
    values 0..4 (and `ground_truth_drug_name` carries the new name).
  Option B (separate file): drop the manually-labeled 5-class CSV at
    `derived/ground_truth_5class.csv` with columns
    [encounter_id, ground_truth_drug_5class, ground_truth_drug_5class_name].
    Pass `--labels 5class` to use it.

Class-name slot: by default we name the 5th class "Class 4"; replace
EXTRA_CLASS_NAME below once the manual label set arrives so the
output/report reads cleanly.

Time-leakage: ONLY uses features in features_triage.csv (no 4h, no
narrative HPI/MDM, no time-series). Same StratifiedKFold(5,
shuffle=True, random_state=42) as the 4-class baseline so results are
directly comparable.

Reports per model (5-fold stratified CV):
  - Log-loss
  - Top-1 accuracy
  - Macro one-vs-rest AUC + per-class OVR AUC
  - Macro PR-AUC + per-class
  - Per-class Brier + BSS
  - Confusion matrix
  - Classification report (precision/recall/F1)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              classification_report, confusion_matrix, log_loss,
                              roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

# 4-class canonical names (matches the rest of the codebase).
NAMES_4 = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]
# Placeholder for the 5th-class name — replace once the manual labels arrive.
EXTRA_CLASS_NAME = "Class 4"


# ---------- Data loading ----------------------------------------------

def load_data(labels_source: str = "auto"
              ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load features + multi-class drug labels.

    labels_source:
      "auto":   use `outcomes.csv :: ground_truth_drug` and auto-detect
                whether it carries 4 or 5 levels.
      "5class": read `derived/ground_truth_5class.csv`.
    """
    X = pd.read_csv(DERIVED / "features_triage.csv")

    if labels_source == "5class":
        labels_path = DERIVED / "ground_truth_5class.csv"
        if not labels_path.exists():
            raise SystemExit(
                f"--labels 5class requested but {labels_path} not present. "
                "Drop the manual 5-class labels there with columns "
                "[encounter_id, ground_truth_drug_5class, "
                "ground_truth_drug_5class_name].")
        outcomes = pd.read_csv(labels_path)
        y_col = "ground_truth_drug_5class"
        name_col = ("ground_truth_drug_5class_name"
                    if "ground_truth_drug_5class_name" in outcomes.columns
                    else None)
    else:
        outcomes = pd.read_csv(DERIVED / "outcomes.csv")
        y_col = "ground_truth_drug"
        name_col = ("ground_truth_drug_name"
                    if "ground_truth_drug_name" in outcomes.columns else None)

    # Defensive drops on the feature side
    for c in ("encounter_disposition_label", "ground_truth_drug",
               "ground_truth_drug_name", "ground_truth_drug_5class",
               "ground_truth_drug_5class_name"):
        if c in X.columns:
            X = X.drop(columns=[c])

    keep_cols = ["encounter_id", y_col] + ([name_col] if name_col else [])
    df = X.merge(outcomes[keep_cols], on="encounter_id", how="inner")
    assert len(df) == len(X), f"Row count mismatch ({len(df)} vs {len(X)})"

    drop = [c for c in ("encounter_id", "encounter_arrival_date",
                         "encounter_disposition_label", y_col,
                         name_col) if c and c in df.columns]
    y = df[y_col].to_numpy(dtype=int)
    X_df = df.drop(columns=drop)

    # Build class-name list of length K = max(y)+1
    K = int(y.max() + 1)
    if name_col is not None:
        mapping = (df[[y_col, name_col]].drop_duplicates()
                   .set_index(y_col)[name_col].to_dict())
        names = [mapping.get(i, f"Class {i}") for i in range(K)]
    elif K == 4:
        names = list(NAMES_4)
    elif K == 5:
        names = list(NAMES_4) + [EXTRA_CLASS_NAME]
    else:
        names = [f"Class {i}" for i in range(K)]

    return X_df, y, names


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

def evaluate(model_name: str, X: pd.DataFrame, y: np.ndarray,
             classes: list[str]) -> tuple[dict[str, float], np.ndarray]:
    K = len(classes)
    labels_idx = list(range(K))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), K))
    oof_pred = np.zeros(len(X), dtype=int)

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        pre = make_preprocessor(X)
        pre.fit(X.iloc[tr])
        X_tr = np.asarray(pre.transform(X.iloc[tr]), dtype=float)
        X_te = np.asarray(pre.transform(X.iloc[te]), dtype=float)

        clf = model_zoo()[model_name]
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)
        # Align probability columns to global class index in case a fold
        # missed a class
        full_proba = np.zeros((len(te), K))
        for col, cls in enumerate(clf.classes_):
            full_proba[:, int(cls)] = proba[:, col]
        oof_proba[te] = full_proba
        oof_pred[te] = full_proba.argmax(axis=1)

    eps = 1e-12
    metrics: dict[str, float] = {
        "model": model_name,
        "logloss": float(log_loss(y, np.clip(oof_proba, eps, 1.0),
                                    labels=labels_idx)),
        "accuracy": float((oof_pred == y).mean()),
    }
    try:
        metrics["macro_auc"] = float(
            roc_auc_score(y, oof_proba, multi_class="ovr",
                           average="macro", labels=labels_idx))
    except ValueError:
        metrics["macro_auc"] = float("nan")
    prauc_per = []
    for k, c in enumerate(classes):
        yb = (y == k).astype(int)
        prev = float(yb.mean())
        metrics[f"prevalence_{k}"] = prev
        try:
            metrics[f"auc_{k}"] = float(roc_auc_score(yb, oof_proba[:, k]))
        except ValueError:
            metrics[f"auc_{k}"] = float("nan")
        try:
            ap = float(average_precision_score(yb, oof_proba[:, k]))
        except ValueError:
            ap = float("nan")
        metrics[f"prauc_{k}"] = ap
        prauc_per.append(ap)
        brier = float(brier_score_loss(yb, oof_proba[:, k]))
        metrics[f"brier_{k}"] = brier
        denom = prev * (1 - prev)
        metrics[f"bss_{k}"] = (1 - brier / denom) if denom > 0 else float("nan")
    metrics["macro_prauc"] = float(np.nanmean(prauc_per))
    return metrics, oof_proba


# ---------- Main -------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", choices=["auto", "5class"], default="auto",
                    help="auto: read outcomes.csv (4- or 5-class). "
                         "5class: read derived/ground_truth_5class.csv.")
    args = ap.parse_args()

    X, y, classes = load_data(args.labels)
    K = len(classes)
    print(f"Loaded {len(y)} encounters, {K}-class target ({classes})")
    print(f"Prevalence: " + "  ".join(
        f"{c}={int((y == k).sum())}({(y == k).mean()*100:.1f}%)"
        for k, c in enumerate(classes)))

    rows = []
    oof_all = {}
    for name in ("logreg", "rforest", "hgb"):
        print(f"\n--- Training {name} ---")
        m, oof = evaluate(name, X, y, classes)
        rows.append(m)
        oof_all[name] = oof
        print(f"  log-loss        {m['logloss']:.4f}")
        print(f"  accuracy        {m['accuracy']:.4f}")
        print(f"  macro ROC-AUC   {m['macro_auc']:.4f}")
        print(f"  macro PR-AUC    {m['macro_prauc']:.4f}")
        for k, c in enumerate(classes):
            print(f"    {c:<14s} AUC={m[f'auc_{k}']:.3f}  "
                  f"PR-AUC={m[f'prauc_{k}']:.3f}  "
                  f"Brier={m[f'brier_{k}']:.3f}  "
                  f"BSS={m[f'bss_{k}']:+.3f}")

    summary = pd.DataFrame(rows)
    print(f"\n=== SUMMARY ({K}-class) ===")
    print(summary.to_string(index=False))
    best = summary.iloc[summary["macro_auc"].idxmax()]["model"]
    print(f"\nBest model by macro AUC: {best}")

    # Confusion matrix + classification report on the best model's OOF
    oof_best = oof_all[best]
    pred = oof_best.argmax(axis=1)
    print(f"\nConfusion matrix ({best}, OOF):")
    cm = confusion_matrix(y, pred, labels=list(range(K)))
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    print(f"\nClassification report ({best}, OOF):")
    print(classification_report(y, pred, labels=list(range(K)),
                                  target_names=classes, digits=3,
                                  zero_division=0))

    # Save artifacts
    out_summary = DERIVED / f"task1_baseline_{K}class_summary.csv"
    summary.to_csv(out_summary, index=False)
    oof_df = pd.DataFrame({"encounter_id":
                            pd.read_csv(DERIVED / "features_triage.csv"
                                          )["encounter_id"].to_numpy(),
                            "true_label_idx": y})
    for name, oof in oof_all.items():
        for k, c in enumerate(classes):
            oof_df[f"p_{name}_{k}_{c.split()[0]}"] = oof[:, k]
        oof_df[f"pred_label_{name}"] = oof.argmax(axis=1)
    out_oof = DERIVED / f"task1_oof_predictions_{K}class.csv"
    oof_df.to_csv(out_oof, index=False)
    print(f"\nSaved: {out_summary}")
    print(f"       {out_oof}")


if __name__ == "__main__":
    main()
