"""Apples-to-apples comparison: 4-class direct vs cascade (tier-1 + tier-2).

Two architectures, same CV folds + same temporal split, same model
choices on both branches.

  - **Direct**: one 4-class classifier predicting
    (None, Kraken, Triton, Coral) directly.
  - **Cascade**:
      tier-1 binary    → P(drug)
      tier-2 multi     → P(K | drug), P(T | drug), P(C | drug)
      combine:         P(None) = 1 - P(drug)
                       P(K)    = P(drug) * P(K | drug)   etc.
    All cascade probabilities sum to 1 by construction.

For each of {logreg, rforest, hgb}, runs the same fold split twice
(once for each architecture) so the comparison is paired.

Reports macro ROC-AUC, macro PR-AUC, accuracy, per-class OVR
ROC-AUC, per-class Brier Skill Score on:

  - 5-fold stratified CV  (n=261)
  - Temporal holdout       (train days < last, test = last day)

Outputs:
  derived/task1_cascade_vs_direct_summary.csv
  derived/task1_cascade_vs_direct_report.md
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

# Final 4-class index — 0=None, 1=Kraken, 2=Triton, 3=Coral.
# Matches outcomes.csv :: ground_truth_drug.
CLASSES4 = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]
DRUG_CLASSES = ["Kraken Candy", "Triton Tabs", "Coral Dust"]
TEXT_COL = "triage_brief_note"


# ---------- Data loading ----------------------------------------------

def load_features_and_y() -> tuple[pd.DataFrame, np.ndarray, pd.Series,
                                    np.ndarray]:
    """Returns X_df (no outcomes), y (0..3), arrival_date, encounter_ids."""
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
    X_df = df.drop(columns=drop)
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


def model_factory(name: str):
    """Fresh classifier instance — used for direct, tier-1, and tier-2."""
    if name == "logreg":
        return LogisticRegression(max_iter=3000, C=0.5,
                                    class_weight="balanced",
                                    solver="lbfgs")
    if name == "rforest":
        return RandomForestClassifier(n_estimators=400, max_depth=8,
                                        min_samples_leaf=4,
                                        class_weight="balanced",
                                        random_state=42, n_jobs=-1)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                                learning_rate=0.05,
                                                l2_regularization=0.5,
                                                class_weight="balanced",
                                                random_state=42)
    raise ValueError(name)


# ---------- Train/predict helpers ------------------------------------

def fit_direct(model_name: str, X_tr_mat: np.ndarray,
                y_tr: np.ndarray, X_te_mat: np.ndarray) -> np.ndarray:
    """4-class direct. Returns prob matrix of shape (n_test, 4)."""
    mdl = model_factory(model_name)
    mdl.fit(X_tr_mat, y_tr)
    p_seen = mdl.predict_proba(X_te_mat)
    p_full = np.zeros((len(X_te_mat), 4))
    for j, cls in enumerate(mdl.classes_):
        p_full[:, cls] = p_seen[:, j]
    return p_full


def fit_cascade(model_name: str,
                  X_tr_mat: np.ndarray, y_tr_4cls: np.ndarray,
                  X_te_mat: np.ndarray) -> np.ndarray:
    """Tier-1 (binary) + tier-2 (multi over drug-positive).

    Returns 4-class prob matrix for the test rows, summing to 1.
    """
    # Tier-1: binary drug vs no drug, fit on all train rows
    y_tr_bin = (y_tr_4cls != 0).astype(int)
    mdl1 = model_factory(model_name)
    mdl1.fit(X_tr_mat, y_tr_bin)
    p_drug = mdl1.predict_proba(X_te_mat)[:, 1]

    # Tier-2: drug class (1/2/3) on drug-positive train subset
    mask = (y_tr_4cls != 0)
    y_tr_drug = y_tr_4cls[mask]
    mdl2 = model_factory(model_name)
    mdl2.fit(X_tr_mat[mask], y_tr_drug)
    p_seen_drug = mdl2.predict_proba(X_te_mat)
    # Map mdl2.classes_ (subset of {1,2,3}) -> local indices 1..3
    p_drug_cond = np.zeros((len(X_te_mat), 3))
    for j, cls in enumerate(mdl2.classes_):
        # cls is in {1,2,3}; local-index for the 3-vector = cls - 1
        p_drug_cond[:, cls - 1] = p_seen_drug[:, j]

    # Combine into 4-class probabilities that sum to 1
    p_full = np.zeros((len(X_te_mat), 4))
    p_full[:, 0] = 1.0 - p_drug                              # P(None)
    p_full[:, 1] = p_drug * p_drug_cond[:, 0]                # P(K)
    p_full[:, 2] = p_drug * p_drug_cond[:, 1]                # P(T)
    p_full[:, 3] = p_drug * p_drug_cond[:, 2]                # P(C)
    # Tiny numerical drift: renormalise so rows sum to exactly 1
    p_full = p_full / p_full.sum(axis=1, keepdims=True)
    return p_full


# ---------- Metrics ---------------------------------------------------

def metric_pack(y_true: np.ndarray, p_full: np.ndarray) -> dict:
    classes_idx = list(range(4))
    pred = p_full.argmax(axis=1)
    try:
        auc_macro = float(roc_auc_score(y_true, p_full,
                                           multi_class="ovr",
                                           average="macro",
                                           labels=classes_idx))
    except ValueError:
        auc_macro = float("nan")
    try:
        ll = float(log_loss(y_true, np.clip(p_full, 1e-12, 1.0),
                              labels=classes_idx))
    except ValueError:
        ll = float("nan")
    acc = float((pred == y_true).mean())
    per_auc = {}
    per_prauc = {}
    per_brier = {}
    per_bss = {}
    prev = {}
    for k, c in enumerate(CLASSES4):
        y_bin = (y_true == k).astype(int)
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
        "per_auc": per_auc, "per_prauc": per_prauc,
        "per_brier": per_brier, "per_bss": per_bss,
        "prevalence": prev,
    }


# ---------- CV evaluation --------------------------------------------

def run_cv() -> list[dict]:
    print("=" * 78)
    print("CV: direct 4-class vs cascade (tier-1 + tier-2)")
    print("=" * 78)
    X_df, y, _, _ = load_features_and_y()
    print(f"X: {X_df.shape}   y: {y.shape}")
    print(f"Class counts: " +
          ", ".join(f"{CLASSES4[k]}={int((y == k).sum())}" for k in range(4)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out_rows = []
    for name in ("logreg", "rforest", "hgb"):
        oof_direct = np.zeros((len(y), 4))
        oof_cascade = np.zeros((len(y), 4))
        for fold, (tr, te) in enumerate(skf.split(X_df, y)):
            pre = make_preprocessor(X_df)
            pre.fit(X_df.iloc[tr])
            X_tr = np.asarray(pre.transform(X_df.iloc[tr]), dtype=float)
            X_te = np.asarray(pre.transform(X_df.iloc[te]), dtype=float)
            oof_direct[te]  = fit_direct(name, X_tr, y[tr], X_te)
            oof_cascade[te] = fit_cascade(name, X_tr, y[tr], X_te)
        m_d = metric_pack(y, oof_direct)
        m_c = metric_pack(y, oof_cascade)
        print(f"\n--- {name} (5-fold OOF on n={len(y)}) ---")
        print(f"  Direct  macro AUC = {m_d['macro_auc']:.4f}  "
              f"PR-AUC = {m_d['macro_prauc']:.4f}  acc = {m_d['accuracy']:.4f}")
        print(f"  Cascade macro AUC = {m_c['macro_auc']:.4f}  "
              f"PR-AUC = {m_c['macro_prauc']:.4f}  acc = {m_c['accuracy']:.4f}")
        print(f"  delta   macro AUC = {m_c['macro_auc'] - m_d['macro_auc']:+.4f}  "
              f"PR-AUC = {m_c['macro_prauc'] - m_d['macro_prauc']:+.4f}  "
              f"acc = {m_c['accuracy'] - m_d['accuracy']:+.4f}")
        out_rows.append({"split": "cv", "model": name,
                          "arch": "direct", **flatten(m_d)})
        out_rows.append({"split": "cv", "model": name,
                          "arch": "cascade", **flatten(m_c)})
    return out_rows


def flatten(m: dict) -> dict:
    out = {"logloss": m["logloss"], "accuracy": m["accuracy"],
           "macro_auc": m["macro_auc"], "macro_prauc": m["macro_prauc"]}
    for k, c in enumerate(CLASSES4):
        short = c.split()[0].lower()
        out[f"prevalence_{short}"] = m["prevalence"][c]
        out[f"auc_{short}"] = m["per_auc"][c]
        out[f"prauc_{short}"] = m["per_prauc"][c]
        out[f"brier_{short}"] = m["per_brier"][c]
        out[f"bss_{short}"] = m["per_bss"][c]
    return out


# ---------- Temporal holdout -----------------------------------------

def run_temporal() -> list[dict]:
    print("\n" + "=" * 78)
    print("Temporal holdout: direct 4-class vs cascade")
    print("=" * 78)
    X_df, y, arrival, _ = load_features_and_y()
    dates = pd.to_datetime(arrival)
    last_day = dates.dt.date.max()
    is_test = (dates.dt.date == last_day).to_numpy()
    is_train = ~is_test
    print(f"Train: {int(is_train.sum())}  Test: {int(is_test.sum())}  "
          f"(test = {last_day})")
    pre = make_preprocessor(X_df)
    pre.fit(X_df.iloc[is_train])
    X_tr = np.asarray(pre.transform(X_df.iloc[is_train]), dtype=float)
    X_te = np.asarray(pre.transform(X_df.iloc[is_test]), dtype=float)
    y_tr = y[is_train]
    y_te = y[is_test]

    out_rows = []
    for name in ("logreg", "rforest", "hgb"):
        p_direct  = fit_direct(name, X_tr, y_tr, X_te)
        p_cascade = fit_cascade(name, X_tr, y_tr, X_te)
        m_d = metric_pack(y_te, p_direct)
        m_c = metric_pack(y_te, p_cascade)
        print(f"\n--- {name} (holdout n={len(y_te)}) ---")
        print(f"  Direct  macro AUC = {m_d['macro_auc']:.4f}  "
              f"PR-AUC = {m_d['macro_prauc']:.4f}  acc = {m_d['accuracy']:.4f}")
        print(f"  Cascade macro AUC = {m_c['macro_auc']:.4f}  "
              f"PR-AUC = {m_c['macro_prauc']:.4f}  acc = {m_c['accuracy']:.4f}")
        print(f"  delta   macro AUC = {m_c['macro_auc'] - m_d['macro_auc']:+.4f}  "
              f"PR-AUC = {m_c['macro_prauc'] - m_d['macro_prauc']:+.4f}  "
              f"acc = {m_c['accuracy'] - m_d['accuracy']:+.4f}")
        out_rows.append({"split": "temporal", "model": name,
                          "arch": "direct", **flatten(m_d)})
        out_rows.append({"split": "temporal", "model": name,
                          "arch": "cascade", **flatten(m_c)})
    return out_rows


# ---------- Report ----------------------------------------------------

def write_report(all_rows: list[dict]) -> None:
    df = pd.DataFrame(all_rows)
    df.to_csv(DERIVED / "task1_cascade_vs_direct_summary.csv", index=False)

    lines = ["# Task 1 — direct 4-class vs cascade (tier-1 binary + tier-2 multiclass)\n"]
    lines.append("Two architectures, identical CV folds + identical "
                  "temporal split. For each {logreg, rforest, hgb}, the "
                  "fold preprocessor + train indices are shared, so the "
                  "comparison is paired.\n")

    for split in ("cv", "temporal"):
        lines.append(f"## {split.upper()} (n={'261 5-fold OOF' if split=='cv' else '74 last-day holdout'})\n")
        sub = df[df["split"] == split]
        lines.append("| Model | Arch | log-loss | accuracy | macro ROC-AUC | macro PR-AUC | AUC None | AUC Kraken | AUC Triton | AUC Coral |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append("| " + " | ".join([
                r["model"], r["arch"],
                f"{r['logloss']:.4f}", f"{r['accuracy']:.4f}",
                f"{r['macro_auc']:.4f}", f"{r['macro_prauc']:.4f}",
                f"{r['auc_none']:.3f}",
                f"{r['auc_kraken']:.3f}",
                f"{r['auc_triton']:.3f}",
                f"{r['auc_coral']:.3f}",
            ]) + " |")
        lines.append("")

        # Delta table — cascade minus direct
        lines.append(f"### {split.upper()} — cascade minus direct (Δ)\n")
        lines.append("| Model | Δ macro AUC | Δ macro PR-AUC | Δ accuracy | Δ AUC None | Δ AUC Kraken | Δ AUC Triton | Δ AUC Coral |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for model in ("logreg", "rforest", "hgb"):
            d = sub[(sub["model"] == model) & (sub["arch"] == "direct")].iloc[0]
            c = sub[(sub["model"] == model) & (sub["arch"] == "cascade")].iloc[0]
            row = [model]
            for col in ("macro_auc", "macro_prauc", "accuracy",
                         "auc_none", "auc_kraken", "auc_triton", "auc_coral"):
                row.append(f"{c[col] - d[col]:+.4f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    out_path = DERIVED / "task1_cascade_vs_direct_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {out_path}")
    print(f"Summary CSV:    {DERIVED / 'task1_cascade_vs_direct_summary.csv'}")


def main() -> None:
    rows = []
    rows.extend(run_cv())
    rows.extend(run_temporal())
    write_report(rows)


if __name__ == "__main__":
    main()
