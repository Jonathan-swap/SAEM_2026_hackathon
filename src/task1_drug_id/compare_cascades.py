"""Compare direct 4-class against three cascade variants.

All four architectures predict the full 4-class outcome
(None / Kraken / Triton / Coral). They differ in how that
prediction is constructed.

Architectures evaluated:
  D = Direct 4-class
        one 4-class classifier on all 261 patients.
  A = Cascade A: tier-1 + tier-2-multiclass
        P(drug)            from binary classifier on all 261
        P(K | drug),
        P(T | drug),
        P(C | drug)        from 3-class classifier on drug-positive
  B = Cascade B: tier-1 + Kraken-vs-rest + prevalence
        P(drug)            from binary on all 261
        P(K | drug)        from binary Kraken-vs-rest on drug-positive
        T vs C split       fixed at training-set prevalence
                            (random guess based on prior)
  C = Cascade C: tier-1 + Kraken-vs-rest + Triton-vs-Coral
        P(drug)            from binary on all 261
        P(K | drug)        from binary Kraken-vs-rest on drug-positive
        P(T | non-K)       from binary Triton-vs-Coral on non-Kraken
                            drug-positive

Each architecture's 4-class probabilities sum to 1 by construction.

Same CV folds + same temporal split, three model families
(logreg, rforest, hgb). Per-fold pre-processing is shared across
architectures so the comparison is paired.

Outputs:
  derived/task1_cascade_combinations_summary.csv
  derived/task1_cascade_combinations_report.md
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

CLASSES4 = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]
TEXT_COL = "triage_brief_note"
ARCHS = ["direct", "casc_A_tier12", "casc_B_K_prev", "casc_C_K_TC"]
ARCH_LABEL = {
    "direct":         "Direct 4-class",
    "casc_A_tier12":  "A: tier-1 + tier-2-multiclass",
    "casc_B_K_prev":  "B: tier-1 + K-vs-rest + prevalence",
    "casc_C_K_TC":    "C: tier-1 + K-vs-rest + Triton-vs-Coral",
}


# ---------- Data + preprocessor --------------------------------------

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


def model_factory(name: str):
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


# ---------- Architecture predictors ----------------------------------

def fit_direct(name, X_tr, y_tr, X_te) -> np.ndarray:
    mdl = model_factory(name)
    mdl.fit(X_tr, y_tr)
    p_full = np.zeros((len(X_te), 4))
    p_seen = mdl.predict_proba(X_te)
    for j, cls in enumerate(mdl.classes_):
        p_full[:, cls] = p_seen[:, j]
    return p_full


def _fit_tier1(name, X_tr, y_tr_4cls, X_te) -> np.ndarray:
    """Returns P(drug-positive) per test row."""
    y_tr_bin = (y_tr_4cls != 0).astype(int)
    mdl = model_factory(name)
    mdl.fit(X_tr, y_tr_bin)
    return mdl.predict_proba(X_te)[:, 1]


def _fit_tier2_multi(name, X_tr, y_tr_4cls, X_te) -> np.ndarray:
    """Returns 3-vector (P(K|drug), P(T|drug), P(C|drug)) per test row."""
    mask = (y_tr_4cls != 0)
    mdl = model_factory(name)
    mdl.fit(X_tr[mask], y_tr_4cls[mask])
    p = mdl.predict_proba(X_te)
    out = np.zeros((len(X_te), 3))
    for j, cls in enumerate(mdl.classes_):
        out[:, cls - 1] = p[:, j]   # cls in {1,2,3} -> idx 0..2
    return out


def _fit_kraken_vs_rest(name, X_tr, y_tr_4cls, X_te) -> np.ndarray:
    """Returns P(Kraken | drug-positive) per test row."""
    mask = (y_tr_4cls != 0)
    y = (y_tr_4cls[mask] == 1).astype(int)
    mdl = model_factory(name)
    mdl.fit(X_tr[mask], y)
    return mdl.predict_proba(X_te)[:, 1]


def _fit_triton_vs_coral(name, X_tr, y_tr_4cls, X_te
                           ) -> tuple[np.ndarray, bool]:
    """Returns (P(Triton | non-Kraken-drug-positive) per test row,
    fitted_flag). Falls back to prevalence if too few non-K samples."""
    mask = (y_tr_4cls == 2) | (y_tr_4cls == 3)
    if mask.sum() < 10:
        # not enough samples — use prevalence
        if mask.sum() == 0:
            p_triton_prev = 0.5
        else:
            p_triton_prev = float((y_tr_4cls[mask] == 2).mean())
        return np.full(len(X_te), p_triton_prev), False
    y = (y_tr_4cls[mask] == 2).astype(int)
    mdl = model_factory(name)
    mdl.fit(X_tr[mask], y)
    return mdl.predict_proba(X_te)[:, 1], True


def fit_cascade_A(name, X_tr, y_tr, X_te) -> np.ndarray:
    """tier-1 + tier-2 multiclass."""
    p_drug = _fit_tier1(name, X_tr, y_tr, X_te)
    p_drug_cond = _fit_tier2_multi(name, X_tr, y_tr, X_te)
    p_full = np.zeros((len(X_te), 4))
    p_full[:, 0] = 1 - p_drug
    p_full[:, 1] = p_drug * p_drug_cond[:, 0]
    p_full[:, 2] = p_drug * p_drug_cond[:, 1]
    p_full[:, 3] = p_drug * p_drug_cond[:, 2]
    return p_full / p_full.sum(axis=1, keepdims=True)


def fit_cascade_B(name, X_tr, y_tr, X_te) -> np.ndarray:
    """tier-1 + Kraken-vs-rest + prevalence split for Triton/Coral."""
    p_drug = _fit_tier1(name, X_tr, y_tr, X_te)
    p_kraken = _fit_kraken_vs_rest(name, X_tr, y_tr, X_te)
    # T-vs-C split from training-set prevalence of Triton among
    # non-Kraken drug-positive
    mask = (y_tr == 2) | (y_tr == 3)
    p_triton_prev = float((y_tr[mask] == 2).mean()) if mask.sum() else 0.5

    p_full = np.zeros((len(X_te), 4))
    p_full[:, 0] = 1 - p_drug
    p_full[:, 1] = p_drug * p_kraken
    p_full[:, 2] = p_drug * (1 - p_kraken) * p_triton_prev
    p_full[:, 3] = p_drug * (1 - p_kraken) * (1 - p_triton_prev)
    return p_full / p_full.sum(axis=1, keepdims=True)


def fit_cascade_C(name, X_tr, y_tr, X_te) -> np.ndarray:
    """tier-1 + Kraken-vs-rest + Triton-vs-Coral classifier."""
    p_drug = _fit_tier1(name, X_tr, y_tr, X_te)
    p_kraken = _fit_kraken_vs_rest(name, X_tr, y_tr, X_te)
    p_triton, _ = _fit_triton_vs_coral(name, X_tr, y_tr, X_te)
    p_full = np.zeros((len(X_te), 4))
    p_full[:, 0] = 1 - p_drug
    p_full[:, 1] = p_drug * p_kraken
    p_full[:, 2] = p_drug * (1 - p_kraken) * p_triton
    p_full[:, 3] = p_drug * (1 - p_kraken) * (1 - p_triton)
    return p_full / p_full.sum(axis=1, keepdims=True)


ARCH_FN = {
    "direct":         fit_direct,
    "casc_A_tier12":  fit_cascade_A,
    "casc_B_K_prev":  fit_cascade_B,
    "casc_C_K_TC":    fit_cascade_C,
}


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
    per_auc, per_prauc, per_brier, per_bss, prev = {}, {}, {}, {}, {}
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


# ---------- Evaluation runners ---------------------------------------

def run_cv() -> list[dict]:
    print("=" * 78)
    print("CV — direct vs three cascade variants")
    print("=" * 78)
    X_df, y, _, _ = load_features_and_y()
    print(f"X: {X_df.shape}   y: {y.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name in ("logreg", "rforest", "hgb"):
        oof = {arch: np.zeros((len(y), 4)) for arch in ARCHS}
        for fold, (tr, te) in enumerate(skf.split(X_df, y)):
            pre = make_preprocessor(X_df)
            pre.fit(X_df.iloc[tr])
            X_tr = np.asarray(pre.transform(X_df.iloc[tr]), dtype=float)
            X_te = np.asarray(pre.transform(X_df.iloc[te]), dtype=float)
            for arch in ARCHS:
                oof[arch][te] = ARCH_FN[arch](name, X_tr, y[tr], X_te)
        print(f"\n--- {name} (5-fold OOF, n={len(y)}) ---")
        for arch in ARCHS:
            m = metric_pack(y, oof[arch])
            print(f"  {ARCH_LABEL[arch]:42s}  "
                  f"AUC = {m['macro_auc']:.4f}  "
                  f"PR-AUC = {m['macro_prauc']:.4f}  "
                  f"acc = {m['accuracy']:.4f}")
            rows.append({"split": "cv", "model": name, "arch": arch,
                          **flatten(m)})
    return rows


def run_temporal() -> list[dict]:
    print("\n" + "=" * 78)
    print("Temporal holdout — direct vs three cascade variants")
    print("=" * 78)
    X_df, y, arrival, _ = load_features_and_y()
    dates = pd.to_datetime(arrival)
    last_day = dates.dt.date.max()
    is_test = (dates.dt.date == last_day).to_numpy()
    is_train = ~is_test
    print(f"Train n={int(is_train.sum())}, Test n={int(is_test.sum())} "
          f"(test = {last_day})")

    pre = make_preprocessor(X_df)
    pre.fit(X_df.iloc[is_train])
    X_tr = np.asarray(pre.transform(X_df.iloc[is_train]), dtype=float)
    X_te = np.asarray(pre.transform(X_df.iloc[is_test]), dtype=float)
    y_tr = y[is_train]
    y_te = y[is_test]

    rows = []
    for name in ("logreg", "rforest", "hgb"):
        print(f"\n--- {name} (holdout n={len(y_te)}) ---")
        for arch in ARCHS:
            p = ARCH_FN[arch](name, X_tr, y_tr, X_te)
            m = metric_pack(y_te, p)
            print(f"  {ARCH_LABEL[arch]:42s}  "
                  f"AUC = {m['macro_auc']:.4f}  "
                  f"PR-AUC = {m['macro_prauc']:.4f}  "
                  f"acc = {m['accuracy']:.4f}")
            rows.append({"split": "temporal", "model": name, "arch": arch,
                          **flatten(m)})
    return rows


# ---------- Report ----------------------------------------------------

def write_report(all_rows: list[dict]) -> None:
    df = pd.DataFrame(all_rows)
    df.to_csv(DERIVED / "task1_cascade_combinations_summary.csv",
              index=False)

    lines = ["# Task 1 — Direct vs three cascade variants\n"]
    lines.append("Same CV folds + same temporal split for all four "
                  "architectures. For each {logreg, rforest, hgb} the "
                  "per-fold preprocessor and train indices are shared, "
                  "so the comparison is paired.\n")
    lines.append("Architectures:\n")
    lines.append("- **D**: Direct 4-class — one classifier predicts None/K/T/C.")
    lines.append("- **A**: tier-1 binary + tier-2 multiclass — `P(drug)` × `P(K/T/C | drug)`.")
    lines.append("- **B**: tier-1 binary + Kraken-vs-rest + prevalence — `P(drug)` × `P(K | drug)`; non-K mass split T/C by training prevalence (no T-vs-C model).")
    lines.append("- **C**: tier-1 binary + Kraken-vs-rest + Triton-vs-Coral — full hierarchical, three binary models.\n")

    for split in ("cv", "temporal"):
        n_label = "261 5-fold OOF" if split == "cv" else "74 last-day holdout"
        lines.append(f"## {split.upper()} (n={n_label})\n")
        sub = df[df["split"] == split]
        lines.append("| Model | Arch | log-loss | accuracy | macro AUC | macro PR-AUC | AUC None | AUC Kraken | AUC Triton | AUC Coral |")
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

        # Delta vs direct (per model)
        lines.append(f"### {split.upper()} — cascades minus direct\n")
        lines.append("| Model | Arch | Δ macro AUC | Δ macro PR-AUC | Δ acc | Δ AUC K | Δ AUC T | Δ AUC C |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for model in ("logreg", "rforest", "hgb"):
            d = sub[(sub["model"] == model) & (sub["arch"] == "direct")].iloc[0]
            for arch in ("casc_A_tier12", "casc_B_K_prev", "casc_C_K_TC"):
                c = sub[(sub["model"] == model) & (sub["arch"] == arch)].iloc[0]
                lines.append("| " + " | ".join([
                    model, arch,
                    f"{c['macro_auc'] - d['macro_auc']:+.4f}",
                    f"{c['macro_prauc'] - d['macro_prauc']:+.4f}",
                    f"{c['accuracy'] - d['accuracy']:+.4f}",
                    f"{c['auc_kraken'] - d['auc_kraken']:+.3f}",
                    f"{c['auc_triton'] - d['auc_triton']:+.3f}",
                    f"{c['auc_coral'] - d['auc_coral']:+.3f}",
                ]) + " |")
        lines.append("")

    out = DERIVED / "task1_cascade_combinations_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out}")
    print(f"CSV:    {DERIVED / 'task1_cascade_combinations_summary.csv'}")


def main() -> None:
    rows = []
    rows.extend(run_cv())
    rows.extend(run_temporal())
    write_report(rows)


if __name__ == "__main__":
    main()
