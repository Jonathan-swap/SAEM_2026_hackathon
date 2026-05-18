"""Task-aligned unsupervised clustering.

Two clustering runs, each mirroring the cohort and feature set of
one supervised model:

- **Task 1 clustering** (this branch, unsupervised mode):
  all 261 encounters, features_triage.csv (triage-horizon features
  only), n_clusters=**5**. The 5 clusters are the new outcome
  categories — no supervised truth label is asserted. The drug
  class (None / Kraken / Triton / Coral) is reported as an
  informational cross-tab but NOT used to evaluate the clusters
  (ARI/NMI are printed for context only; the algorithm is targeting
  5 clusters, the supervised label only has 4 values).
- **Task 2 clustering**: 157 drug-positive encounters
  (ground_truth_drug != 0), features_fourh.csv (4h horizon),
  n_clusters=3 (the 3 drugs). Mirrors the Task-2 baseline cohort
  filter in `src/task2_deterioration/train_baseline.py`.

For each run:
  - drop id / target columns; one-hot categoricals; impute median;
    standardize
  - KMeans (hard, n_clusters as above)
  - Gaussian Mixture Model (soft, same n_clusters) — picks up
    ellipsoidal cluster shapes the spherical KMeans misses; per-
    point class probabilities are more honest about overlap
  - PCA(2) scatter — one panel coloured by KMeans cluster, one by
    the ground-truth drug; centroids overlaid with full class-
    fraction annotations
  - cross-tabulate clusters against the manual ground-truth drug
    name AND (for Task 2) disposition; ARI + NMI vs ground truth
    for both KMeans and GMM

Writes:
  derived/clusters_task1.csv
  derived/clusters_task2.csv
  derived/cluster_pca_task1.png
  derived/cluster_pca_task2.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (adjusted_rand_score,
                              normalized_mutual_info_score)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

OUTCOMES_PATH = DERIVED / "outcomes.csv"

DRUG_CLASSES = ["None", "Kraken Candy", "Triton Tabs", "Coral Dust"]
DISPO_CLASSES = ["Discharge", "Floor", "ICU"]

# Columns that are IDs or targets — never feed to clustering.
DROP_NEVER = {
    "encounter_id",
    "encounter_arrival_date",
    "encounter_disposition_label",  # Task 2 target
    "ground_truth_drug",
    "ground_truth_drug_name",
}
DROP_TEXT = {"triage_brief_note"}  # free text — not vectorised here


# ---------- feature prep ----------------------------------------------

def prepare_X(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Strip id/target/text; one-hot categoricals; bool→float."""
    df = df.copy()
    drop = [c for c in df.columns if c in DROP_NEVER or c in DROP_TEXT]
    df = df.drop(columns=drop)

    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, dummy_na=True)
    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(float)
    return df, obj_cols


# ---------- one clustering pass --------------------------------------

def cluster_one(
    name: str,
    features_path: Path,
    ground_truth: pd.DataFrame,
    cohort_filter: callable | None,
    n_kmeans: int,
    truth_col: str = "ground_truth_drug_name",
    truth_classes: list[str] | None = None,
    extra_label_col: str | None = None,
    unsupervised_mode: bool = False,
    top_features_per_cluster: int = 12,
) -> dict:
    """One task-aligned clustering run.

    Args:
        name: short label, e.g. ``task1`` or ``task2``.
        features_path: csv to read features from.
        ground_truth: DataFrame with at least
            ``encounter_id, ground_truth_drug, ground_truth_drug_name``.
            Merged in so the cohort filter has access to drug labels.
        cohort_filter: optional callable receiving the joined DF and
            returning a boolean mask; used to drop None for Task 2.
        n_kmeans: number of KMeans clusters.
        truth_col: column in the joined DF used as the supervised
            truth label for plot annotations + ARI/NMI. Defaults to
            drug class. For Task 2 set this to
            ``encounter_disposition_label`` so the clusters are
            evaluated against the actual Task-2 outcome.
        truth_classes: optional ordered list of class names. Used to
            keep colours consistent across the cluster panel and the
            truth panel of the PCA scatter.
        extra_label_col: optional secondary column from features_path
            to cross-tab against.
        unsupervised_mode: if True, treat the clusters as the outcome
            and don't rely on truth_col for evaluation. ARI/NMI vs
            truth are still computed as informational only; the
            primary outputs are cluster sizes, BIC, GMM entropy, and
            **per-cluster centroid feature loadings** (top distinguishing
            features by absolute z-score in the standardised space).
        top_features_per_cluster: how many distinguishing features to
            print per cluster when in unsupervised_mode.
    """
    print(f"\n{'='*72}\nClustering — {name}\n{'='*72}")
    df = pd.read_csv(features_path)
    # Defensive: features files must never carry outcome columns.
    for c in ("encounter_disposition_label", "ground_truth_drug",
               "ground_truth_drug_name"):
        if c in df.columns:
            df = df.drop(columns=[c])
    df = df.merge(ground_truth, on="encounter_id", how="inner")
    n_before = len(df)
    if cohort_filter is not None:
        df = df[cohort_filter(df)].reset_index(drop=True)
    print(f"Cohort: {n_before} -> {len(df)} encounters "
          f"({features_path.name})")
    print(f"Truth label: {truth_col}")

    encounter_ids = df["encounter_id"].to_numpy()
    # truth may have NaN (drug name is NaN for None-class rows in
    # task1). Coerce NaN -> "None" so sklearn metrics survive.
    truth = df[truth_col].fillna("None").astype(str).to_numpy()
    if truth_classes is None:
        truth_classes = sorted(set(truth.tolist()))
    class_to_idx = {c: i for i, c in enumerate(truth_classes)}
    truth_idx = np.array([class_to_idx.get(t, 0) for t in truth])
    extra = (df[extra_label_col].astype(str).to_numpy()
             if extra_label_col and extra_label_col in df.columns
             else None)

    X_df, obj_cols = prepare_X(df)
    print(f"  Encoded features: {X_df.shape[1]} (after one-hot of "
          f"{len(obj_cols)} categorical cols)")

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    X = pipe.fit_transform(X_df)

    # ---- KMeans ----
    km = KMeans(n_clusters=n_kmeans, random_state=42, n_init=20)
    km_labels = km.fit_predict(X)
    km_sizes = pd.Series(km_labels).value_counts().sort_index().to_dict()
    print(f"  KMeans({n_kmeans}): sizes = {km_sizes}")

    # Per-point distance to every centroid (in the standardised
    # full-dim feature space — same space the kmeans was fit on).
    distances = km.transform(X)  # shape (n_samples, n_clusters)

    # ---- Gaussian Mixture (soft, n_components = n_kmeans) ----
    # Replaces HDBSCAN. Toxidrome features are ellipsoidal and
    # overlapping; density-based clustering produces all-noise on
    # this scale. GMM gives soft per-point posterior probabilities
    # that we can compare to KMeans' hard assignments.
    gmm = GaussianMixture(n_components=n_kmeans,
                            covariance_type="diag",
                            random_state=42, n_init=5,
                            reg_covar=1e-4)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)
    gmm_proba = gmm.predict_proba(X)
    gmm_bic = gmm.bic(X)
    gmm_entropy = float(
        -(gmm_proba * np.log(np.clip(gmm_proba, 1e-12, 1.0))).sum(axis=1).mean())
    gmm_sizes = pd.Series(gmm_labels).value_counts().sort_index().to_dict()
    print(f"  GMM({n_kmeans}, diag): sizes = {gmm_sizes}  "
          f"BIC={gmm_bic:.1f}  mean-entropy={gmm_entropy:.3f}")

    # ---- Full candidate-label distribution per cluster ----
    # No unique-assignment constraint: if two clusters are both
    # dominantly the same class (e.g. None at 40% prevalence covering
    # 2 of 4 Task-1 clusters), label them as such — that IS the data.
    # We list EVERY ground-truth class with its within-cluster
    # fraction (always summing to 1.0). Classes absent from a cluster
    # show 0%.
    all_classes = sorted(set(truth.tolist()))
    cluster_top3: dict[int, list[tuple[str, float]]] = {}
    cluster_dom: dict[int, tuple[str, float, int]] = {}
    for k in sorted(set(km_labels)):
        mask = km_labels == k
        vc = pd.Series(truth[mask]).value_counts(normalize=True)
        # All classes, ranked by fraction; absent classes get 0.
        full = [(cls, float(vc.get(cls, 0.0))) for cls in all_classes]
        full.sort(key=lambda t: t[1], reverse=True)
        cluster_top3[int(k)] = full
        top_cls, top_frac = full[0]
        cluster_dom[int(k)] = (top_cls, top_frac, int(mask.sum()))

    # ---- PCA scatter (2 panels: cluster vs truth) ----
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    # Project KMeans centroids into the same 2-D PCA space.
    centroids_2d = pca.transform(km.cluster_centers_)

    # Plot heading derived from the task name + truth column so the
    # title says exactly what's coloured on each panel.
    pretty_name = {"task1": "Task 1 (drug ID at triage, n=261)",
                    "task2": "Task 2 (4h deterioration)"}\
                    .get(name, name)
    pretty_truth = {"ground_truth_drug_name": "drug class",
                     "encounter_disposition_label": "disposition"}\
                     .get(truth_col, truth_col)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"{pretty_name}  |  KMeans({n_kmeans}) "
                  f"vs ground-truth {pretty_truth}",
                  fontsize=12, fontweight="bold", y=1.02)

    sc0 = axes[0].scatter(coords[:, 0], coords[:, 1],
                            c=km_labels, cmap="tab10", s=18, alpha=0.75)
    axes[0].set_title(f"KMeans cluster assignment "
                       f"(n_clusters = {n_kmeans})")
    axes[0].set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.colorbar(sc0, ax=axes[0], label="cluster")

    # Centroid markers + convex-hull boundary per cluster on panel 0.
    # Each centroid is annotated with its TOP-3 candidate ground-
    # truth labels and their cluster-internal fractions, so the
    # empirical class mix is visible (no spurious unique-assignment).
    def _fmt_top3(k: int) -> str:
        lines = [f"C{k}"]
        for cls, frac in cluster_top3[k]:
            lines.append(f"{cls} {frac * 100:.0f}%")
        return "\n".join(lines)

    for k in range(n_kmeans):
        cx, cy = centroids_2d[k]
        axes[0].scatter([cx], [cy], marker="X", s=260,
                          edgecolor="black", facecolor="white",
                          linewidth=2.0, zorder=5)
        axes[0].annotate(_fmt_top3(k),
                           (cx, cy), xytext=(0, 14),
                           textcoords="offset points",
                           fontsize=8, fontweight="bold",
                           ha="center", va="bottom",
                           bbox=dict(boxstyle="round,pad=0.25",
                                      facecolor="white",
                                      edgecolor="black",
                                      alpha=0.85),
                           zorder=6)
        member = coords[km_labels == k]
        if len(member) >= 3:
            try:
                hull = ConvexHull(member)
                poly = member[hull.vertices]
                poly = np.vstack([poly, poly[0]])  # close polygon
                axes[0].plot(poly[:, 0], poly[:, 1], "-",
                              color="black", linewidth=1.2, alpha=0.5)
                axes[0].fill(poly[:, 0], poly[:, 1],
                              alpha=0.05, color="black")
            except Exception:
                pass

    sc1 = axes[1].scatter(coords[:, 0], coords[:, 1],
                            c=truth_idx, cmap="tab10", s=18, alpha=0.75)
    axes[1].set_title(f"Ground-truth {pretty_truth} "
                       f"({len(truth_classes)} classes)")
    axes[1].set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[1].set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    cbar = plt.colorbar(sc1, ax=axes[1])
    cbar.set_label(truth_col)
    # Overlay KMeans centroids on the truth panel too, for direct
    # comparison. Same top-3 candidate annotation as panel 0.
    for k in range(n_kmeans):
        cx, cy = centroids_2d[k]
        axes[1].scatter([cx], [cy], marker="X", s=260,
                          edgecolor="black", facecolor="white",
                          linewidth=2.0, zorder=5)
        axes[1].annotate(_fmt_top3(k),
                           (cx, cy), xytext=(0, 14),
                           textcoords="offset points",
                           fontsize=8, fontweight="bold",
                           ha="center", va="bottom",
                           bbox=dict(boxstyle="round,pad=0.25",
                                      facecolor="white",
                                      edgecolor="black",
                                      alpha=0.85),
                           zorder=6)

    fig.tight_layout()
    fig_path = DERIVED / f"cluster_pca_{name}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"  PCA scatter -> {fig_path}")

    # ---- Per-cluster review CSVs (KMeans only) ----
    review_dir = DERIVED / f"cluster_review_{name}"
    review_dir.mkdir(exist_ok=True)
    extra_for_review = None
    if extra_label_col and extra_label_col in df.columns:
        extra_for_review = df[extra_label_col].astype(str).to_numpy()
    dist_cols = [f"dist_to_centroid_{k}" for k in range(n_kmeans)]
    for k in range(n_kmeans):
        mask = km_labels == k
        rev = pd.DataFrame({
            "encounter_id": encounter_ids[mask],
            "assigned_cluster": km_labels[mask],
            truth_col: truth[mask],
        })
        if extra_for_review is not None:
            rev[extra_label_col] = extra_for_review[mask]
        for j, col in enumerate(dist_cols):
            rev[col] = distances[mask, j]
        # Sort by ascending distance to OWN centroid (most-central first)
        rev = rev.sort_values(f"dist_to_centroid_{k}").reset_index(drop=True)
        path = review_dir / f"cluster_{k}.csv"
        rev.to_csv(path, index=False)
    print(f"  Per-cluster review CSVs -> {review_dir}\\cluster_<k>.csv "
          f"({n_kmeans} files)")

    # ---- Cluster <-> ground truth cross-tabs ----
    print(f"\n  KMeans cluster vs {truth_col}:")
    ct_km = pd.crosstab(pd.Series(km_labels, name="cluster"),
                         pd.Series(truth, name="truth"))
    print(_indent(ct_km.to_string()))

    print(f"\n  GMM cluster vs {truth_col}:")
    ct_gmm = pd.crosstab(pd.Series(gmm_labels, name="cluster"),
                          pd.Series(truth, name="truth"))
    print(_indent(ct_gmm.to_string()))

    if extra is not None:
        print(f"\n  KMeans cluster vs {extra_label_col}:")
        ct_extra = pd.crosstab(pd.Series(km_labels, name="cluster"),
                                pd.Series(extra, name=extra_label_col))
        print(_indent(ct_extra.to_string()))

    # ---- ARI / NMI vs ground truth ----
    ari_km = adjusted_rand_score(truth, km_labels)
    nmi_km = normalized_mutual_info_score(truth, km_labels)
    ari_gmm = adjusted_rand_score(truth, gmm_labels)
    nmi_gmm = normalized_mutual_info_score(truth, gmm_labels)
    tag = " (informational — unsupervised mode)" if unsupervised_mode else ""
    print(f"\n  KMeans  vs {truth_col}{tag}: "
          f"ARI = {ari_km:.3f}   NMI = {nmi_km:.3f}")
    print(f"  GMM     vs {truth_col}{tag}: "
          f"ARI = {ari_gmm:.3f}   NMI = {nmi_gmm:.3f}")

    # ---- Centroid feature loadings (unsupervised mode) -------------------
    # Each cluster's centroid in the standardised space IS a z-score
    # vector vs the global mean. Features with the largest absolute
    # values are what distinguishes that cluster — same idea as PCA
    # loadings but per-cluster.
    feature_names = X_df.columns.tolist()
    centroid_loadings: dict[int, list[tuple[str, float]]] = {}
    for k in range(n_kmeans):
        c = km.cluster_centers_[k]
        ranked = sorted(
            zip(feature_names, c),
            key=lambda kv: abs(kv[1]), reverse=True,
        )[:top_features_per_cluster]
        centroid_loadings[int(k)] = [(f, float(v)) for f, v in ranked]

    if unsupervised_mode:
        print(f"\n  Centroid feature loadings (z-score vs global "
              f"mean; top {top_features_per_cluster} per cluster):")
        for k in sorted(centroid_loadings.keys()):
            n_members = int((km_labels == k).sum())
            print(f"    cluster {k} (n={n_members:>3d}):")
            for f, v in centroid_loadings[k]:
                arrow = "++" if v > 0 else "--"
                print(f"      {arrow} {f:<40s} {v:+.3f}")

        # Persist the loadings for downstream consumers.
        loadings_rows = []
        for k, items in centroid_loadings.items():
            for rank, (feat, z) in enumerate(items, start=1):
                loadings_rows.append({
                    "cluster": k,
                    "rank": rank,
                    "feature": feat,
                    "z_score": z,
                })
        loadings_path = DERIVED / f"cluster_centroid_loadings_{name}.csv"
        pd.DataFrame(loadings_rows).to_csv(loadings_path, index=False)
        print(f"  Centroid loadings -> {loadings_path}")

    # ---- Print full candidate-label distribution per cluster ----
    # Every ground-truth class with its within-cluster fraction,
    # ranked. Top-1 drives macro-purity; the rest expose the mix.
    n_classes_print = len(all_classes)
    print(f"\n  KMeans cluster -> all {n_classes_print} class fractions "
          f"(ranked):")
    for k in sorted(cluster_top3.keys()):
        top = cluster_top3[k]
        n_members = (km_labels == k).sum()
        parts = [f"{cls} {frac*100:.0f}%" for cls, frac in top]
        print(f"    cluster {k} (n={n_members:>3d}): "
              + "  |  ".join(parts))
    macro_purity = float(np.mean([m[1] for m in cluster_dom.values()]))
    print(f"  Macro top-1 purity: {macro_purity:.3f}")

    # ---- Persist ----
    out = pd.DataFrame({
        "encounter_id": encounter_ids,
        f"kmeans_{name}": km_labels,
        f"gmm_{name}": gmm_labels,
        truth_col: truth,
    })
    # GMM posterior probabilities for soft membership
    for k in range(n_kmeans):
        out[f"gmm_proba_cluster_{k}"] = gmm_proba[:, k]
    out_path = DERIVED / f"clusters_{name}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Assignments -> {out_path}")

    return {
        "name": name,
        "n_encounters": len(df),
        "n_features": X_df.shape[1],
        "ari_km": ari_km,
        "nmi_km": nmi_km,
        "ari_gmm": ari_gmm,
        "nmi_gmm": nmi_gmm,
        "gmm_bic": gmm_bic,
        "gmm_entropy": gmm_entropy,
        "macro_purity": macro_purity,
        "kmeans_mapping": {int(k): v[0] for k, v in cluster_dom.items()},
    }


def _indent(s: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in s.splitlines())


# ---------- orchestration --------------------------------------------

def main() -> None:
    gt = pd.read_csv(OUTCOMES_PATH)
    print(f"Outcomes file: {len(gt)} rows")
    print(f"  drug classes:  "
          f"{gt['ground_truth_drug_name'].fillna('None').value_counts().to_dict()}")
    print(f"  disposition:   "
          f"{gt['encounter_disposition_label'].value_counts().to_dict()}")

    # Task 1 — all 261 encounters, triage features, **5 unsupervised
    # clusters**. The clusters themselves are the outcome categories
    # in this branch; ground_truth_drug_name (4 values) is reported
    # as an informational cross-tab but not used to evaluate the
    # clustering.
    res_t1 = cluster_one(
        name="task1",
        features_path=DERIVED / "features_triage.csv",
        ground_truth=gt,
        cohort_filter=None,
        n_kmeans=5,
        truth_col="ground_truth_drug_name",
        truth_classes=DRUG_CLASSES,
        extra_label_col=None,
        unsupervised_mode=True,
    )

    # Task 2 — drug-positive cohort, 4h features, 3 expected clusters.
    # Truth = DISPOSITION (the actual Task-2 outcome). Drug-class is
    # kept as a secondary cross-tab so we can still see Kraken/Triton/
    # Coral split inside each cluster.
    res_t2 = cluster_one(
        name="task2",
        features_path=DERIVED / "features_fourh.csv",
        ground_truth=gt,
        cohort_filter=lambda d: d["ground_truth_drug"] != 0,
        n_kmeans=3,
        truth_col="encounter_disposition_label",
        truth_classes=DISPO_CLASSES,
        extra_label_col="ground_truth_drug_name",
    )

    print(f"\n{'='*72}\nSummary\n{'='*72}")
    print(f"{'Run':<8s}  {'n':>4s}  {'feats':>6s}  "
          f"{'ARI_km':>7s}  {'NMI_km':>7s}  "
          f"{'ARI_gmm':>8s}  {'NMI_gmm':>8s}  "
          f"{'BIC':>9s}  {'H_gmm':>6s}  {'purity':>7s}")
    for r in (res_t1, res_t2):
        print(f"{r['name']:<8s}  {r['n_encounters']:>4d}  "
              f"{r['n_features']:>6d}  "
              f"{r['ari_km']:>7.3f}  {r['nmi_km']:>7.3f}  "
              f"{r['ari_gmm']:>8.3f}  {r['nmi_gmm']:>8.3f}  "
              f"{r['gmm_bic']:>9.1f}  {r['gmm_entropy']:>6.3f}  "
              f"{r['macro_purity']:>7.3f}")


if __name__ == "__main__":
    main()
