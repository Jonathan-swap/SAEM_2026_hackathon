"""Option 3 — track A: unsupervised clustering on triage and 4h feature
sets, with cross-tabulation against B+C consensus labels.

For each horizon:
  - One-hot / numeric-encode features
  - SimpleImputer + StandardScaler
  - KMeans(n_clusters=4)  (expect 3 drug toxidromes + non-festival)
  - sklearn HDBSCAN (unbiased cluster count)
  - PCA(2) for a quick visual scatter saved as PNG

Then compare each clustering to the B+C consensus label (majority of
the three agent labels) and the disposition label.

Writes:
  derived/clusters_triage.csv  (encounter_id, kmeans, hdbscan)
  derived/clusters_fourh.csv
  derived/cluster_pca_triage.png
  derived/cluster_pca_fourh.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"

LABEL_PATH = DERIVED / "derived_labels.csv"

TEXT_COLS_TO_DROP = {
    "triage_brief_note",  # raw text; not vectorized here
}
ID_COLS = {
    "encounter_id",
    "encounter_arrival_date",
    "encounter_disposition_label",
}


def prepare_X(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop ID/label/text columns; numerify everything else."""
    df = df.copy()
    drop = [c for c in df.columns
            if c in ID_COLS or c in TEXT_COLS_TO_DROP]
    df = df.drop(columns=drop)
    # One-hot encode object/string columns
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, dummy_na=True)
    # Cast booleans to float
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(float)
    return df, obj_cols


def cluster_one(name: str, features_path: Path, labels: pd.DataFrame) -> None:
    print(f"\n=== Clustering on {name} ({features_path.name}) ===")
    df = pd.read_csv(features_path)
    encounter_ids = df["encounter_id"].to_numpy()
    X_df, obj_cols = prepare_X(df)
    print(f"  Encoded features: {X_df.shape[1]} (after one-hot of "
          f"{len(obj_cols)} categoricals)")

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    X = pipe.fit_transform(X_df)

    # KMeans(4)
    km = KMeans(n_clusters=4, random_state=42, n_init=20)
    km_labels = km.fit_predict(X)

    # HDBSCAN
    hdb = HDBSCAN(min_cluster_size=15, min_samples=5)
    hdb_labels = hdb.fit_predict(X)
    print(f"  KMeans:  4 clusters, sizes = {pd.Series(km_labels).value_counts().to_dict()}")
    print(f"  HDBSCAN: discovered {len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)} "
          f"clusters + {(hdb_labels == -1).sum()} noise points")
    print(f"           sizes = {pd.Series(hdb_labels).value_counts().to_dict()}")

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, lbl, title in [(axes[0], km_labels, f"KMeans(4) — {name}"),
                            (axes[1], hdb_labels, f"HDBSCAN — {name}")]:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=lbl,
                              cmap="tab10", s=14, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    fig_path = DERIVED / f"cluster_pca_{name}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"  PCA scatter: {fig_path}")

    # Save cluster assignments
    out = pd.DataFrame({
        "encounter_id": encounter_ids,
        f"kmeans_{name}": km_labels,
        f"hdbscan_{name}": hdb_labels,
    })
    out_path = DERIVED / f"clusters_{name}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Assignments: {out_path}")

    # Cross-tab vs B+C consensus and disposition
    join = out.merge(labels, on="encounter_id", how="left")
    print(f"\n  KMeans({name}) vs majority_label:")
    print("  " + str(pd.crosstab(join[f"kmeans_{name}"],
                                  join["majority_label"]))
                .replace("\n", "\n  "))
    print(f"\n  KMeans({name}) vs disposition:")
    print("  " + str(pd.crosstab(join[f"kmeans_{name}"],
                                  join["encounter_disposition_label"]))
                .replace("\n", "\n  "))

    ari_km = adjusted_rand_score(join["majority_label"].astype(str),
                                  join[f"kmeans_{name}"])
    nmi_km = normalized_mutual_info_score(join["majority_label"].astype(str),
                                            join[f"kmeans_{name}"])
    print(f"\n  KMeans vs majority_label:  ARI = {ari_km:.3f}  "
          f"NMI = {nmi_km:.3f}")
    # HDBSCAN ARI
    ari_hdb = adjusted_rand_score(join["majority_label"].astype(str),
                                    join[f"hdbscan_{name}"])
    nmi_hdb = normalized_mutual_info_score(join["majority_label"].astype(str),
                                             join[f"hdbscan_{name}"])
    print(f"  HDBSCAN vs majority_label: ARI = {ari_hdb:.3f}  "
          f"NMI = {nmi_hdb:.3f}")


def main() -> None:
    labels = pd.read_csv(LABEL_PATH, keep_default_na=False, na_values=[""])
    # Bring disposition in via features_triage.csv (it's there)
    dispo = pd.read_csv(DERIVED / "features_triage.csv")[
        ["encounter_id", "encounter_disposition_label"]]
    labels = labels.merge(dispo, on="encounter_id", how="left")

    cluster_one("triage", DERIVED / "features_triage.csv", labels)
    cluster_one("fourh", DERIVED / "features_fourh.csv", labels)


if __name__ == "__main__":
    main()
