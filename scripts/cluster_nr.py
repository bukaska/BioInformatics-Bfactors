#!/usr/bin/env python3
"""
Cluster normalized flexibility profiles of nuclear receptor structures (or receptors).

Inputs (produced by qc_visualize.py):
  - data/meta/structure_summary_normalized.csv
  - data/processed/ca_bfactors_normalized.csv

Outputs:
  - data/meta/clustering/structure_features.csv              (structure-level features)
  - data/meta/clustering/receptor_features.csv               (receptor-level features)
  - data/meta/clustering/kmeans_labels_{level}_k{K}.csv      (cluster labels)
  - data/meta/clustering/{level}_cluster_summary.csv         (per-cluster feature means)
  - data/meta/plots/{level}_clusters_pca2d.png               (2D PCA scatter)
  - data/meta/plots/{level}_silhouette_by_k.png              (k vs silhouette)

Usage:
  python scripts/cluster_nr.py --level structure
  python scripts/cluster_nr.py --level receptor
  python scripts/cluster_nr.py --level structure --min_structures 3
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DATA_SUMMARY = Path("data/meta/structure_summary_normalized.csv")
DATA_RESID   = Path("data/processed/ca_bfactors_normalized.csv")
OUT_DIR_META = Path("data/meta/clustering")
OUT_DIR_PLOT = Path("data/meta/plots")

FEATURE_COLUMNS = ["mean_norm","std_norm","frac_z_gt1","frac_z_lt_neg1","q25","q50","q75","count"]

def build_structure_features():
    """Make a feature row per PDB structure using normalized residues."""
    # load per-residue normalized data
    usecols = ["pdb_id","z_bfactor","symbol","subfamily"]
    resid = pd.read_csv(DATA_RESID, usecols=usecols+["resolution"])
    # basic safety
    resid = resid.dropna(subset=["pdb_id","z_bfactor"])

    # fractions and quantiles per PDB
    def summarize(group):
        z = group["z_bfactor"].to_numpy()
        if z.size == 0:
            return pd.Series({
                "mean_norm": np.nan,
                "std_norm": np.nan,
                "frac_z_gt1": np.nan,
                "frac_z_lt_neg1": np.nan,
                "q25": np.nan,
                "q50": np.nan,
                "q75": np.nan,
                "count": 0
            })
        return pd.Series({
            "mean_norm": float(np.mean(z)),
            "std_norm": float(np.std(z, ddof=1)) if z.size > 1 else 0.0,
            "frac_z_gt1": float((z > 1.0).mean()),
            "frac_z_lt_neg1": float((z < -1.0).mean()),
            "q25": float(np.percentile(z, 25)),
            "q50": float(np.percentile(z, 50)),
            "q75": float(np.percentile(z, 75)),
            "count": int(z.size)
        })

    # compute features
    feats = resid.groupby("pdb_id").apply(summarize).reset_index()
    # add symbol/subfamily (from any row of that PDB)
    ann = resid.groupby("pdb_id")[["symbol","subfamily"]].agg(lambda s: s.iloc[0]).reset_index()
    feats = feats.merge(ann, on="pdb_id", how="left")

    OUT_DIR_META.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_DIR_META / "structure_features.csv", index=False)
    return feats

def build_receptor_features(struct_feats, min_structures=2):
    """Aggregate structure features per receptor symbol."""
    # keep receptors with enough structures
    counts = struct_feats.groupby("symbol")["pdb_id"].count().rename("n_structures")
    by_symbol = struct_feats.merge(counts, on="symbol")
    by_symbol = by_symbol[by_symbol["n_structures"] >= min_structures]

    agg = (by_symbol.groupby(["symbol","subfamily"])
           [FEATURE_COLUMNS]
           .mean()
           .reset_index())
    agg = agg.merge(counts.reset_index(), on="symbol", how="left")

    agg.to_csv(OUT_DIR_META / "receptor_features.csv", index=False)
    return agg

def choose_k_and_cluster(X, k_range=(2,8), random_state=42):
    """Try multiple k, return best model by silhouette score."""
    sil_scores = []
    best_k, best_model, best_score, best_labels = None, None, -1, None

    for k in range(k_range[0], k_range[1]+1):
        try:
            model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
            labels = model.fit_predict(X)
            # silhouette needs at least 2 clusters and no singleton behavior
            score = silhouette_score(X, labels)
            sil_scores.append((k, score))
            if score > best_score:
                best_k, best_model, best_score, best_labels = k, model, score, labels
        except Exception:
            # can fail if X is too small for chosen k
            continue

    return best_k, best_model, best_score, best_labels, sil_scores

def plot_silhouette(sil_scores, level):
    if not sil_scores:
        return
    ks, scores = zip(*sil_scores)
    plt.figure(figsize=(6,4))
    plt.plot(ks, scores, marker="o")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Silhouette score")
    plt.title(f"Silhouette vs k ({level})")
    plt.tight_layout()
    OUT_DIR_PLOT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR_PLOT / f"{level}_silhouette_by_k.png", dpi=300)
    plt.close()

def pca_scatter(X, labels, meta_ids, level):
    """2D PCA projection colored by cluster label."""
    if X.shape[0] < 2:
        return
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(Z[:,0], Z[:,1], c=labels, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{level.capitalize()} clusters (PCA 2D)")
    plt.tight_layout()
    OUT_DIR_PLOT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR_PLOT / f"{level}_clusters_pca2d.png", dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["structure","receptor"], default="structure",
                    help="Cluster per PDB structure or per receptor symbol")
    ap.add_argument("--min_structures", type=int, default=2,
                    help="Minimum PDBs required for a receptor when level=receptor")
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=8)
    args = ap.parse_args()

    # build features
    struct_feats = build_structure_features()
    if args.level == "structure":
        feats = struct_feats.copy()
        item_ids = feats["pdb_id"]
        meta_cols = ["pdb_id","symbol","subfamily"]
    else:
        feats = build_receptor_features(struct_feats, min_structures=args.min_structures)
        item_ids = feats["symbol"]
        meta_cols = ["symbol","subfamily","n_structures"]

    # select features & drop rows with NaNs
    X = feats[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna()
    keep_idx = X.index
    ids_kept = item_ids.loc[keep_idx].reset_index(drop=True)
    meta_kept = feats.loc[keep_idx, meta_cols].reset_index(drop=True)
    X = X.to_numpy()

    if X.shape[0] < 4:
        print("Not enough rows to cluster. Try level=structure or lower min_structures.")
        return

    # scale features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # choose k and cluster
    best_k, model, best_score, labels, sil_scores = choose_k_and_cluster(
        Xs, k_range=(args.k_min, args.k_max), random_state=42
    )

    plot_silhouette(sil_scores, args.level)

    if best_k is None:
        print("Could not find a valid k for clustering.")
        return

    print(f"Chosen k={best_k} (silhouette={best_score:.3f}) for {args.level}-level clustering")

    # save labels
    out = meta_kept.copy()
    out["cluster"] = labels
    OUT_DIR_META.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR_META / f"kmeans_labels_{args.level}_k{best_k}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved cluster labels → {out_path}")

    # PCA scatter
    pca_scatter(Xs, labels, ids_kept, args.level)

    # per-cluster feature means (to interpret clusters)
    feat_means = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    feat_means["cluster"] = labels
    cluster_summary = feat_means.groupby("cluster")[FEATURE_COLUMNS].mean().reset_index()
    # add counts and some metadata distribution
    cluster_counts = out.groupby("cluster").size().rename("n_items").reset_index()
    cluster_summary = cluster_summary.merge(cluster_counts, on="cluster", how="left")
    summary_path = OUT_DIR_META / f"{args.level}_cluster_summary.csv"
    cluster_summary.to_csv(summary_path, index=False)
    print(f"Saved cluster summary → {summary_path}")

    #simple enrichment-like view: counts by subfamily in each cluster
    subfam_counts = out.groupby(["cluster","subfamily"]).size().rename("count").reset_index()
    subfam_counts.to_csv(OUT_DIR_META / f"{args.level}_cluster_subfamily_counts.csv", index=False)

if __name__ == "__main__":
    main()
