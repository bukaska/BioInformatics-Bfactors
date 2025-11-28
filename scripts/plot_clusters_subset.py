#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

base     = Path("data/meta/clustering")
feats    = pd.read_csv(base / "structure_features.csv")
meta     = pd.read_csv("data/meta/structures.csv")  # to get symbol & subfamily
feats = feats.merge(meta[["pdb_id","symbol","subfamily"]], on="pdb_id", how="left")

feature_cols = ["mean_norm","std_norm","frac_z_gt1","frac_z_lt_neg1","q25","q50","q75","count"]
X = StandardScaler().fit_transform(feats[feature_cols].fillna(0.0).values)
Z = PCA(n_components=2, random_state=42).fit_transform(X)  # or UMAP if installed
feats["PC1"], feats["PC2"] = Z[:,0], Z[:,1]

plot_dir = Path("data/meta/plots"); plot_dir.mkdir(parents=True, exist_ok=True)

def plot_for_k(k):
    labels = pd.read_csv(base / f"kmeans_labels_structure_k{k}.csv")
    df = feats.merge(labels, on="pdb_id", how="left")
    # 1) colored by cluster
    plt.figure(figsize=(8,6))
    sc = plt.scatter(df["PC1"], df["PC2"], c=df["cluster"], cmap="tab10", s=12, alpha=0.8)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"Structures (PCA) colored by cluster (k={k})")
    plt.colorbar(sc, label="cluster")
    plt.tight_layout()
    plt.savefig(plot_dir / f"pca_by_cluster_k{k}.png", dpi=300); plt.close()

    # 2) colored by subfamily (≈16 colors — still OK)
    subfam_codes = {s:i for i,s in enumerate(sorted(df["subfamily"].dropna().unique()))}
    cvals = df["subfamily"].map(subfam_codes)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(df["PC1"], df["PC2"], c=cvals, cmap="tab20", s=12, alpha=0.8)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"Structures (PCA) colored by subfamily (k={k})")
    plt.colorbar(sc, ticks=range(len(subfam_codes)), label="subfamily")
    plt.tight_layout()
    plt.savefig(plot_dir / f"pca_by_subfamily_k{k}.png", dpi=300); plt.close()

    # 3) focused subsets – EXACTLY what your prof asked
    subsets = {
        "PPARs": ["PPARA","PPARD","PPARG"],
        "RXRs":  ["RXRA","RXRB","RXRG"],
        "TRs":   ["THRA","THRB"],
        "RARs":  ["RARA","RARB","RARG"],
        "ERs":   ["ESR1","ESR2"],
    }
    for name, symbols in subsets.items():
        df["focus"] = np.where(df["symbol"].isin(symbols), df["symbol"], "others")
        # keep only points for symbols of interest + light gray for others
        plt.figure(figsize=(8,6))
        # plot others very light
        m_others = (df["focus"]=="others")
        plt.scatter(df.loc[m_others,"PC1"], df.loc[m_others,"PC2"], s=6, alpha=0.15, color="#cccccc")
        # plot focus symbols with distinct colors
        focus_df = df.loc[~m_others].copy()
        sym_codes = {s:i for i,s in enumerate(symbols)}
        cvals = focus_df["symbol"].map(sym_codes)
        sc = plt.scatter(focus_df["PC1"], focus_df["PC2"], c=cvals, cmap="tab10", s=18, alpha=0.9)
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title(f"{name} only (colored), others in gray (k={k})")
        handles, _ = sc.legend_elements()
        plt.legend(handles, symbols, title=name, loc="best", frameon=True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"pca_subset_{name}_k{k}.png", dpi=300); plt.close()

for k in (5,8,12):  # adjust to your chosen candidates
    plot_for_k(k)

print("Saved PCA cluster/subset plots.")
