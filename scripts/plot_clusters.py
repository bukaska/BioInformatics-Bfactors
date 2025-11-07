#!/usr/bin/env python3
"""
Visualize clustering results from cluster_nr.py.
Saves plots to data/meta/plots/.
"""

from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Paths
feat_path = Path("data/meta/clustering/structure_features.csv")
plot_dir  = Path("data/meta/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# --- pick the latest labels file automatically ---
cands = sorted(glob("data/meta/clustering/kmeans_labels_structure_k*.csv"))
if not cands:
    raise FileNotFoundError(
        "No labels found. Run clustering first:\n"
        "  python scripts/cluster_nr.py --level structure"
    )
labels_path = Path(cands[-1])
print(f"Using labels file: {labels_path.name}")

# Load
feats  = pd.read_csv(feat_path)
labels = pd.read_csv(labels_path)

# If subfamily is missing in features, bring it from meta
if "subfamily" not in feats.columns:
    meta = pd.read_csv("data/meta/structures.csv")[["pdb_id", "subfamily"]].drop_duplicates()
    feats = feats.merge(meta, on="pdb_id", how="left")

# Merge labels
merged = feats.merge(labels[["pdb_id", "cluster"]], on="pdb_id", how="left")

# ---- PCA scatter ----
feature_cols = ["mean_norm","std_norm","frac_z_gt1","frac_z_lt_neg1","q25","q50","q75","count"]
X = merged[feature_cols].fillna(0)
X_scaled = StandardScaler().fit_transform(X)
Z = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sc = plt.scatter(Z[:,0], Z[:,1], c=merged["cluster"], cmap="tab10", alpha=0.7)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Structure clustering (PCA projection)")
plt.colorbar(sc, label="Cluster")
plt.tight_layout()
plt.savefig(plot_dir / "structure_clusters_pca2d_clean.png", dpi=300)
plt.close()
print("✅ PCA scatter saved: structure_clusters_pca2d_clean.png")

# ---- Heatmap of cluster feature means ----
summary = pd.read_csv("data/meta/clustering/structure_cluster_summary.csv")
cmat = summary.set_index("cluster")[feature_cols].sort_index()

plt.figure(figsize=(8,6))
plt.imshow(cmat, aspect="auto", cmap="viridis")
plt.colorbar(label="Feature mean")
plt.yticks(np.arange(len(cmat.index)), cmat.index)
plt.xticks(np.arange(len(feature_cols)), feature_cols, rotation=45, ha="right")
plt.title("Cluster feature profiles (heatmap)")
plt.tight_layout()
plt.savefig(plot_dir / "structure_clusters_feature_heatmap.png", dpi=300)
plt.close()
print("✅ Feature heatmap saved: structure_clusters_feature_heatmap.png")

# ---- Subfamily composition (stacked bars) ----
subfam_counts = merged.groupby(["cluster","subfamily"]).size().unstack(fill_value=0).sort_index()
ax = subfam_counts.plot(kind="bar", stacked=True, colormap="tab20", figsize=(10,6))
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of structures")
ax.set_title("Subfamily composition per cluster")
ax.legend(ncol=2, fontsize=8)
ax.figure.tight_layout()
ax.figure.savefig(plot_dir / "structure_clusters_subfamily_stacked.png", dpi=300)
plt.close(ax.figure)
print("✅ Subfamily composition plot saved: structure_clusters_subfamily_stacked.png")

# ---- Boxplots per feature by cluster ----
for col in feature_cols:
    data = [merged.loc[merged["cluster"] == c, col].dropna().to_numpy()
            for c in sorted(merged["cluster"].dropna().unique())]
    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=[str(c) for c in sorted(merged["cluster"].dropna().unique())])
    plt.xlabel("Cluster"); plt.ylabel(col)
    plt.title(f"{col} distribution by cluster")
    plt.tight_layout()
    plt.savefig(plot_dir / f"boxplot_{col}_by_cluster.png", dpi=300)
    plt.close()
print("✅ Feature boxplots saved for all features")
