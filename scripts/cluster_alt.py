#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, fcluster

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- paths ----------
feat_path = Path("data/meta/clustering/structure_features.csv")
out_dir   = Path("data/meta/clustering"); out_dir.mkdir(parents=True, exist_ok=True)
plot_dir  = Path("data/meta/plots");      plot_dir.mkdir(parents=True, exist_ok=True)

# ---------- data ----------
feats = pd.read_csv(feat_path)
feature_cols = ["mean_norm","std_norm","frac_z_gt1","frac_z_lt_neg1","q25","q50","q75","count"]
X = feats[feature_cols].fillna(0.0).to_numpy()
X = StandardScaler().fit_transform(X)

# helper: safe silhouette
def safe_silhouette(X, labels):
    labels = np.asarray(labels)
    if labels.ndim != 1: return np.nan
    # if noise exists (DBSCAN), exclude it from BOTH X and y
    mask = labels != -1
    if mask.sum() <= 1: return np.nan
    if np.unique(labels[mask]).size < 2: return np.nan
    return float(silhouette_score(X[mask], labels[mask]))

# ---------- DBSCAN ----------
dbscan_rows = []
for eps in (0.8, 1.0, 1.2, 1.4):
    model  = DBSCAN(eps=eps, min_samples=10)
    labels = model.fit_predict(X)
    n_noise = int((labels == -1).sum())
    n_clust = int(len(set(labels)) - (1 if -1 in labels else 0))
    sil = safe_silhouette(X, labels)
    dbscan_rows.append({"eps": eps, "clusters": n_clust, "noise": n_noise, "silhouette": sil})
    # save labels
    feats.assign(cluster=labels)[["pdb_id","cluster"]].to_csv(out_dir / f"dbscan_eps{eps:.1f}.csv", index=False)
    print(f"DBSCAN eps={eps:.1f}: clusters={n_clust}, noise={n_noise}, silhouette={sil:.3f}")

pd.DataFrame(dbscan_rows).to_csv(out_dir / "dbscan_summary.csv", index=False)

# ---------- Hierarchical (Ward) ----------
Z = linkage(X, method="ward")
for k in (5, 8, 12):
    labels = fcluster(Z, t=k, criterion="maxclust")
    sil = safe_silhouette(X, labels)
    feats.assign(cluster=labels)[["pdb_id","cluster"]].to_csv(out_dir / f"hierarchical_k{k}.csv", index=False)
    print(f"Ward k={k}: silhouette={sil:.3f}")

# ---------- PCA scatter for the last 'labels' ----------
pca = PCA(n_components=2, random_state=42)
Z2  = pca.fit_transform(X)
plt.figure(figsize=(8,6))
sc = plt.scatter(Z2[:,0], Z2[:,1], c=labels, cmap="tab20", s=9, alpha=0.8)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Hierarchical clustering (Ward) – PCA projection")
plt.colorbar(sc, label="cluster")
plt.tight_layout()
plt.savefig(plot_dir / "hierarchical_pca.png", dpi=300)
plt.close()
