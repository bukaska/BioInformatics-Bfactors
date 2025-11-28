#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--kmin", type=int, default=2)
ap.add_argument("--kmax", type=int, default=12)
ap.add_argument("--best_k", type=int, default=5)   # choose 5 if you want Ward/KMeans to match
args = ap.parse_args()


feat_path = Path("data/meta/clustering/structure_features.csv")
out_dir   = Path("data/meta/clustering"); out_dir.mkdir(parents=True, exist_ok=True)

feats = pd.read_csv(feat_path)
feature_cols = ["mean_norm","std_norm","frac_z_gt1","frac_z_lt_neg1","q25","q50","q75","count"]
X = feats[feature_cols].fillna(0.0).values
X = StandardScaler().fit_transform(X)

rows = []
for k in range(args.kmin, args.kmax + 1):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)  # labels length == X rows
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

pd.DataFrame(rows).to_csv("data/meta/clustering/silhouette_by_k.csv", index=False)

# (optional) save labels for a chosen k, e.g. best k = 12
best_k = args.best_k
km = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit(X)
feats.assign(cluster=km.labels_)[["pdb_id","cluster"]].to_csv(
    f"data/meta/clustering/kmeans_labels_structure_k{best_k}.csv", index=False
)


print("Saved:", out_dir / "silhouette_by_k.csv")
