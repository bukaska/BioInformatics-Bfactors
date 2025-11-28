#!/usr/bin/env python3
"""
QC and visualization for nuclear receptor structure dataset.
Saves plots to PNGs instead of showing interactive windows.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# --- make matplotlib non-interactive & fast ---
import matplotlib
matplotlib.use("Agg")           # render to files, not GUI
import matplotlib.pyplot as plt

#ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Paths
meta_path = Path("data/meta/structures.csv")
bf_path   = Path("data/processed/ca_bfactors.csv")
plot_dir  = Path("data/meta/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# Load
meta = pd.read_csv(meta_path)
bf   = pd.read_csv(bf_path)

proteins = pd.read_csv("proteins.csv")

# Which proteins have at least one PDB in meta
covered = set(meta["uniprot"])
proteins["has_structure"] = proteins["uniprot"].isin(covered)

missing_proteins = proteins[~proteins["has_structure"]]
missing_proteins.to_csv("data/meta/missing_receptors.csv", index=False)


print("\nProteins with no structures:")
print(missing_proteins)

# Some people named the column "b_factor" in older versions, normalize:
if "bfactor" not in bf.columns and "b_factor" in bf.columns:
    bf = bf.rename(columns={"b_factor": "bfactor"})

print("Loaded:")
print(f"  Structures meta: {len(meta)} rows")
print(f"  B-factor rows:   {len(bf)}")
print(f"  Unique PDBs in B-factors: {bf['pdb_id'].nunique()}")

# coverage
counts = meta.groupby("symbol")["pdb_id"].nunique().sort_values(ascending=False)
print("\nStructures per protein (top 15):")
print(counts.head(15))

missing = counts[counts == 0]
if not missing.empty:
    print("\nProteins with no PDB structures:")
    print(missing.index.tolist())

#resolution / R-free QC
# bad_res = meta[meta["resolution"] > 3.0]
# if len(bad_res) > 0:
#     print(f"\n{len(bad_res)} structures have resolution worse than 3.0 Å:")
#     print(bad_res[["pdb_id", "symbol", "resolution"]].to_string(index=False))
# else:
#     print("\nAll structures have acceptable resolution (≤3.5 Å)")

# if meta["rfree"].isna().any():
#     print(f"{meta['rfree'].isna().sum()} structures missing R-free.")

# plot - Resolution distribution per subfamily
plt.figure(figsize=(10,6))
meta.boxplot(column="resolution", by="subfamily", rot=45)
plt.ylabel("Resolution (Å)")
plt.title("Distribution of structure resolution per NR subfamily")
plt.suptitle("")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_by_subfamily.png", dpi=300)
plt.close()

#plot: Average B-factor per receptor
avg_b = bf.groupby("symbol")["bfactor"].mean().sort_values()
plt.figure(figsize=(10,12))
avg_b.plot(kind="barh")
plt.xlabel("Average Cα B-factor")
plt.title("Average flexibility per receptor (all structures)")
plt.tight_layout()
plt.savefig(plot_dir / "avg_bfactor_per_receptor.png", dpi=300)
plt.close()

#plot: Std B-factor per receptor
avg_b = bf.groupby("symbol")["bfactor"].std().sort_values()
plt.figure(figsize=(10,12))
avg_b.plot(kind="barh")
plt.xlabel("Std Cα B-factor")
plt.title("Std flexibility per receptor (all structures)")
plt.tight_layout()
plt.savefig(plot_dir / "std_bfactor_per_receptor.png", dpi=300)
plt.close()

#plot resolution vs mean B-factor per structure
merged = (bf.groupby("pdb_id")["bfactor"].mean()
            .reset_index()
            .merge(meta, on="pdb_id", how="left"))

plt.figure(figsize=(7,6))
plt.scatter(merged["resolution"], merged["bfactor"], alpha=0.6)
plt.xlabel("Resolution (Å)")
plt.ylabel("Mean Cα B-factor")

a, b = np.polyfit(merged["resolution"], merged["bfactor"], 1)
x = np.array([merged["resolution"].min(), merged["resolution"].max()])
plt.plot(x, a*x + b, color="red", linestyle="--", label=f"y={a:.1f}x+{b:.1f}")
plt.legend()

df = pd.read_csv("data/meta/structure_summary.csv")
df.dropna(subset=["mean", "resolution"], inplace=True)
corr = df["resolution"].corr(df["mean"])
print(f"\nCorrelation between resolution and mean B-factor: {corr:.3f}")


plt.title("Resolution vs mean B-factor per structure")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_vs_mean_bfactor.png", dpi=300)
plt.close()

#resolution vs std b factor
merged = (bf.groupby("pdb_id")["bfactor"].std()
            .reset_index()
            .merge(meta, on="pdb_id", how="left"))

plt.figure(figsize=(10,6))
plt.scatter(merged["resolution"], merged["bfactor"], alpha=0.6)
plt.xlabel("Resolution (Å)")
plt.ylabel("Std Cα B-factor")
plt.title("Resolution vs std B-factor per structure")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_vs_std_bfactor.png", dpi=300)
plt.close()

#summary table
summary = (bf.groupby("pdb_id")["bfactor"]
             .agg(mean="mean", std="std", count="count")
             .merge(meta, on="pdb_id", how="left"))
summary_out = Path("data/meta/structure_summary.csv")
summary.to_csv(summary_out, index=False)

print("\nSummary saved:", summary_out)
print("Plots saved to:", plot_dir)
print("QC complete (non-interactive).")

# --- Per-receptor correlation ---
corr_per_receptor = (
    summary.dropna(subset=["mean", "resolution"])
           .groupby("symbol")
           .apply(lambda df: df["resolution"].corr(df["mean"]))
           .reset_index(name="correlation")
)
# --- Per-subfamily correlation ---
corr_per_receptor.to_csv("data/meta/correlation_per_receptor.csv", index=False)


corr_per_subfam = (
    summary.dropna(subset=["mean", "resolution"])
           .groupby("subfamily")
           .apply(lambda df: df["resolution"].corr(df["mean"]))
           .reset_index(name="correlation")
)

corr_per_subfam.to_csv("data/meta/correlation_per_subfamily.csv", index=False)

# --- Mean ± SD B-factors and all resolutions per receptor ---
b_summary = (
    summary.groupby("symbol")
           .agg(mean_bfactor=("mean", "mean"),
                std_bfactor=("mean", "std"),
                n_structures=("pdb_id", "count"))
           .reset_index()
)

res_lookup = (summary.groupby("symbol")["resolution"]
                     .apply(lambda x: ", ".join(sorted(map(lambda v: f"{v:.2f}", x.unique()))))
                     .reset_index(name="resolutions"))

b_summary = b_summary.merge(res_lookup, on="symbol", how="left")
b_summary.to_csv("data/meta/bfactor_resolution_summary_per_receptor.csv", index=False)

print("\nSaved: bfactor_resolution_summary_per_receptor.csv")


# Plot: Mean ± SD B-factors per receptor
b_summary = pd.read_csv("data/meta/bfactor_resolution_summary_per_receptor.csv")
b_summary = b_summary.sort_values("mean_bfactor", ascending=True)

plt.figure(figsize=(10, 12))
plt.barh(b_summary["symbol"],
         b_summary["mean_bfactor"],
         xerr=b_summary["std_bfactor"],
         color="skyblue",
         ecolor="gray",
         capsize=3)

plt.xlabel("Mean Cα B-factor (Å²)")
plt.ylabel("Receptor (symbol)")
plt.title("Mean ± SD B-factors per receptor")
plt.tight_layout()
plt.savefig("data/meta/plots/mean_sd_bfactor_per_receptor.png", dpi=300)
plt.close()


# ============================
# normalization by resolution bins
# ============================

# attach resolution to every residue B-factor row
bf_with_res = bf.merge(meta[["pdb_id","resolution"]], on="pdb_id", how="left")
bf_with_res = bf_with_res.dropna(subset=["resolution", "bfactor"])

# define resolution bins
bin_edges  = [1.0, 2.0, 2.5, 3.0, 3.5, 4.5, 10.0]
bin_labels = ["1.0–2.0", "2.0–2.5", "2.5–3.0", "3.0–3.5", "3.5–4.5", "4.5–10.0"]
bf_with_res["res_bin"] = pd.cut(bf_with_res["resolution"], bins=bin_edges,
                                labels=bin_labels, include_lowest=True, right=True)

# bin stats
bin_stats = (bf_with_res.groupby("res_bin")["bfactor"]
             .agg(mean_bin="mean", std_bin="std").reset_index())

# if std_bin is 0 or NaN (very small bin), set a floor value
EPS = 1e-6
bin_stats["std_bin"] = bin_stats["std_bin"].fillna(0.0).clip(lower=EPS)

# merge stats back and compute z-score per residue: z = (B - mean_bin) / std_bin
bf_norm = bf_with_res.merge(bin_stats, on="res_bin", how="left")
bf_norm["z_bfactor"] = (bf_norm["bfactor"] - bf_norm["mean_bin"]) / bf_norm["std_bin"]

# save normalized residue-level table
norm_out = Path("data/processed/ca_bfactors_normalized.csv")
bf_norm.to_csv(norm_out, index=False)
print(f"\nWrote z-normalized residue table → {norm_out}")

# average z across its residues
summary_norm = (bf_norm.groupby("pdb_id")["z_bfactor"]
                .agg(mean_norm="mean", std_norm="std", count="count")
                .reset_index()
                .merge(meta, on="pdb_id", how="left"))
summary_norm_out = Path("data/meta/structure_summary_normalized.csv")
summary_norm.to_csv(summary_norm_out, index=False)
print(f"Wrote normalized per-structure summary → {summary_norm_out}")

# correlation check: before vs after normalization
raw_summary = pd.read_csv("data/meta/structure_summary.csv").dropna(subset=["mean","resolution"])
print("Correlation BEFORE normalization (resolution vs mean B):",
      f"{raw_summary['resolution'].corr(raw_summary['mean']):.3f}")

summary_norm_clean = summary_norm.dropna(subset=["mean_norm","resolution"])
print("Correlation AFTER normalization (resolution vs mean_norm):",
      f"{summary_norm_clean['resolution'].corr(summary_norm_clean['mean_norm']):.3f}")



#plot: Resolution vs mean_norm 
plt.figure(figsize=(7,6))
plt.scatter(summary_norm_clean["resolution"], summary_norm_clean["mean_norm"], alpha=0.6)
plt.xlabel("Resolution (Å)")
plt.ylabel("Mean normalized Cα B-factor (z)")
plt.title("Resolution vs mean normalized B-factor per structure")
#best-fit line
a2, b2 = np.polyfit(summary_norm_clean["resolution"], summary_norm_clean["mean_norm"], 1)
xx = np.array([summary_norm_clean["resolution"].min(), summary_norm_clean["resolution"].max()])
plt.plot(xx, a2*xx + b2, linestyle="--", label=f"y={a2:.2f}x+{b2:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "resolution_vs_mean_norm_bfactor.png", dpi=300)
plt.close()

# Plot: Mean normalized B-factor per receptor
per_receptor_norm = (summary_norm_clean.groupby("symbol")["mean_norm"]
                     .mean().sort_values())
plt.figure(figsize=(10,12))
per_receptor_norm.plot(kind="barh")
plt.xlabel("Mean normalized Cα B-factor (z)")
plt.title("Normalized flexibility per receptor (mean of structure z-means)")
plt.tight_layout()
plt.savefig(plot_dir / "mean_norm_bfactor_per_receptor.png", dpi=300)
plt.close()


# Per-receptor correlation
def _safe_corr(g):
    g = g.dropna(subset=["resolution","mean_norm"])
    return g["resolution"].corr(g["mean_norm"]) if len(g) > 2 else np.nan


corr_per_receptor_norm = (summary_norm.groupby("symbol", group_keys=False)
                          .apply(_safe_corr)
                          .reset_index(name="correlation_norm"))

corr_per_receptor_norm.to_csv("data/meta/correlation_per_receptor_NORMALIZED.csv", index=False)
print("\nSaved: correlation_per_receptor_NORMALIZED.csv")

# Per-subfamily correlation
corr_per_subfam_norm = (summary_norm.groupby("subfamily", group_keys=False)
                        .apply(_safe_corr)
                        .reset_index(name="correlation_norm"))
corr_per_subfam_norm.to_csv("data/meta/correlation_per_subfamily_NORMALIZED.csv", index=False)
print("Saved: correlation_per_subfamily_NORMALIZED.csv")
