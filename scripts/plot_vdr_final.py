#!/usr/bin/env python3
"""
Final summary plots for VDR LBD flexibility.

Generates:
1) vdr_helix_boxplot.png
2) vdr_helix_vs_loop_violin.png
3) vdr_structure_mean_flexibility.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
BASE = Path(".")
BF_PATH = BASE / "data/processed/ca_bfactors_normalized.csv"
HELIX_PATH = BASE / "data/processed/vdr_consensus_helices.csv"
SSE_PER_RES_PATH = BASE / "data/processed/vdr_sse_consensus_per_residue.csv"

OUT_DIR = BASE / "data/meta/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECEPTOR_SYMBOL = "VDR"
LBD_START = 118
LBD_END = 427

# IMPORTANT: real column names in your file
COL_PDB = "pdb_id"
COL_SYMBOL = "symbol"
COL_RESSEQ = "resnum"      # residue number in B-factor file
COL_Z = "z_bfactor"        # normalized B-factor

# ----------------- LOAD DATA -----------------

print("Loading normalized B-factors...")
bf = pd.read_csv(BF_PATH)

for c in [COL_PDB, COL_SYMBOL, COL_RESSEQ, COL_Z]:
    if c not in bf.columns:
        raise ValueError(f"Column '{c}' not found in {BF_PATH}. "
                         f"Available: {list(bf.columns)}")

bf_vdr = bf[
    (bf[COL_SYMBOL] == RECEPTOR_SYMBOL) &
    (bf[COL_RESSEQ] >= LBD_START) &
    (bf[COL_RESSEQ] <= LBD_END)
].copy()

bf_vdr[COL_Z] = pd.to_numeric(bf_vdr[COL_Z], errors="coerce")
bf_vdr = bf_vdr.dropna(subset=[COL_Z])

print(f"  VDR rows: {len(bf_vdr)}")

# Load helices (H1–H14)
helix_df = pd.read_csv(HELIX_PATH)
required = {"label", "start", "end"}
if not required.issubset(helix_df.columns):
    raise ValueError("vdr_consensus_helices.csv missing required columns.")

# Load consensus SSE per residue and harmonise column names
sse_res = pd.read_csv(SSE_PER_RES_PATH)

# resseq column is named 'resseq' there – map to COL_RESSEQ ('resnum')
if "resseq" in sse_res.columns and COL_RESSEQ not in sse_res.columns:
    sse_res = sse_res.rename(columns={"resseq": COL_RESSEQ})

# find which column holds the consensus SSE
cols = set(sse_res.columns)
if "consensus_sse" in cols:
    col_cons = "consensus_sse"
elif "consensus" in cols:
    col_cons = "consensus"
else:
    raise ValueError(
        f"No consensus SSE column found in {SSE_PER_RES_PATH}. "
        f"Available: {list(sse_res.columns)}"
    )

# keep only residue number + consensus SSE, and standardise the name
sse_res = sse_res[[COL_RESSEQ, col_cons]].rename(columns={col_cons: "consensus_sse"})

# ----------------- MAP HELICES TO RESIDUES -----------------

res_to_helix = {}
for _, r in helix_df.iterrows():
    for res in range(int(r.start), int(r.end) + 1):
        res_to_helix[res] = r.label

bf_vdr["helix_label"] = bf_vdr[COL_RESSEQ].map(res_to_helix)

# merge SSE consensus onto B-factor table
bf_vdr = bf_vdr.merge(sse_res, on=COL_RESSEQ, how="left")

# classify residues as helix/loop based on consensus SSE
bf_vdr["region_type"] = np.where(bf_vdr["consensus_sse"] == "H", "helix", "loop")

# ----------------- PLOT 1: HELIX BOXPLOT -----------------

print("Plot 1: per-helix boxplot...")

helix_data = bf_vdr.dropna(subset=["helix_label"]).copy()

def sort_key(label: str) -> int:
    # convert 'H10' -> 10 for sorting
    return int(label.replace("H", ""))

helix_labels = sorted(helix_data["helix_label"].unique(), key=sort_key)

data = [helix_data.loc[helix_data["helix_label"] == h, COL_Z].values
        for h in helix_labels]

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=helix_labels, showfliers=False)
plt.axhline(0, linestyle="--", linewidth=1)
plt.ylabel("Normalized Cα B-factor (z-score)")
plt.xlabel("Helix")
plt.title("VDR LBD flexibility per helix")
plt.tight_layout()

out1 = OUT_DIR / "vdr_helix_boxplot.png"
plt.savefig(out1, dpi=300)
plt.close()
print(f"Saved {out1}")

# ----------------- PLOT 2: HELIX VS LOOP -----------------

print("Plot 2: helix vs loop violin...")

groups = ["helix", "loop"]
vdata = [bf_vdr.loc[bf_vdr["region_type"] == g, COL_Z].values for g in groups]

plt.figure(figsize=(6, 6))
plt.violinplot(vdata, showmeans=True, showmedians=True)
plt.xticks([1, 2], ["Helix residues", "Loop residues"])
plt.axhline(0, linestyle="--", linewidth=1)
plt.ylabel("Normalized Cα B-factor")
plt.title("VDR LBD: helix vs loop flexibility")
plt.tight_layout()

out2 = OUT_DIR / "vdr_helix_vs_loop_violin.png"
plt.savefig(out2, dpi=300)
plt.close()
print(f"Saved {out2}")

# ----------------- PLOT 3: STRUCTURE MEAN FLEXIBILITY -----------------

print("Plot 3: per-structure mean flexibility...")

struct_means = (
    bf_vdr.groupby(COL_PDB)[COL_Z]
    .mean()
    .reset_index()
    .rename(columns={COL_Z: "mean_z"})
)

plt.figure(figsize=(8, 5))
plt.hist(struct_means["mean_z"], bins=10, edgecolor="black")
plt.axvline(struct_means["mean_z"].mean(), linestyle="--", color="black")
plt.xlabel("Mean normalized Cα B-factor (z-score)")
plt.ylabel("Number of structures")
plt.title("Distribution of VDR structure flexibility")
plt.tight_layout()

out3 = OUT_DIR / "vdr_structure_mean_flexibility.png"
plt.savefig(out3, dpi=300)
plt.close()
print(f"Saved {out3}")

print("Done.")
