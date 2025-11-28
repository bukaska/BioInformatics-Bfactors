#!/usr/bin/env python3
"""
Step 3 (+4): VDR LBD per-residue violin plot with SSE track.

Inputs
------
- data/processed/ca_bfactors_normalized.csv   (per-residue z_bfactor)
- data/meta/structures.csv                    (to get symbol = VDR)
- data/processed/vdr_dssp_coverage.csv        (pdb_id + chosen chain)
- data/processed/vdr_consensus_helices.csv    (H1..Hn boundaries)

Output
------
- data/meta/plots/vdr_lbd_violin_with_sse.png

Each x-position = residue number 118..427.
Each violin = distribution of z-normalized Cα B-factors across all VDR
structures at that position.
The bottom track shows helices (H1..Hn) as colored bars and loops as grey bars.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_vdr_bfactors(bf_path, meta_path, coverage_path,
                      start_res=118, end_res=427):
    """
    Load normalized Cα B-factors and filter to:
      - VDR structures only
      - chains selected in DSSP coverage
      - residue window [start_res, end_res]

    Returns a dataframe with columns:
      pdb_id, chain_id, resseq, z_bfactor
    """
    bf = pd.read_csv(bf_path)
    meta = pd.read_csv(meta_path)
    cov = pd.read_csv(coverage_path)

    # --- normalize column names ---
    # Original normalized file has 'resnum', not 'resseq':
    if "resseq" not in bf.columns and "resnum" in bf.columns:
        bf = bf.rename(columns={"resnum": "resseq"})

    # Some earlier scripts might have 'chain' instead of 'chain_id'
    if "chain" in bf.columns and "chain_id" not in bf.columns:
        bf = bf.rename(columns={"chain": "chain_id"})
    # --------------------------------

    # VDR PDB ids from meta
    vdr_ids = meta.loc[meta["symbol"] == "VDR", "pdb_id"].unique()
    bf = bf[bf["pdb_id"].isin(vdr_ids)]

    # keep only chains that were selected in DSSP (same pdb_id + chain_id)
    keep_pairs = cov[["pdb_id", "chain_id"]].drop_duplicates()
    bf = bf.merge(keep_pairs, on=["pdb_id", "chain_id"], how="inner")

    # residue window
    bf = bf[(bf["resseq"] >= start_res) & (bf["resseq"] <= end_res)]

    # drop NA z-scores if any
    bf = bf.dropna(subset=["z_bfactor"])

    return bf



def build_violin_data(bf_df, start_res=118, end_res=427, min_points=1):
    """
    Turn per-residue z_bfactor table into a list-of-arrays for violinplot.

    Only keeps positions that have at least `min_points` values
    (default = 1, you can set to 3 if you want more robust stats).

    Returns:
      positions: np.array of residue numbers (only non-empty)
      data:      list of 1D arrays, length = len(positions)
      means:     np.array of per-residue means
      medians:   np.array of per-residue medians
    """
    grouped = bf_df.groupby("resseq")["z_bfactor"].apply(list).to_dict()

    positions_all = np.arange(start_res, end_res + 1)
    data_all = [np.array(grouped.get(res, []), dtype=float) for res in positions_all]

    positions = []
    data = []
    means = []
    medians = []

    for res, vals in zip(positions_all, data_all):
        if len(vals) >= min_points:
            positions.append(res)
            data.append(vals)
            means.append(vals.mean())
            medians.append(np.median(vals))

    return np.array(positions), data, np.array(means), np.array(medians)


def add_sse_track(ax, helices_df, start_res=118, end_res=427):
    """
    Draw SSE annotation on the given axis:

    - helices: colored rectangles with labels H1, H2, ...
    - loops:   grey rectangles in between helices
    """
    # Draw loop (non-helix) regions as grey background rectangles
    current = start_res
    helices_sorted = helices_df.sort_values("start_res")

    for _, row in helices_sorted.iterrows():
        h_start = int(row["start_res"])
        h_end = int(row["end_res"])

        # loop before this helix
        if h_start > current:
            ax.add_patch(
                Rectangle(
                    (current - 0.5, 0),
                    width=h_start - current,
                    height=1,
                    facecolor="lightgrey",
                    edgecolor="none",
                    alpha=0.6,
                )
            )
        # helix itself
        ax.add_patch(
            Rectangle(
                (h_start - 0.5, 0),
                width=h_end - h_start + 1,
                height=1,
                facecolor="#ffb366",  # orange-ish
                edgecolor="none",
                alpha=0.9,
            )
        )
        # label in the middle
        mid = (h_start + h_end) / 2.0
        ax.text(
            mid,
            0.5,
            row["helix_id"],
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

        current = h_end + 1

    # tail loop after last helix
    if current <= end_res:
        ax.add_patch(
            Rectangle(
                (current - 0.5, 0),
                width=end_res - current + 1,
                height=1,
                facecolor="lightgrey",
                edgecolor="none",
                alpha=0.6,
            )
        )

    ax.set_xlim(start_res - 0.5, end_res + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Residue number (VDR LBD)")
    ax.set_title("Consensus secondary structure (helices and loops)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=118, help="LBD start residue")
    ap.add_argument("--end", type=int, default=427, help="LBD end residue")
    args = ap.parse_args()

    base = Path(".")

    bf_path       = base / "data/processed/ca_bfactors_normalized.csv"
    meta_path     = base / "data/meta/structures.csv"
    coverage_path = base / "data/processed/vdr_dssp_coverage.csv"
    helices_path  = base / "data/processed/vdr_consensus_helices.csv"
    plot_dir      = base / "data/meta/plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load data ----------
    print("Loading normalized B-factors...")
    bf_vdr = load_vdr_bfactors(
        bf_path, meta_path, coverage_path, start_res=args.start, end_res=args.end
    )
    print(f"  B-factor rows after filtering: {len(bf_vdr)}")

    print("Building violin data...")
    positions, data, means, medians = build_violin_data(
        bf_vdr, start_res=args.start, end_res=args.end
    )
    print(f"  Residues covered: {len(positions)}")

    print("Loading consensus helices...")
    helices_df = pd.read_csv(helices_path)

    # ---------- make figure ----------
    fig, (ax_violin, ax_sse) = plt.subplots(
        2,
        1,
        figsize=(18, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    # --- main violin plot ---
    print("Drawing violin plot...")
    parts = ax_violin.violinplot(
        data,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # style violins
    for body in parts["bodies"]:
        body.set_facecolor("#b0dfff")
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    # overlay median (black line) and mean (dark blue dots)
    ax_violin.plot(
        positions,
        medians,
        color="black",
        linewidth=1,
        label="median z-score",
    )
    ax_violin.plot(
        positions,
        means,
        "o",
        markersize=2,
        color="navy",
        label="mean z-score",
    )

    # reference lines
    ax_violin.axhline(0, color="k", linestyle="--", linewidth=1, label="z = 0")
    ax_violin.axhline(1, color="grey", linestyle=":", linewidth=0.8, label="z = 1")
    ax_violin.axhline(-1, color="grey", linestyle=":", linewidth=0.8, label="z = -1")

    ax_violin.set_ylabel("Normalized Cα B-factor (z-score)")
    ax_violin.set_title("VDR LBD per-residue flexibility (118–427)")
    ax_violin.legend(loc="upper right", fontsize=8)

    # --- SSE annotation track ---
    print("Adding SSE track...")
    add_sse_track(ax_sse, helices_df, start_res=args.start, end_res=args.end)

    fig.tight_layout()
    out_png = plot_dir / "vdr_lbd_violin_with_sse.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved plot → {out_png}")


if __name__ == "__main__":
    main()
