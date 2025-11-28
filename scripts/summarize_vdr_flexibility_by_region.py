#!/usr/bin/env python3
"""
Step 6: Summarize VDR LBD flexibility per helix and loop.

Inputs:
    data/processed/ca_bfactors_normalized.csv
        columns: pdb_id, symbol, resseq, z_bfactor, ...
    data/processed/vdr_consensus_helices.csv
        columns: label (H1..), start, end

Outputs:
    data/processed/vdr_region_flexibility_summary.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


START_RES = 118
END_RES   = 427


def load_vdr_bfactors(base: Path) -> pd.DataFrame:
    """Load normalized Cα B-factors for VDR LBD only."""
    bf_path = base / "data/processed/ca_bfactors_normalized.csv"
    meta_path = base / "data/meta/structures.csv"

    bf   = pd.read_csv(bf_path)
    meta = pd.read_csv(meta_path)[["pdb_id", "symbol"]]

    # attach symbol; keep only VDR
    bf = bf.merge(meta, on="pdb_id", how="left")
    bf = bf[bf["symbol"] == "VDR"].copy()

    # keep only residues in LBD window
    if "resseq" not in bf.columns:
        raise RuntimeError("Expected 'resseq' column in ca_bfactors_normalized.csv")

    bf = bf[(bf["resseq"] >= START_RES) & (bf["resseq"] <= END_RES)].copy()

    # keep only residue-level z-scores
    if "z_bfactor" not in bf.columns:
        raise RuntimeError("Expected 'z_bfactor' column (normalized B-factor).")

    return bf


def build_loop_regions(helices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create loop regions between helices, e.g.:

    H1: 130-145
    H2: 150-160
    => loop L1: 146-149
    """
    helices_df = helices_df.sort_values("start").reset_index(drop=True)
    loops = []

    current = START_RES
    loop_id = 0

    for _, h in helices_df.iterrows():
        if current < h["start"]:
            loop_id += 1
            loops.append({
                "region_type": "loop",
                "label": f"L{loop_id}",
                "start": current,
                "end": h["start"] - 1
            })
        current = h["end"] + 1

    if current <= END_RES:
        loop_id += 1
        loops.append({
            "region_type": "loop",
            "label": f"L{loop_id}",
            "start": current,
            "end": END_RES
        })

    return pd.DataFrame(loops)


def summarize_region(bf: pd.DataFrame, start: int, end: int, label: str, region_type: str) -> dict:
    """Compute statistics for B-factors in [start, end]."""
    sub = bf[(bf["resseq"] >= start) & (bf["resseq"] <= end)]
    z = sub["z_bfactor"].values
    if len(z) == 0:
        return {
            "region_type": region_type,
            "label": label,
            "start": start,
            "end": end,
            "length": end - start + 1,
            "n_values": 0,
            "mean_z": np.nan,
            "median_z": np.nan,
            "q25_z": np.nan,
            "q75_z": np.nan
        }

    return {
        "region_type": region_type,
        "label": label,
        "start": start,
        "end": end,
        "length": end - start + 1,
        "n_values": len(z),
        "mean_z": float(np.mean(z)),
        "median_z": float(np.median(z)),
        "q25_z": float(np.percentile(z, 25)),
        "q75_z": float(np.percentile(z, 75))
    }


def main():
    base = Path(".")

    print("Loading VDR B-factors...")
    bf_vdr = load_vdr_bfactors(base)
    print(f"  rows: {len(bf_vdr)}")

    helices_path = base / "data/processed/vdr_consensus_helices.csv"
    helices_df   = pd.read_csv(helices_path)
    helices_df["region_type"] = "helix"

    # Build loop regions
    loops_df = build_loop_regions(helices_df)

    # Combine helices and loops into one list of regions
    regions = []

    for _, h in helices_df.iterrows():
        regions.append(summarize_region(bf_vdr, int(h["start"]), int(h["end"]),
                                        label=h["label"], region_type="helix"))

    for _, l in loops_df.iterrows():
        regions.append(summarize_region(bf_vdr, int(l["start"]), int(l["end"]),
                                        label=l["label"], region_type="loop"))

    summary_df = pd.DataFrame(regions).sort_values(
        ["region_type", "start"], ascending=[False, True]
    )

    out_path = base / "data/processed/vdr_region_flexibility_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"Saved regional flexibility summary → {out_path}")

    # Print a quick textual summary you can quote in the report
    print("\nHelix regions (mean_z, median_z):")
    for _, row in summary_df[summary_df["region_type"] == "helix"].iterrows():
        print(f"  {row['label']:>3}  {int(row['start']):3d}-{int(row['end']):3d} "
              f"(n={int(row['n_values'])}): "
              f"mean_z={row['mean_z']:.2f}, median_z={row['median_z']:.2f}")

    print("\nMost flexible helix (by median_z):")
    helix_sorted = summary_df[summary_df["region_type"] == "helix"].sort_values("median_z", ascending=False)
    if len(helix_sorted):
        top = helix_sorted.iloc[0]
        print(f"  {top['label']} (median_z={top['median_z']:.2f})")

    print("\nMost rigid helix (by median_z):")
    helix_sorted = summary_df[summary_df["region_type"] == "helix"].sort_values("median_z", ascending=True)
    if len(helix_sorted):
        top = helix_sorted.iloc[0]
        print(f"  {top['label']} (median_z={top['median_z']:.2f})")


if __name__ == "__main__":
    main()
