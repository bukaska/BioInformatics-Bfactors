#!/usr/bin/env python3
"""
Step 2: Build consensus secondary structure for VDR LBD from DSSP matrix.

Inputs:
    data/processed/vdr_dssp_matrix_118_427.csv
        rows   = residue numbers (118..427)
        cols   = pdb_chain IDs (e.g. 1DB1_A)
        values = 'H', 'E', 'C' (3-state) or NaN

Outputs:
    data/processed/vdr_sse_consensus_per_residue.csv
        per residue: counts, fractions, consensus state, confidence
    data/processed/vdr_consensus_helices.csv
        list of helices H1..Hn with start/end/length and consensus score
"""

from pathlib import Path
import pandas as pd
import numpy as np


def main():
    base = Path(".")
    mat_path = base / "data/processed/vdr_dssp_matrix_118_427.csv"
    out_res  = base / "data/processed/vdr_sse_consensus_per_residue.csv"
    out_hel  = base / "data/processed/vdr_consensus_helices.csv"

    df = pd.read_csv(mat_path, index_col=0)   # index = resseq
    print(f"Loaded DSSP matrix: {df.shape[0]} residues x {df.shape[1]} structures")

    # ----- majority voting per residue -----
    rows = []
    for resseq, row in df.iterrows():
        vals = row.dropna().astype(str)

        n_total = len(vals)
        if n_total == 0:
            rows.append({
                "resseq": resseq,
                "n_total": 0,
                "n_H": 0, "n_E": 0, "n_C": 0,
                "frac_H": np.nan, "frac_E": np.nan, "frac_C": np.nan,
                "consensus": "C",
                "consensus_frac": np.nan
            })
            continue

        counts = vals.value_counts()
        n_H = int(counts.get("H", 0))
        n_E = int(counts.get("E", 0))
        n_C = int(counts.get("C", 0))

        frac_H = n_H / n_total
        frac_E = n_E / n_total
        frac_C = n_C / n_total

        # majority vote: choose state with highest count
        majority_state = max(("H", "E", "C"), key=lambda s: counts.get(s, 0))
        majority_frac  = counts.get(majority_state, 0) / n_total

        rows.append({
            "resseq": resseq,
            "n_total": n_total,
            "n_H": n_H, "n_E": n_E, "n_C": n_C,
            "frac_H": frac_H, "frac_E": frac_E, "frac_C": frac_C,
            "consensus": majority_state,
            "consensus_frac": majority_frac
        })

    cons_df = pd.DataFrame(rows).set_index("resseq")
    cons_df.to_csv(out_res)
    print(f"Saved per-residue consensus: {out_res}")

    # ----- identify continuous helices H1..Hn -----
    HELIX_THRESHOLD = 0.6   # require ≥60% of structures to be helical

    helix_regions = []
    in_helix = False
    current_start = None
    helix_id = 0

    for resseq, row in cons_df.iterrows():
        is_helix = (row["consensus"] == "H") and (row["consensus_frac"] >= HELIX_THRESHOLD)

        if is_helix and not in_helix:
            # start a new helix
            in_helix = True
            current_start = resseq

        elif (not is_helix) and in_helix:
            # helix ends at previous residue
            in_helix = False
            helix_id += 1
            start = current_start
            end   = resseq - 1
            region = cons_df.loc[start:end]
            mean_conf = region["consensus_frac"].mean()
            helix_regions.append({
                "helix_id": helix_id,
                "label": f"H{helix_id}",
                "start": start,
                "end": end,
                "length": end - start + 1,
                "mean_consensus_frac": mean_conf
            })

    # handle helix that runs to the last residue
    if in_helix:
        helix_id += 1
        start = current_start
        end   = cons_df.index.max()
        region = cons_df.loc[start:end]
        mean_conf = region["consensus_frac"].mean()
        helix_regions.append({
            "helix_id": helix_id,
            "label": f"H{helix_id}",
            "start": start,
            "end": end,
            "length": end - start + 1,
            "mean_consensus_frac": mean_conf
        })

    helices_df = pd.DataFrame(helix_regions)
    helices_df.to_csv(out_hel, index=False)
    print(f"Identified {len(helices_df)} helices; saved: {out_hel}")


if __name__ == "__main__":
    main()
