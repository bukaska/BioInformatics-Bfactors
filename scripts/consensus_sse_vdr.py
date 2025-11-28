#!/usr/bin/env python3
"""
Step 2: Consensus secondary structure (SSE) for VDR LBD.

Takes the residue × structure SSE matrix produced by compute_dssp_vdr.py and:

1. Computes, for each residue (118–427), the consensus 3-state SSE (H/E/C)
   by majority vote across all VDR structures.
2. Records how many structures support that consensus (fractions, counts).
3. Scans the consensus sequence to identify continuous helices (H1–H13).
4. Writes:
   - data/processed/vdr_consensus_sse.csv
   - data/processed/vdr_consensus_helices.csv
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def compute_consensus(row, threshold: float = 0.6):
    """
    Majority vote for one residue position.

    row:      Series with values in {'H','E','C', NaN}
    threshold: fraction (0–1). If the most common SSE appears
               with frequency >= threshold, we accept it as consensus.
               Otherwise consensus is '?' (no clear majority).
    Returns: (consensus_sse, top_frac, n_total, n_H, n_E, n_C)
    """
    vals = row.dropna().astype(str)
    # keep only valid states
    vals = vals[vals.isin(["H", "E", "C"])]

    n_total = len(vals)
    n_H = (vals == "H").sum()
    n_E = (vals == "E").sum()
    n_C = (vals == "C").sum()

    if n_total == 0:
        return "?", np.nan, n_total, n_H, n_E, n_C

    counts = vals.value_counts()
    top_sse = counts.index[0]
    top_frac = counts.iloc[0] / n_total

    if top_frac >= threshold:
        cons = top_sse
    else:
        cons = "?"

    return cons, top_frac, n_total, n_H, n_E, n_C


def find_helices(consensus_sse, start_res: int, end_res: int):
    """
    Scan a consensus SSE series and find continuous 'H' segments.

    consensus_sse: pd.Series indexed by residue number, values in {'H','E','C','?'}
    Returns a DataFrame with columns:
      helix_id, start_res, end_res, length, frac_H_in_window
    """
    helices = []
    in_helix = False
    helix_start = None

    # ensure sorted by residue number
    consensus_sse = consensus_sse.sort_index()

    for resnum, sse in consensus_sse.items():
        if sse == "H" and not in_helix:
            # start a new helix
            in_helix = True
            helix_start = resnum
        elif sse != "H" and in_helix:
            # helix ends at previous residue
            helix_end = resnum - 1
            helices.append((helix_start, helix_end))
            in_helix = False
            helix_start = None

    # handle helix that goes to the end
    if in_helix and helix_start is not None:
        helices.append((helix_start, int(consensus_sse.index.max())))

    # Build dataframe
    rows = []
    for i, (s, e) in enumerate(helices, start=1):
        length = e - s + 1
        # fraction of 'H' inside this window (useful QC)
        window = consensus_sse.loc[s:e]
        frac_H = (window == "H").sum() / len(window)
        rows.append({
            "helix_id": f"H{i}",
            "start_res": s,
            "end_res": e,
            "length": length,
            "frac_H_in_window": frac_H,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix",
        default="data/processed/vdr_dssp_matrix_118_427.csv",
        help="Residue × structure SSE matrix CSV from compute_dssp_vdr.py",
    )
    ap.add_argument("--start", type=int, default=118, help="Start residue (default: 118)")
    ap.add_argument("--end", type=int, default=427, help="End residue (default: 427)")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Majority threshold for consensus (fraction, default: 0.60)",
    )
    args = ap.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise SystemExit(f"[ERROR] Matrix file not found: {matrix_path}")

    out_consensus = Path("data/processed/vdr_consensus_sse.csv")
    out_helices   = Path("data/processed/vdr_consensus_helices.csv")
    out_consensus.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading SSE matrix from: {matrix_path}")
    mat = pd.read_csv(matrix_path, index_col=0)  # index = resseq
    # ensure integer index
    mat.index = mat.index.astype(int)
    # restrict to requested window
    mat = mat.loc[args.start:args.end]

    print(f"Matrix shape in window {args.start}-{args.end}: {mat.shape}")

    # ---- Consensus per residue ----
    records = []

    for resnum, row in mat.iterrows():
        cons, top_frac, n_total, n_H, n_E, n_C = compute_consensus(
            row, threshold=args.threshold
        )
        records.append({
            "resseq": resnum,
            "consensus_sse": cons,
            "top_fraction": top_frac,
            "n_structures": n_total,
            "n_H": n_H,
            "n_E": n_E,
            "n_C": n_C,
        })

    cons_df = pd.DataFrame(records).set_index("resseq")
    cons_df.to_csv(out_consensus)

    print(f"Wrote consensus SSE table → {out_consensus}")
    print("Head:")
    print(cons_df.head())

    # ---- Identify helices H1–Hn ----
    helix_df = find_helices(cons_df["consensus_sse"], args.start, args.end)
    helix_df.to_csv(out_helices, index=False)

    print(f"\nIdentified {len(helix_df)} helices:")
    print(helix_df)
    print(f"\nWrote helix boundaries → {out_helices}")

    print("\nDone. These files will be used for:")
    print("  • SSE track under the VDR LBD violin plot.")
    print("  • Per-helix flexibility summaries (Step 6).")


if __name__ == "__main__":
    main()
