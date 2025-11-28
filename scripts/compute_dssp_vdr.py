#!/usr/bin/env python3
"""
Run DSSP on all VDR LBD structures and build:
- per-structure DSSP CSVs
- long tidy table
- residue x structure matrix for positions 118..427
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.DSSP import DSSP

# ---------- Helpers ----------
DSSP_TO_3 = {
    # 8-state DSSP -> 3-state mapping
    'H':'H', 'G':'H', 'I':'H',        # helices -> H
    'E':'E', 'B':'E',                 # strands -> E
    'T':'C', 'S':'C', '-':'C',        # turns, bends, none -> coil
}

def best_chain_for_lbd(dssp_df, start, end):
    """Choose the chain with the largest coverage within [start, end]."""
    cov = (dssp_df
           .query(f"{start} <= resseq <= {end}")
           .groupby("chain_id")["resseq"].nunique()
           .sort_values(ascending=False))
    return cov.index[0] if len(cov) else None

def run_dssp_on_structure(cif_path, pdb_id):
    """Return a tidy DataFrame of DSSP for all chains in a structure."""
    parser = MMCIFParser(QUIET=True) if cif_path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id.upper(), str(cif_path))
    model = next(structure.get_models())  # first model

    # Run DSSP (this calls external 'mkdssp')
    dssp = DSSP(model, str(cif_path))     # works with PDB or mmCIF

    rows = []
    for key in dssp.keys():
        chain_id, (hetflag, resseq, icode) = key[0], key[1]
        if hetflag.strip():      # skip hetero/waters
            continue
        rec = dssp[key]
        aa = rec.residue.get_resname() if hasattr(rec, "residue") else rec[1]
        dssp8 = rec[2] if isinstance(rec, (list, tuple)) else rec.secondary_structure
        dssp8 = dssp8 if dssp8 in "HGI EBTS-" else '-'  # normalize
        sse3  = DSSP_TO_3.get(dssp8, 'C')
        rows.append({
            "pdb_id": pdb_id.upper(),
            "chain_id": chain_id,
            "resseq": int(resseq),
            "icode": "" if icode == " " else str(icode).strip(),
            "aa": aa,
            "dssp_8": dssp8,
            "sse_3": sse3
        })
    return pd.DataFrame(rows).sort_values(["chain_id","resseq","icode"]).reset_index(drop=True)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="VDR", help="Gene symbol to filter (default: VDR)")
    ap.add_argument("--start", type=int, default=118, help="Start residue number (LBD)")
    ap.add_argument("--end",   type=int, default=427, help="End residue number (LBD)")
    args = ap.parse_args()

    base = Path(".")
    meta = pd.read_csv(base / "data/meta/structures.csv")
    # keep only requested receptor
    meta_vdr = meta[meta["symbol"] == args.symbol].copy()

    raw_dir   = base / "data/raw/mmcif"
    out_dir   = base / "data/processed/dssp"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_long = []
    coverage_rows = []

    for pdb_id in sorted(meta_vdr["pdb_id"].unique()):
        cif = raw_dir / f"{pdb_id.lower()}.cif"
        if not cif.exists():
            print(f"[WARN] mmCIF missing for {pdb_id}, skipping.")
            continue
        try:
            ddf = run_dssp_on_structure(cif, pdb_id)
            if ddf.empty:
                print(f"[WARN] DSSP empty for {pdb_id}")
                continue

            # pick best chain within LBD window
            chain = best_chain_for_lbd(ddf, args.start, args.end)
            if chain is None:
                # fall back to longest chain overall
                chain = ddf.groupby("chain_id")["resseq"].nunique().idxmax()

            chosen = ddf[ddf["chain_id"] == chain].copy()

            # save per-structure file
            out_csv = out_dir / f"{pdb_id.upper()}_{chain}.csv"
            chosen.to_csv(out_csv, index=False)

            # track coverage
            cov = chosen.query(f"{args.start} <= resseq <= {args.end}")["resseq"].nunique()
            coverage_rows.append({"pdb_id": pdb_id.upper(), "chain_id": chain,
                                  "covered_in_window": cov})

            # stash for combined long table
            all_long.append(chosen.assign(pdb_chain=f"{pdb_id.upper()}_{chain}"))

            print(f"[OK] {pdb_id} chain {chain}: saved {len(chosen)} residues → {out_csv.name}")

        except Exception as e:
            print(f"[ERR] {pdb_id}: {e}")

    # write combined long table
    if all_long:
        long_df = pd.concat(all_long, ignore_index=True)
        long_df.to_csv(base / "data/processed/vdr_dssp_long.csv", index=False)

        # coverage report
        pd.DataFrame(coverage_rows).to_csv(base / "data/processed/vdr_dssp_coverage.csv", index=False)

        # build residue × structure matrix in the window
        win = long_df.query(f"{args.start} <= resseq <= {args.end}").copy()
        # keep one row per site (resseq, pdb_chain); if multiple icode at same resseq, pick first
        win = (win.sort_values(["pdb_chain","resseq","icode"])
                  .drop_duplicates(subset=["pdb_chain","resseq"], keep="first"))
        mat = (win.pivot(index="resseq", columns="pdb_chain", values="sse_3")
                  .reindex(range(args.start, args.end+1)))
        mat.to_csv(base / "data/processed/vdr_dssp_matrix_118_427.csv")

        print("\nOutputs written:")
        print("  • data/processed/vdr_dssp_long.csv")
        print("  • data/processed/vdr_dssp_coverage.csv")
        print("  • data/processed/vdr_dssp_matrix_118_427.csv")
    else:
        print("No DSSP results collected. Check mkdssp availability and inputs.")

if __name__ == "__main__":
    main()
