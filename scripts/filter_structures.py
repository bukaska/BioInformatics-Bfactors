#!/usr/bin/env python3
from pathlib import Path
import argparse, re, sys, json
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

MM = re.compile(r'\bZN\b', re.I)
NUC_TOKENS = re.compile(r'\b(DNA|RNA|DNA/RNA HYBRID| DA | DC | DG | DT | U )\b', re.I)

def sniff_flags(cif_path: Path, max_bytes: int = 2_000_000):
    """Fast text sniff: return (has_nucleic, has_zn, decided_bool)."""
    try:
        with cif_path.open('rb') as fh:
            chunk = fh.read(max_bytes)
        text = chunk.decode('utf-8', errors='ignore')
        has_zn = bool(MM.search(text))
        has_nuc = bool(NUC_TOKENS.search(text))
        # If we saw any token, we're decided
        decided = has_zn or has_nuc
        return has_nuc, has_zn, decided
    except Exception:
        return False, False, True  # treat unreadable as decided (will drop later)

def parse_flags_full(cif_path: Path):
    """Slow but precise: parse mmCIF dictionary and look for signals."""
    try:
        d = MMCIF2Dict(str(cif_path))
        # nucleic acids
        has_nuc = False
        for k in ["_entity_poly.type"]:
            if k in d:
                vals = d[k] if isinstance(d[k], list) else [d[k]]
                vals = [str(v).upper() for v in vals]
                if any(x in vals for x in ["DNA","RNA","DNA/RNA HYBRID"]):
                    has_nuc = True
        if not has_nuc:
            for k in ["_entity_poly.pdbx_strand_id","_entity_poly.pdbx_seq_one_letter_code_can"]:
                if k in d:
                    vals = d[k] if isinstance(d[k], list) else [d[k]]
                    text = " ".join(map(str, vals)).upper()
                    if any(tok in text for tok in [" DA "," DC "," DG "," DT "," U ","RNA"]):
                        has_nuc = True
                        break
        # zinc
        has_zn = False
        for k in ["_chem_comp.id","_pdbx_entity_nonpoly.comp_id",
                  "_atom_site.type_symbol","_atom_site.label_comp_id"]:
            if k in d:
                vals = d[k] if isinstance(d[k], list) else [d[k]]
                vals = [str(v).upper() for v in vals]
                if "ZN" in vals:
                    has_zn = True
                    break
        return has_nuc, has_zn, True
    except Exception:
        return False, False, True  # will drop on error/unreadable

def decide_one(args):
    pdb_id, cif_dir = args
    fp = cif_dir / f"{pdb_id.lower()}.cif"
    if not fp.exists() or fp.stat().st_size == 0:
        return pdb_id, "drop", "missing"

    has_nuc, has_zn, decided = sniff_flags(fp)
    if not decided:
        has_nuc, has_zn, _ = parse_flags_full(fp)

    if has_nuc or has_zn:
        reason = "DNA/RNA" if has_nuc else "Zn"
        return pdb_id, "drop", reason
    return pdb_id, "keep", ""

def main():
    ap = argparse.ArgumentParser(description="Fast filter: drop PDBs with DNA/RNA or Zn.")
    ap.add_argument("--meta", default="data/meta/structures.csv")
    ap.add_argument("--mmcif_dir", default="data/raw/mmcif")
    ap.add_argument("--out_keep", default="data/meta/pdb_keep_noDNA_noZN.txt")
    ap.add_argument("--out_drop", default="data/meta/pdb_drop_DNA_or_ZN.txt")
    ap.add_argument("--out_csv",  default="data/meta/pdb_filter_decisions.csv")
    ap.add_argument("--nprocs", type=int, default=4)
    args = ap.parse_args()

    mmcif_dir = Path(args.mmcif_dir)
    meta = pd.read_csv(args.meta)
    pdb_ids = sorted(meta["pdb_id"].dropna().unique())

    # Resume: load prior decisions if present
    prior = None
    if Path(args.out_csv).exists():
        prior = pd.read_csv(args.out_csv)
        done = set(prior["pdb_id"])
        pdb_ids = [p for p in pdb_ids if p not in done]

    rows = [] if prior is None else prior.to_dict("records")

    with ProcessPoolExecutor(max_workers=args.nprocs) as ex:
        futures = {ex.submit(decide_one, (p, mmcif_dir)): p for p in pdb_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Filtering"):
            pdb_id, decision, reason = fut.result()
            rows.append({"pdb_id": pdb_id, "decision": decision, "reason": reason})

    df = pd.DataFrame(rows).drop_duplicates(subset=["pdb_id"], keep="last")
    df.to_csv(args.out_csv, index=False)

    keep = sorted(df.loc[df["decision"]=="keep","pdb_id"].unique())
    drop = sorted(df.loc[df["decision"]=="drop","pdb_id"].unique())

    Path(args.out_keep).write_text("\n".join(keep) + ("\n" if keep else ""))
    Path(args.out_drop).write_text("\n".join(drop) + ("\n" if drop else ""))

    print(f"Keep (no DNA/RNA, no Zn): {len(keep)}  → {args.out_keep}")
    print(f"Drop (DNA/RNA or Zn / missing): {len(drop)}  → {args.out_drop}")
    print(f"Decisions table: {args.out_csv}")

if __name__ == "__main__":
    sys.exit(main())
