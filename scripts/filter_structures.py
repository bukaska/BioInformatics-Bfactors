#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

mmcif_dir = Path("data/raw/mmcif")
meta_path = Path("data/meta/structures.csv")
out_keep  = Path("data/meta/pdb_keep_noDNA_noZN.txt")
out_drop  = Path("data/meta/pdb_drop_DNA_or_ZN.txt")

meta = pd.read_csv(meta_path)
keep = []
drop = []

def has_dna_or_rna(d):
    # look for polymer classes: 'DNA','RNA','DNA/RNA hybrid'
    keys = ["_entity_poly.type","_struct_biol.entity_id"]
    for k in ["_entity_poly.type"]:
        if k in d:
            vals = d[k] if isinstance(d[k], list) else [d[k]]
            vals = [str(v).upper() for v in vals]
            if any(x in vals for x in ["DNA","RNA","DNA/RNA HYBRID"]):
                return True
    # quick lexical fallback: polymer comp type table
    for k in ["_entity_poly.pdbx_strand_id","_entity_poly.pdbx_seq_one_letter_code_can"]:
        if k in d:
            text = " ".join(d[k] if isinstance(d[k], list) else [d[k]]).upper()
            if any(nuc in text for nuc in [" DA "," DC "," DG "," DT "," U ","RNA"]):
                return True
    return False

def has_zn(d):
    # look for ZN as a non-polymer (ligand) component
    for k in ["_chem_comp.id","_pdbx_entity_nonpoly.comp_id","_atom_site.type_symbol","_atom_site.label_comp_id"]:
        if k in d:
            vals = d[k] if isinstance(d[k], list) else [d[k]]
            vals = [str(v).upper() for v in vals]
            if "ZN" in vals:
                return True
    return False

for pdb_id in meta["pdb_id"].dropna().unique():
    fp = mmcif_dir / f"{pdb_id.lower()}.cif"
    if not fp.exists():
        # if missing, keep it out
        drop.append(pdb_id)
        continue
    try:
        d = MMCIF2Dict(str(fp))
        if has_dna_or_rna(d) or has_zn(d):
            drop.append(pdb_id)
        else:
            keep.append(pdb_id)
    except Exception:
        drop.append(pdb_id)

Path(out_keep).write_text("\n".join(sorted(set(keep))) + "\n")
Path(out_drop).write_text("\n".join(sorted(set(drop))) + "\n")
print(f"Keep (no DNA/RNA, no Zn): {len(keep)}  → {out_keep}")
print(f"Drop (DNA/RNA or Zn):     {len(drop)}  → {out_drop}")
