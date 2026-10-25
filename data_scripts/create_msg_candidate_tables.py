#!/usr/bin/env python3
"""Build MassSpecGym retrieval candidate tables (cands_df_test_{mass,formula}_256.tsv).

Uses the official MassSpecGym candidate lists from Hugging Face
(``MassSpecGym_retrieval_candidates_{mass,formula}.json``: query SMILES -> list
of candidate SMILES) and writes them in the layout of
data_scripts/pubchem/04_make_retrieval_lists.py: one row per candidate with
``spec, smiles, ionization, inchikey, precursor, collision_energies, instrument``,
the true structure first. Candidates are normalised the same way as there
(SMILES -> InChI -> SMILES round trip, InChIKey recomputed, duplicates by
InChIKey dropped).

These tables are rebuilt from the public candidate lists, so they may differ
in small ways from the tables the ms-pred authors distribute.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import ms_pred.common as common


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True,
                        help="labels.tsv whose spectra need candidates (e.g. data/spec_datasets/msg/labels.tsv)")
    parser.add_argument("--source-table", required=True,
                        help="MassSpecGym1.5.tsv (path or URL), used for the 'fold' column")
    parser.add_argument("--candidates", nargs=2, action="append", metavar=("NAME", "JSON"), required=True,
                        help="Output name and candidate JSON, e.g. --candidates mass cands_mass.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def normalize(smi: str):
    norm_smi = common.smi_inchi_round_smi(smi)
    if norm_smi is None:
        return None
    norm_ikey = common.inchikey_from_smiles(norm_smi)
    if not norm_ikey:
        return None
    return norm_smi, norm_ikey


def main():
    args = get_args()
    labels = pd.read_csv(args.labels, sep="\t")
    folds = pd.read_csv(args.source_table, sep="\t", usecols=["identifier", "fold"])
    test_specs = set(folds.loc[folds["fold"].astype(str).str.strip().str.lower() == "test", "identifier"].astype(str))
    test = labels[labels["spec"].astype(str).isin(test_specs)].copy()
    print(f"Test spectra in labels: {len(test)}")

    cand_maps = {name: json.loads(Path(path).read_text()) for name, path in args.candidates}

    # Normalise every distinct SMILES once (queries, their candidates, and the JSON keys)
    to_norm = set(test["smiles"].astype(str))
    for cmap in cand_maps.values():
        to_norm.update(cmap.keys())
        for q in set(test["smiles"].astype(str)) & set(cmap):
            to_norm.update(cmap[q])
    to_norm = sorted(to_norm)
    with Pool(args.workers) as pool:
        normed = dict(zip(to_norm, pool.map(normalize, to_norm, chunksize=2000)))
    print(f"Normalised {len(to_norm)} distinct SMILES")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {"test_spectra": int(len(test))}
    for name, cmap in cand_maps.items():
        # Look up candidates by exact query SMILES, else by normalised InChIKey of the query
        ikey_to_key = {}
        for key in cmap:
            n = normed.get(key)
            if n is not None:
                ikey_to_key.setdefault(n[1], key)

        rows, missing, unparsable = [], 0, 0
        for _, r in test.iterrows():
            true = normed.get(str(r["smiles"]))
            if true is None:
                unparsable += 1
                continue
            key = r["smiles"] if r["smiles"] in cmap else ikey_to_key.get(true[1])
            if key is None:
                missing += 1
                continue
            cands = {true[1]: true[0]}
            for c in cmap[key]:
                n = normed.get(c) or normalize(c)
                if n is not None and n[1] not in cands:
                    cands[n[1]] = n[0]
                if len(cands) >= args.max_k:
                    break
            for ikey, smi in cands.items():
                rows.append({
                    "spec": r["spec"], "smiles": smi, "ionization": r["ionization"], "inchikey": ikey,
                    "precursor": r["precursor"], "collision_energies": r["collision_energies"],
                    "instrument": r.get("instrument", "Orbitrap"),
                })
        out = pd.DataFrame(rows)
        out_path = args.output_dir / f"cands_df_test_{name}_{args.max_k}.tsv"
        out.to_csv(out_path, sep="\t", index=False)
        per_spec = out.groupby("spec").size() if len(out) else pd.Series(dtype=int)
        audit[name] = {
            "output": str(out_path),
            "spectra_with_candidates": int(per_spec.size),
            "spectra_without_candidate_list": missing,
            "spectra_with_unparsable_smiles": unparsable,
            "rows": int(len(out)),
            "candidates_per_spectrum_median": float(per_spec.median()) if per_spec.size else 0.0,
        }
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
