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
    print(f"Test spectra in labels: {len(test)}", flush=True)

    cand_maps = {name: json.loads(Path(path).read_text()) for name, path in args.candidates}
    print(f"Loaded candidate lists: {', '.join(f'{k} ({len(v)} queries)' for k, v in cand_maps.items())}", flush=True)

    normed = {}

    def normalize_all(smiles, label):
        # Normalise each distinct SMILES once, in parallel; failures are cached as None too
        todo = sorted(set(smiles) - normed.keys())
        print(f"{label}: normalising {len(todo)} SMILES with {args.workers} workers", flush=True)
        with Pool(args.workers) as pool:
            for i, (smi, res) in enumerate(zip(todo, pool.imap(normalize, todo, chunksize=500)), 1):
                normed[smi] = res
                if i % 50000 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)

    # 1) queries and every JSON key, so queries can be matched to keys by InChIKey
    query_smiles = set(test["smiles"].astype(str))
    normalize_all(query_smiles.union(*(cmap.keys() for cmap in cand_maps.values())), "queries + candidate-list keys")

    # 2) pick the candidate list of each query, then normalise those candidates
    matched = {}
    for name, cmap in cand_maps.items():
        ikey_to_key = {}
        for key in cmap:
            if normed.get(key) is not None:
                ikey_to_key.setdefault(normed[key][1], key)
        for q in query_smiles:
            if normed.get(q) is not None:
                matched[(name, q)] = q if q in cmap else ikey_to_key.get(normed[q][1])
    needed = set()
    for (name, q), key in matched.items():
        if key is not None:
            needed.update(cand_maps[name][key])
    normalize_all(needed, "candidates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {"test_spectra": int(len(test))}
    for name, cmap in cand_maps.items():
        rows, missing, unparsable = [], 0, 0
        for _, r in test.iterrows():
            true = normed.get(str(r["smiles"]))
            if true is None:
                unparsable += 1
                continue
            key = matched.get((name, str(r["smiles"])))
            if key is None:
                missing += 1
                continue
            cands = {true[1]: true[0]}
            for c in cmap[key]:
                n = normed.get(c)
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
