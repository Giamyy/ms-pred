#!/usr/bin/env python3
"""Build 50-candidate PubChem retrieval tables for the public datasets (massbank, msnlib).

Same candidates as the authors' PubChem pipeline (data_scripts/pubchem/01-04, used for NIST20): for every spectrum
of the test fold, all PubChem molecules with the same (uncharged) formula, normalised by an InChI round trip and
deduplicated by InChIKey, ranked by Tanimoto similarity (Morgan, 2048 bits, radius 3) to the true molecule; the true
molecule plus the max_k - 1 most similar isomers are kept. Output:

    data/spec_datasets/<ds>/retrieval/cands_df_split_1_50.tsv      (test fold, the name run_all.sh / the baseline
                                                                    runner look for)
    data/spec_datasets/<ds>/retrieval/cands_df_split_1_val_50.tsv  (with --subsets test val)

with columns spec, smiles, ionization, inchikey, precursor, collision_energies, instrument (instrument is added so
QTOF / IT-FT spectra are not read as the default Orbitrap by predict_smis.py).

Unlike 02_make_formula_subsets.py, PubChem is streamed and only molecules whose formula occurs in the selected
folds are kept, so memory stays small. Stages:

    scan   read CID-SMILES(.gz) in chunks, keep (formula, smiles, inchikey) of matching molecules.
           --num-shards / --shard-id split the chunks over several machines or runs.
    build  merge the shards and write the candidate tables.

    python data_scripts/public/make_pubchem_candidates.py --datasets massbank msnlib --download --workers 32
    # or in parts:  --stage scan --num-shards 4 --shard-id 0 … 3, then --stage build --num-shards 4
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PUBCHEM_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["massbank", "msnlib"])
    parser.add_argument("--split", default="split_1.tsv")
    parser.add_argument("--subsets", nargs="+", default=["test"], choices=["test", "val"])
    parser.add_argument("--max-k", type=int, default=50, help="candidates per spectrum, true molecule included")
    parser.add_argument("--pubchem", type=Path, default=Path("data/retrieval/pubchem/CID-SMILES.gz"),
                        help="CID-SMILES from PubChem (gzip or plain text: CID<TAB>SMILES per line)")
    parser.add_argument("--download", action="store_true", help="download CID-SMILES.gz if it is missing")
    parser.add_argument("--work-dir", type=Path, default=Path("data/retrieval/pubchem/public_candidates"))
    parser.add_argument("--stage", choices=["scan", "build", "all"], default="all")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=20000, help="PubChem lines per task")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


# ---------------------------------------------------------------- targets

def fold_labels(dataset: str, split: str, subset: str) -> pd.DataFrame:
    data_dir = Path("data/spec_datasets") / dataset
    labels = pd.read_csv(data_dir / "labels.tsv", sep="\t")
    split_df = pd.read_csv(data_dir / "splits" / split, sep="\t")
    name_col, fold_col = split_df.columns[:2]
    names = set(split_df.loc[split_df[fold_col] == subset, name_col].astype(str))
    return labels[labels["spec"].astype(str).isin(names)].reset_index(drop=True)


def target_formulas(args) -> list[str]:
    forms = set()
    for ds in args.datasets:
        for subset in args.subsets:
            forms.update(fold_labels(ds, args.split, subset)["formula"].astype(str))
    return sorted(forms)


# ---------------------------------------------------------------- scan

def iter_chunks(path: Path, chunk_size: int, num_shards: int, shard_id: int):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as fp:
        chunk, index = [], 0
        for line in fp:
            chunk.append(line)
            if len(chunk) == chunk_size:
                if index % num_shards == shard_id:
                    yield chunk
                chunk, index = [], index + 1
        if chunk and index % num_shards == shard_id:
            yield chunk


TARGETS = set()


def init_scan(targets):
    global TARGETS
    TARGETS = targets


def scan_chunk(lines):
    """(formula, smiles, inchikey) of molecules whose formula is a target, as 02_make_formula_subsets.py computes them."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    import re

    RDLogger.DisableLog("rdApp.*")
    out = []
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        mol = Chem.MolFromSmiles(parts[1].strip())
        if mol is None:
            continue
        form = re.findall(r"^([^\+,^\-]*)", CalcMolFormula(mol))[0]    # common.uncharged_formula
        if form not in TARGETS:
            continue
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)            # stereochemistry removed, as the authors
            ikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
        except Exception:
            continue
        if ikey:
            out.append((form, smi, ikey))
    return len(lines), out


def run_scan(args, targets):
    if not args.pubchem.exists():
        if not args.download:
            raise SystemExit(f"Không thấy {args.pubchem}; thêm --download để tải {PUBCHEM_URL}")
        args.pubchem.parent.mkdir(parents=True, exist_ok=True)
        print(f"Tải {PUBCHEM_URL} -> {args.pubchem}", flush=True)
        subprocess.run(["curl", "-fL", "--retry", "4", "-o", f"{args.pubchem}.part", PUBCHEM_URL], check=True)
        Path(f"{args.pubchem}.part").rename(args.pubchem)

    out = args.work_dir / f"matches_shard{args.shard_id}of{args.num_shards}.tsv"
    done = out.with_suffix(".done.json")
    if done.exists():
        print(f"Shard {args.shard_id}/{args.num_shards} đã quét xong ({done})")
        return
    t0, n_lines, n_hits = time.time(), 0, 0
    with Pool(args.workers, initializer=init_scan, initargs=(set(targets),)) as pool, open(out, "w") as fp:
        chunks = iter_chunks(args.pubchem, args.chunk_size, args.num_shards, args.shard_id)
        for i, (n, rows) in enumerate(pool.imap(scan_chunk, chunks, chunksize=1), 1):
            n_lines += n
            n_hits += len(rows)
            fp.writelines(f"{f}\t{s}\t{k}\n" for f, s, k in rows)
            if i % 100 == 0:
                print(f"  {n_lines:,} phân tử PubChem, {n_hits:,} trùng công thức, {(time.time() - t0) / 60:.0f} phút", flush=True)
    done.write_text(json.dumps({"pubchem_lines": n_lines, "matches": n_hits, "minutes": round((time.time() - t0) / 60, 1),
                                "targets": len(targets)}, indent=2))
    print(f"Shard {args.shard_id}/{args.num_shards}: {n_lines:,} phân tử, {n_hits:,} trùng công thức -> {out}")


# ---------------------------------------------------------------- build

def load_authors_module():
    """data_scripts/pubchem/04_make_retrieval_lists.py: candidate normalisation of the authors."""
    path = REPO / "data_scripts/pubchem/04_make_retrieval_lists.py"
    spec = importlib.util.spec_from_file_location("make_retrieval_lists", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_and_fp(smi):
    """Normalised (smiles, inchikey) and Morgan fingerprint of one candidate (None if it fails)."""
    from rdkit import RDLogger
    import ms_pred.common as common

    RDLogger.DisableLog("rdApp.*")
    norm = MOD.normalize_candidate(smi)
    if norm is None:
        return None
    fp = common.get_morgan_fp_smi(norm[0])
    return norm[0], norm[1], np.packbits(fp.astype(bool))


def run_build(args, targets):
    global MOD
    MOD = load_authors_module()
    import ms_pred.common as common

    shards = [args.work_dir / f"matches_shard{i}of{args.num_shards}.tsv" for i in range(args.num_shards)]
    missing = [str(p) for p in shards if not p.with_suffix(".done.json").exists()]
    if missing:
        raise SystemExit(f"Chưa quét xong các shard: {missing}")

    # formula -> {raw inchikey: smiles}: unique isomers, as np.unique on the inchikey in 04_make_retrieval_lists.py
    isomers = {f: {} for f in targets}
    for path in shards:
        for line in open(path):
            form, smi, ikey = line.rstrip("\n").split("\t")
            if form in isomers:
                isomers[form].setdefault(ikey, smi)
    print(f"{sum(map(len, isomers.values())):,} đồng phân cho {sum(1 for v in isomers.values() if v)} / {len(targets)} công thức")

    # Normalise every needed candidate once (04 does it per spectrum; the result per candidate is the same)
    uniq = sorted({smi for d in isomers.values() for smi in d.values()})
    with Pool(args.workers) as pool:
        normed = dict(zip(uniq, pool.map(normalize_and_fp, uniq, chunksize=500)))
    print(f"Chuẩn hoá {len(uniq):,} ứng viên", flush=True)

    summary = {}
    for ds in args.datasets:
        data_dir = Path("data/spec_datasets") / ds
        (data_dir / "retrieval").mkdir(exist_ok=True)
        for subset in args.subsets:
            labels = fold_labels(ds, args.split, subset)
            rows, no_isomer, bad_true = [], 0, 0
            for r in labels.itertuples(index=False):
                raw = isomers.get(str(r.formula), {})
                if not raw:
                    no_isomer += 1          # 04 skips spectra whose formula has no PubChem isomer
                    continue
                true = MOD.normalize_candidate(r.smiles)
                if true is None:
                    bad_true += 1
                    continue
                true_smi, true_ikey = true
                label_ikey = str(r.inchikey)
                cands = {}          # normalised inchikey -> (smiles, packed fingerprint), first one kept as in 04
                for ikey, smi in raw.items():
                    if ikey == label_ikey or normed.get(smi) is None:
                        continue
                    c_smi, c_ikey, c_fp = normed[smi]
                    if c_ikey != true_ikey:
                        cands.setdefault(c_ikey, (c_smi, c_fp))
                chosen = [(true_smi, true_ikey)]
                if cands:
                    keys = list(cands)
                    fps = np.unpackbits(np.stack([cands[k][1] for k in keys]), axis=1)[:, :2048].astype(np.float32)
                    orig = common.get_morgan_fp_smi(true_smi).astype(np.float32)
                    inter = fps @ orig
                    tani = inter / (fps.sum(1) + orig.sum() - inter + 1e-22)
                    # high to low; ties keep isomer order (04 uses an unstable argsort, so its tie order is arbitrary)
                    order = np.argsort(-tani, kind="stable")[: args.max_k - 1]
                    chosen += [(cands[keys[i]][0], keys[i]) for i in order]
                for smi, ikey in chosen:
                    rows.append({"spec": r.spec, "smiles": smi, "ionization": r.ionization, "inchikey": ikey,
                                 "precursor": r.precursor, "collision_energies": r.collision_energies,
                                 "instrument": r.instrument})
            out = pd.DataFrame(rows)
            stem = Path(args.split).stem
            name = f"cands_df_{stem}_{args.max_k}.tsv" if subset == "test" else f"cands_df_{stem}_{subset}_{args.max_k}.tsv"
            out.to_csv(data_dir / "retrieval" / name, sep="\t", index=False)
            per = out.groupby("spec").size() if len(out) else pd.Series(dtype=int)
            summary[f"{ds}/{name}"] = {"spectra_in_fold": len(labels), "spectra_with_candidates": int(per.size),
                                       "no_pubchem_isomer": no_isomer, "true_molecule_failed": bad_true,
                                       "rows": len(out), "median_candidates": float(per.median()) if per.size else 0}
            print(f"{data_dir / 'retrieval' / name}: {summary[f'{ds}/{name}']}", flush=True)
    (args.work_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    args = get_args()
    import os
    os.chdir(REPO)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    targets = target_formulas(args)
    (args.work_dir / "targets.json").write_text(json.dumps({"datasets": args.datasets, "subsets": args.subsets,
                                                             "split": args.split, "formulas": targets}))
    print(f"{len(targets)} công thức cần tìm đồng phân ({', '.join(args.datasets)}; {', '.join(args.subsets)})")
    if args.stage in ("scan", "all"):
        run_scan(args, targets)
    if args.stage in ("build", "all"):
        run_build(args, targets)


if __name__ == "__main__":
    main()
