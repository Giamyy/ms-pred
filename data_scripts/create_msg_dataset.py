#!/usr/bin/env python3
"""Rebuild a MassSpecGym spec_dataset folder from the public Hugging Face files.

Creates the layout expected by the ICEBERG / GLACIER / MARASON pipelines:

    <output-dataset>/
    ├── labels.tsv
    ├── spec_files.hdf5          # one SIRIUS-style "<spec>.ms" string per spectrum
    ├── splits/split.tsv         # MassSpecGym fold column (train / val / test)
    └── retrieval/
        ├── cands_df_test_formula_256.tsv
        └── cands_df_test_mass_256.tsv

Subsets:
  simulation  rows with simulation_challenge == True and a known collision
              energy (this matches the committed data/spec_datasets/msg/labels.tsv)
  known_ce    every row with a known collision energy
  all         every row; missing collision energies are written as ['nan']

Only pandas, numpy, h5py and rdkit are required (ms_pred / torch are not).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


HF_ROOT = "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data"
DEFAULT_SOURCE_TABLE = f"{HF_ROOT}/MassSpecGym1.5.tsv"
DEFAULT_FORMULA_CANDIDATES = f"{HF_ROOT}/molecules/MassSpecGym_retrieval_candidates_formula.json"
DEFAULT_MASS_CANDIDATES = f"{HF_ROOT}/molecules/MassSpecGym_retrieval_candidates_mass.json"
DEFAULT_DOWNLOAD_DIR = Path("data/raw/massspecgym")
DEFAULT_OUTPUT_DATASET = Path("data/spec_datasets/msg")
LINKABLE_RESOURCES = ("spec_files.hdf5", "magma_outputs", "subformulae")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE, help="URL or path of MassSpecGym*.tsv")
    parser.add_argument("--formula-candidates", default=DEFAULT_FORMULA_CANDIDATES,
                        help="URL or path of the formula-bonus retrieval candidate JSON. Pass '' to skip.")
    parser.add_argument("--mass-candidates", default=DEFAULT_MASS_CANDIDATES,
                        help="URL or path of the mass-based retrieval candidate JSON. Pass '' to skip.")
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR,
                        help="Where downloaded source files are cached.")
    parser.add_argument("--output-dataset", type=Path, default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--subset", choices=["simulation", "known_ce", "all"], default="simulation")
    parser.add_argument("--link-resources-from", type=Path, default=None,
                        help="Existing dataset folder (e.g. data/spec_datasets/msg_all) whose spec_files.hdf5, "
                             "magma_outputs and subformulae are symlinked instead of writing a new spec_files.hdf5.")
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fetch(source: str, download_dir: Path) -> Path:
    """Return a local path for source, downloading URLs into download_dir once."""
    if not source.startswith(("http://", "https://")):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    download_dir.mkdir(parents=True, exist_ok=True)
    path = download_dir / source.rsplit("/", 1)[-1]
    if path.exists():
        print(f"Using cached {path}")
        return path
    print(f"Downloading {source} -> {path}")
    tmp_path = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(source) as resp, open(tmp_path, "wb") as fp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True) as pbar:
            while chunk := resp.read(1 << 20):
                fp.write(chunk)
                pbar.update(len(chunk))
    tmp_path.rename(path)
    return path


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def format_collision_energy(value) -> str:
    """Format a MassSpecGym collision energy the same way as the committed labels ('30', '7.5', 'nan')."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(val):
        return "nan"
    return str(int(val)) if val.is_integer() else f"{val:g}"


def normalize_fold(value) -> str:
    val = str(value).strip().lower()
    if val == "train":
        return "train"
    if val in {"val", "valid", "validation"}:
        return "val"
    return "test"


def load_source(source_table: Path, subset: str) -> pd.DataFrame:
    df = pd.read_csv(source_table, sep="\t")
    df["identifier"] = df["identifier"].astype(str)
    df = df.drop_duplicates(subset=["identifier"], keep="first")
    df["ce_str"] = df["collision_energy"].map(format_collision_energy)
    known_ce = df["ce_str"] != "nan"
    if subset == "simulation":
        df = df[truthy(df["simulation_challenge"]) & known_ce]
    elif subset == "known_ce":
        df = df[known_ce]
    return df.reset_index(drop=True)


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": "MassSpecGym",
            "spec": df["identifier"],
            "ionization": df["adduct"],
            "formula": df["formula"],
            "smiles": df["smiles"],
            "inchikey": df["inchikey"],
            "instrument": df["instrument_type"],
            "collision_energies": [f"['{ce}']" for ce in df["ce_str"]],
            "precursor": df["precursor_mz"],
        }
    )


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"name": df["identifier"], "split": df["fold"].map(normalize_fold)})


def ms_string(row) -> str:
    """SIRIUS-style .ms text read back by ms_pred.common.parse_spectra."""
    mzs = str(row.mzs).split(",")
    intens = np.array([float(i) for i in str(row.intensities).split(",")])
    if intens.max() > 0:
        intens = intens / intens.max()
    header = "\n".join(
        [
            f">compound {row.identifier}",
            f">formula {row.formula}",
            f">parentmass {row.precursor_mz}",
            f">ionization {row.adduct}",
            f">instrumentation {row.instrument_type}",
            f">smiles {row.smiles}",
            f">InChIKey {row.inchikey}",
        ]
    )
    peaks = "\n".join(f"{mz.strip()} {inten:.6g}" for mz, inten in zip(mzs, intens))
    return f"{header}\n\n>collision {row.ce_str}\n{peaks}\n"


def write_spec_files(df: pd.DataFrame, out_path: Path):
    str_dtype = h5py.special_dtype(vlen=str)
    with h5py.File(out_path, "w") as h5:
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="spec_files.hdf5"):
            ds = h5.create_dataset(f"{row.identifier}.ms", (1,), dtype=str_dtype, compression="gzip")
            ds[0] = ms_string(row)


def smiles_to_inchikey(smi: str) -> str:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def make_candidate_table(
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    candidate_json: Path,
    max_candidates: int,
    num_workers: int,
) -> tuple[pd.DataFrame, dict]:
    """Expand MassSpecGym {query smiles: [candidate smiles]} into the ms-pred cands_df format."""
    with open(candidate_json) as fp:
        cand_map = json.load(fp)

    test_specs = set(splits.loc[splits["split"] == "test", "name"])
    queries = labels[labels["spec"].isin(test_specs)]
    missing_queries = queries[~queries["smiles"].isin(cand_map)]
    queries = queries[queries["smiles"].isin(cand_map)]

    uniq_smiles = sorted({smi for q in queries["smiles"].unique() for smi in cand_map[q]})
    print(f"Computing InChIKeys for {len(uniq_smiles)} candidate SMILES ({candidate_json.name})")
    with Pool(num_workers) as pool:
        ikeys = list(tqdm(pool.imap(smiles_to_inchikey, uniq_smiles, chunksize=1000), total=len(uniq_smiles)))
    smi_to_ikey = dict(zip(uniq_smiles, ikeys))

    columns = ["spec", "smiles", "ionization", "inchikey", "precursor", "collision_energies"]
    entries = []
    for row in queries.itertuples(index=False):
        seen = set()
        for cand_smi in cand_map[row.smiles]:
            cand_ikey = smi_to_ikey[cand_smi]
            if not cand_ikey or cand_ikey in seen:
                continue
            seen.add(cand_ikey)
            entries.append(
                (row.spec, cand_smi, row.ionization, cand_ikey, row.precursor, row.collision_energies)
            )
            if len(seen) >= max_candidates:
                break
    stats = {
        "test_queries": int(len(queries) + len(missing_queries)),
        "queries_with_candidates": int(len(queries)),
        "queries_without_candidates": int(len(missing_queries)),
        "candidate_rows": int(len(entries)),
        "invalid_candidate_smiles": int(sum(1 for i in ikeys if not i)),
    }
    return pd.DataFrame(entries, columns=columns), stats


def replace_path(path: Path, overwrite: bool):
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"{path} exists. Re-run with --overwrite to replace it.")
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def write_tsv(df: pd.DataFrame, path: Path, overwrite: bool):
    replace_path(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def main():
    args = get_args()
    out_dir = args.output_dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    source_table = fetch(args.source_table, args.download_dir)
    df = load_source(source_table, args.subset)
    labels = make_labels(df)
    splits = make_splits(df)

    audit = {
        "source_table": str(args.source_table),
        "subset": args.subset,
        "rows": int(len(labels)),
        "missing_collision_energy": int((df["ce_str"] == "nan").sum()),
        "splits": {k: int(v) for k, v in splits["split"].value_counts().items()},
    }

    write_tsv(labels, out_dir / "labels.tsv", overwrite=args.overwrite)
    write_tsv(splits, out_dir / "splits/split.tsv", overwrite=args.overwrite)

    if args.link_resources_from is not None:
        linked = {}
        for name in LINKABLE_RESOURCES:
            src = args.link_resources_from / name
            dst = out_dir / name
            linked[name] = src.exists()
            if src.exists():
                replace_path(dst, overwrite=args.overwrite)
                os.symlink(os.path.relpath(src.resolve(), out_dir.resolve()), dst, target_is_directory=src.is_dir())
        audit["linked_resources"] = linked
    else:
        spec_path = out_dir / "spec_files.hdf5"
        replace_path(spec_path, overwrite=args.overwrite)
        write_spec_files(df, spec_path)

    audit["retrieval"] = {}
    for split_name, source in [("formula", args.formula_candidates), ("mass", args.mass_candidates)]:
        if not source:
            continue
        cands, stats = make_candidate_table(
            labels, splits, fetch(source, args.download_dir), args.max_candidates, args.num_workers
        )
        out_path = out_dir / f"retrieval/cands_df_test_{split_name}_{args.max_candidates}.tsv"
        write_tsv(cands, out_path, overwrite=args.overwrite)
        audit["retrieval"][split_name] = stats

    (out_dir / "audit_create_msg_dataset.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
