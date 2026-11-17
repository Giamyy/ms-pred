#!/usr/bin/env python3
"""Convert public MS/MS libraries into ms-pred spec_datasets for MARASON / ICEBERG.

Sources (see data_scripts/public/download_public_datasets.sh):
  msnlib    MSnLib MS2-only positive-mode .mgf files (Zenodo)
  massbank  MassBank EU NIST-format .msp (GitHub release)
  nplib1    canopus_train_public (NPLIB1 / GNPS subset, Zenodo)

Output layout, independent of the MassSpecGym folders:

    data/spec_datasets/<name>/
    ├── labels.tsv
    ├── spec_files.hdf5        # one SIRIUS-style "<spec>.ms" string per entry
    ├── splits/split_1.tsv     # InChIKey-2D grouped split (nplib1 keeps its official splits)
    └── audit.json

Every collision energy is written in eV, which is what ms-pred models expect.
Entries are grouped like NIST20: one spec per (molecule, adduct, instrument)
holding one spectrum per collision energy.

Only pandas, numpy, h5py and rdkit are required (ms_pred / torch are not).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# Mirrors ms_pred.common.chem_utils (ion2onehot_pos, instrument2onehot_pos, VALID_ELEMENTS)
SUPPORTED_ADDUCTS = {
    "[M+H]+": "[M+H]+",
    "[M+Na]+": "[M+Na]+",
    "[M+K]+": "[M+K]+",
    "[M-H2O+H]+": "[M-H2O+H]+",
    "[M+H-H2O]+": "[M-H2O+H]+",
    "[M+NH4]+": "[M+NH4]+",
    "[M+H3N+H]+": "[M+NH4]+",
    "[M]+": "[M]+",
    "[M-H4O2+H]+": "[M-H4O2+H]+",
    "[M+H-2H2O]+": "[M-H4O2+H]+",
    "[M-2H2O+H]+": "[M-H4O2+H]+",
}
VALID_ELEMENTS = {"C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B", "Se", "Fe", "Co", "As", "Na", "K"}
MAX_PRECURSOR_MZ = 1500.0
_H, _O, _N, _E = 1.007825032, 15.99491462, 14.003074, 0.00054858
ADDUCT_SHIFT = {  # precursor m/z = exact mass + shift, as ms_pred.common.ion2mass
    "[M+H]+": _H - _E,
    "[M+Na]+": 22.98976928 - _E,
    "[M+K]+": 38.96370668 - _E,
    "[M-H2O+H]+": -_O - _H - _E,
    "[M+NH4]+": _N + 4 * _H - _E,
    "[M]+": -_E,
    "[M-H4O2+H]+": -2 * _O - 3 * _H - _E,
}


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["msnlib", "massbank", "nplib1"])
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="msnlib: *_pos_ms2.mgf files; massbank: MassBank_NISTformat.msp; "
                             "nplib1: the extracted canopus_train_public folder")
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--split-frac", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--min-peaks", type=int, default=3, help="Drop spectra with fewer peaks.")
    parser.add_argument("--keep-unknown-instruments", action="store_true",
                        help="massbank: keep QQ / ion-trap spectra with instrument 'Unknown' (dropped by default).")
    parser.add_argument("--nplib1-ce", choices=["merge", "keep"], default="merge",
                        help="nplib1: 'merge' merges all MS2 blocks into one spectrum without collision energy "
                             "(as in the ICEBERG paper); 'keep' keeps the few annotated collision energies.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------- parsers

def iter_mgf(path: Path):
    """Yield (metadata dict, peaks array) for each BEGIN IONS ... END IONS block."""
    meta, peaks = None, []
    with open(path, errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if line == "BEGIN IONS":
                meta, peaks = {}, []
            elif line == "END IONS":
                if meta is not None:
                    yield meta, np.array(peaks, dtype=float).reshape(-1, 2)
                meta = None
            elif meta is None or not line:
                continue
            elif "=" in line and not line[0].isdigit():
                key, val = line.split("=", 1)
                meta[key.strip().upper()] = val.strip()
            else:
                vals = line.split()
                if len(vals) >= 2:
                    peaks.append((float(vals[0]), float(vals[1])))


def iter_msp(path: Path):
    """Yield (metadata dict, peaks array) for each NIST msp record."""
    meta, peaks, in_peaks = {}, [], False
    with open(path, errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                if meta:
                    yield meta, np.array(peaks, dtype=float).reshape(-1, 2)
                meta, peaks, in_peaks = {}, [], False
            elif in_peaks:
                vals = line.split('"')[0].replace(";", " ").split()
                if len(vals) >= 2:
                    peaks.append((float(vals[0]), float(vals[1])))
            elif ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip().lower()] = val.strip()
                in_peaks = key.strip().lower() == "num peaks"
    if meta:
        yield meta, np.array(peaks, dtype=float).reshape(-1, 2)


def parse_ms_file(text: str):
    """Parse a SIRIUS .ms file into (header dict, [(block name, peaks array)])."""
    header, blocks, cur = {}, [], None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            key, _, val = line[1:].partition(" ")
            if key.startswith("collision") or key.endswith("peaks") or key.endswith("merged"):
                cur = (line[1:], [])
                blocks.append(cur)
            else:
                header[key] = val.strip()
                cur = None
        elif line.startswith("#"):
            key, _, val = line[1:].partition(" ")
            header.setdefault(key, val.strip())
        elif cur is not None:
            vals = line.split()
            if len(vals) >= 2:
                cur[1].append((float(vals[0]), float(vals[1])))
    return header, [(name, np.array(p, dtype=float).reshape(-1, 2)) for name, p in blocks]


# ---------------------------------------------------------------- unit conversion

def nce_to_ev(nce: float, precursor_mz: float) -> float:
    """Same conversion as ms_pred.common.nce_to_ev."""
    return nce * precursor_mz / 500


def ce_key(ev: float) -> str:
    return f"{ev:.0f}"


def parse_massbank_ce(text: str, instrument_type: str, precursor_mz: float):
    """Return a single collision energy in eV, or None when it cannot be resolved.

    Ramps, stepped / multiple energies and kV entries are rejected. NCE / % /
    "(nominal)" values are converted with the precursor m/z. Bare numbers are
    read as NCE on Orbitrap instruments and as eV otherwise.
    """
    if not text:
        return None
    t = text.strip().lower()
    if any(tok in t for tok in ("ramp", "kv", " or ", "->")):
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    if len(nums) != 1:
        return None
    val = abs(float(nums[0]))
    is_nce = "%" in t or "nce" in t or "nominal" in t
    is_ev = re.search(r"\d\s*e?v\b", t) is not None or "ev" in t
    if is_nce:
        return nce_to_ev(val, precursor_mz)
    if is_ev:
        return val
    if "FT" in instrument_type.upper():  # Orbitrap (QFT, ITFT) report NCE
        return nce_to_ev(val, precursor_mz)
    return val


def massbank_instrument(instrument_type: str, keep_unknown: bool):
    t = instrument_type.upper()
    if "EI" in t.split("-") or t.startswith("GC") or "MALDI" in t or "FAB" in t:
        return None
    if "ITFT" in t:
        return "IT-FT"
    if "QFT" in t or t.endswith("-FT"):
        return "Orbitrap"
    if "TOF" in t:
        return "QTOF"
    return "Unknown" if keep_unknown else None


def nplib1_instrument(text: str) -> str:
    t = text.lower()
    if "orbitrap" in t:
        return "Orbitrap"
    if "q-tof" in t or "qtof" in t:
        return "QTOF"
    return "Unknown"


# ---------------------------------------------------------------- molecules

def standardize_smiles(smi: str, allow_charged: bool = False):
    """Return (canonical smiles, inchikey, formula) or None for unusable molecules."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    RDLogger.DisableLog("rdApp.*")
    if not isinstance(smi, str) or not smi or smi.lower() in {"n/a", "na", "none", "nan"}:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None or "." in Chem.MolToSmiles(mol):
        return None
    if Chem.GetFormalCharge(mol) != 0 and not allow_charged:
        return None
    if {a.GetSymbol() for a in mol.GetAtoms()} - VALID_ELEMENTS:
        return None
    if mol.GetNumHeavyAtoms() < 2:
        return None
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol), CalcMolFormula(mol)


def standardize_all(smiles, num_workers: int, allow_charged: bool = False) -> dict:
    uniq = sorted({s for s in smiles if isinstance(s, str)})
    fn = partial(standardize_smiles, allow_charged=allow_charged)
    with Pool(num_workers) as pool:
        out = list(tqdm(pool.imap(fn, uniq, chunksize=500), total=len(uniq), desc="rdkit"))
    return dict(zip(uniq, out))


# ---------------------------------------------------------------- records -> dataset

class Collector:
    """Group spectra into NIST-style entries: one spec per (inchikey, adduct, instrument)."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.groups = {}
        self.dropped = Counter()

    def drop(self, reason: str):
        self.dropped[reason] += 1

    def add(self, mol_info, adduct, instrument, precursor_mz, ev, peaks, min_peaks, name="", spec=None, source=""):
        """Add one spectrum. With spec set, the entry keeps that name and is not merged by molecule.

        source separates replicate measurements (e.g. MassBank contributors) into their own entries.
        """
        peaks = peaks[(peaks[:, 1] > 0) & (peaks[:, 0] <= precursor_mz + 1.0)] if len(peaks) else peaks
        if len(peaks) < min_peaks:
            return self.drop("too_few_peaks")
        canon_smi, inchikey, formula = mol_info
        key = spec if spec is not None else (inchikey, adduct, instrument, source)
        group = self.groups.get(key)
        if group is None:
            group = self.groups[key] = {
                "spec": spec,
                "name": name,
                "smiles": canon_smi,
                "inchikey": inchikey,
                "formula": formula,
                "ionization": adduct,
                "instrument": instrument,
                "precursors": [],
                "specs": {},
            }
        group["precursors"].append(precursor_mz)
        ce = "nan" if ev is None else ce_key(ev)
        old = group["specs"].get(ce)
        if old is not None:
            self.drop("duplicate_collision_energy")
            if len(old) >= len(peaks):
                return
        group["specs"][ce] = peaks

    def entries(self):
        ordered = sorted(self.groups.values(),
                         key=lambda g: (g["spec"] or "", g["inchikey"], g["ionization"], g["instrument"]))
        for ind, group in enumerate(ordered):
            if group["spec"] is None:
                group["spec"] = f"{self.prefix}_{ind:07d}"
            group["precursor"] = float(np.median(group["precursors"]))
            yield group


def ce_sort_key(ce: str):
    return float("inf") if ce == "nan" else float(ce)


def ms_string(entry) -> str:
    header = "\n".join(
        [
            f">compound {entry['spec']}",
            f">formula {entry['formula']}",
            f">parentmass {entry['precursor']:.6f}",
            f">ionization {entry['ionization']}",
            f">instrumentation {entry['instrument']}",
            f">smiles {entry['smiles']}",
            f">InChIKey {entry['inchikey']}",
        ]
    )
    blocks = []
    for ce in sorted(entry["specs"], key=ce_sort_key):
        peaks = entry["specs"][ce]
        intens = peaks[:, 1] / peaks[:, 1].max()
        rows = "\n".join(f"{mz:.6f} {inten:.6g}" for mz, inten in zip(peaks[:, 0], intens))
        blocks.append(f">collision {ce}\n{rows}")
    return header + "\n\n" + "\n\n".join(blocks) + "\n"


def grouped_split(labels: pd.DataFrame, fracs, seed: int) -> pd.DataFrame:
    """Random split over InChIKey first blocks so no 2D structure is shared across folds."""
    keys = labels["inchikey"].str.split("-").str[0]
    uniq = np.array(sorted(keys.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fracs = np.array(fracs, dtype=float) / np.sum(fracs)
    n_train = int(round(fracs[0] * len(uniq)))
    n_val = int(round(fracs[1] * len(uniq)))
    fold = {k: "train" for k in uniq[:n_train]}
    fold.update({k: "val" for k in uniq[n_train:n_train + n_val]})
    fold.update({k: "test" for k in uniq[n_train + n_val:]})
    return pd.DataFrame({"spec": labels["spec"], "Fold_0": keys.map(fold)})


def write_dataset(collector: Collector, dataset_name: str, out_dir: Path, args, splits=None, extra_audit=None):
    entries = list(collector.entries())
    if not entries:
        raise ValueError("No entries left after filtering.")
    labels = pd.DataFrame(
        {
            "dataset": dataset_name,
            "spec": [e["spec"] for e in entries],
            "name": [e["name"] for e in entries],
            "ionization": [e["ionization"] for e in entries],
            "formula": [e["formula"] for e in entries],
            "smiles": [e["smiles"] for e in entries],
            "inchikey": [e["inchikey"] for e in entries],
            "instrument": [e["instrument"] for e in entries],
            "collision_energies": [str(sorted(e["specs"], key=ce_sort_key)) for e in entries],
            "precursor": [round(e["precursor"], 6) for e in entries],
        }
    )

    # Only replace what this script writes; magma_outputs / subformulae from earlier runs are kept.
    generated = [out_dir / n for n in ("labels.tsv", "spec_files.hdf5", "splits", "retrieval", "audit.json")]
    if any(p.exists() for p in generated):
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already holds a dataset. Re-run with --overwrite to replace it.")
        for p in generated:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    (out_dir / "splits").mkdir(parents=True)
    labels.to_csv(out_dir / "labels.tsv", sep="\t", index=False)

    if splits is None:
        splits = {"split_1.tsv": grouped_split(labels, args.split_frac, args.split_seed)}
    for split_name, split_df in splits.items():
        split_df.to_csv(out_dir / "splits" / split_name, sep="\t", index=False)

    str_dtype = h5py.special_dtype(vlen=str)
    with h5py.File(out_dir / "spec_files.hdf5", "w") as h5:
        for entry in tqdm(entries, desc="spec_files.hdf5"):
            ds = h5.create_dataset(f"{entry['spec']}.ms", (1,), dtype=str_dtype, compression="gzip")
            ds[0] = ms_string(entry)

    n_spectra = sum(len(e["specs"]) for e in entries)
    audit = {
        "dataset": dataset_name,
        "inputs": [str(p) for p in args.input],
        "entries": len(entries),
        "spectra": n_spectra,
        "unique_inchikey_2d": int(labels["inchikey"].str.split("-").str[0].nunique()),
        "spectra_without_collision_energy": sum("nan" in e["specs"] for e in entries),
        "adducts": labels["ionization"].value_counts().to_dict(),
        "instruments": labels["instrument"].value_counts().to_dict(),
        "splits": {name: df.iloc[:, 1].value_counts().to_dict() for name, df in splits.items()},
        "dropped": dict(collector.dropped.most_common()),
    }
    audit.update(extra_audit or {})
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return labels


# ---------------------------------------------------------------- sources

def build_msnlib(args):
    records = []
    for path in args.input:
        for meta, peaks in tqdm(iter_mgf(path), desc=path.name):
            records.append((meta, peaks))
    mol_map = standardize_all([m.get("SMILES") for m, _ in records], args.num_workers)

    col = Collector("msnlib")
    for meta, peaks in records:
        if meta.get("MSLEVEL", "2") != "2" or meta.get("SPECTYPE") != "SINGLE_BEST_SCAN":
            # MS2 files also hold merged spectra (SAME_ENERGY, ALL_ENERGIES, ALL_MSN_TO_PSEUDO_MS2)
            col.drop("not_single_energy_scan")
            continue
        if meta.get("IONMODE", "Positive").lower() != "positive":
            col.drop("negative_mode")
            continue
        if meta.get("QUALITY_CHIMERIC", "PASSED") != "PASSED":
            col.drop("chimeric")
            continue
        adduct = SUPPORTED_ADDUCTS.get(meta.get("ADDUCT", ""))
        if adduct is None:
            col.drop("unsupported_adduct")
            continue
        mol_info = mol_map.get(meta.get("SMILES"))
        if mol_info is None:
            col.drop("invalid_molecule")
            continue
        try:
            precursor_mz = float(meta["PEPMASS"].split()[0])
            nce = float(meta["COLLISION_ENERGY"].strip("[]"))
        except (KeyError, ValueError):
            col.drop("missing_precursor_or_ce")
            continue
        if precursor_mz > MAX_PRECURSOR_MZ:
            col.drop("precursor_too_large")
            continue
        col.add(mol_info, adduct, "Orbitrap", precursor_mz, nce_to_ev(nce, precursor_mz),
                peaks, args.min_peaks, name=meta.get("NAME", ""))
    return write_dataset(col, "MSnLib", args.output_dataset, args)


def build_massbank(args):
    records = []
    for path in args.input:
        for meta, peaks in tqdm(iter_msp(path), desc=path.name):
            records.append((meta, peaks))
    mol_map = standardize_all([m.get("smiles") for m, _ in records], args.num_workers)

    col = Collector("massbank")
    for meta, peaks in records:
        if meta.get("spectrum_type", "").upper() != "MS2":
            col.drop("not_ms2")
            continue
        if meta.get("ion_mode", "").upper() != "POSITIVE":
            col.drop("negative_mode")
            continue
        adduct = SUPPORTED_ADDUCTS.get(meta.get("precursor_type", ""))
        if adduct is None:
            col.drop("unsupported_adduct")
            continue
        instrument = massbank_instrument(meta.get("instrument_type", ""), args.keep_unknown_instruments)
        if instrument is None:
            col.drop("unsupported_instrument")
            continue
        mol_info = mol_map.get(meta.get("smiles"))
        if mol_info is None:
            col.drop("invalid_molecule")
            continue
        try:
            precursor_mz = float(meta["precursormz"].split()[0].rstrip(","))
        except (KeyError, ValueError, IndexError):
            try:
                precursor_mz = float(meta["exactmass"]) + ADDUCT_SHIFT[adduct]
            except (KeyError, ValueError):
                col.drop("missing_precursor")
                continue
        if precursor_mz > MAX_PRECURSOR_MZ:
            col.drop("precursor_too_large")
            continue
        ev = parse_massbank_ce(meta.get("collision_energy", ""), meta.get("instrument_type", ""), precursor_mz)
        if ev is None:
            col.drop("unresolved_collision_energy")
            continue
        # DB# looks like MSBNK-<contributor>-<id>
        contributor = meta.get("db#", "").split("-")[1] if meta.get("db#", "").count("-") >= 2 else ""
        col.add(mol_info, adduct, instrument, precursor_mz, ev, peaks, args.min_peaks,
                name=meta.get("name", ""), source=contributor)
    return write_dataset(col, "MassBank", args.output_dataset, args)


def build_nplib1(args):
    src = args.input[0]
    old_labels = pd.read_csv(src / "labels.tsv", sep="\t")
    # Charged molecules (quaternary ammoniums, phosphocholines) were part of the published NPLIB1 benchmark.
    mol_map = standardize_all(old_labels["smiles"], args.num_workers, allow_charged=True)

    col = Collector("nplib1")
    for row in tqdm(old_labels.itertuples(index=False), total=len(old_labels), desc="spec_files"):
        ms_path = src / "spec_files" / f"{row.spec}.ms"
        adduct = SUPPORTED_ADDUCTS.get(row.ionization)
        mol_info = mol_map.get(row.smiles)
        if not ms_path.exists():
            col.drop("missing_spec_file")
            continue
        if adduct is None:
            col.drop("unsupported_adduct")
            continue
        if mol_info is None:
            col.drop("invalid_molecule")
            continue
        mol_info = (mol_info[0], mol_info[1], row.formula)  # neutral formula as published
        header, blocks = parse_ms_file(ms_path.read_text(errors="replace"))
        precursor_mz = float(header["parentmass"])
        instrument = nplib1_instrument(header.get("instrumentation", ""))
        ms2 = [(name, peaks) for name, peaks in blocks if not name.startswith("ms1") and len(peaks)]
        known = [(float(m.group(1)), p) for n, p in ms2 if (m := re.match(r"collision\s+(\d+(?:\.\d+)?)", n))]
        # Entries stay one-to-one with the original GNPS spectra so the official splits apply.
        if args.nplib1_ce == "keep" and known:
            for ev, peaks in known:
                col.add(mol_info, adduct, instrument, precursor_mz, ev, peaks, args.min_peaks,
                        name=row.spec, spec=row.spec)
        else:
            col.add(mol_info, adduct, instrument, precursor_mz, None, merge_spectra([p for _, p in ms2]),
                    args.min_peaks, name=row.spec, spec=row.spec)

    kept = set(col.groups)
    splits = {}
    for split_path in sorted((src / "splits").glob("*.tsv")):
        df = pd.read_csv(split_path, sep="\t")
        splits[split_path.name] = df[df.iloc[:, 0].isin(kept)].reset_index(drop=True)
    labels = write_dataset(col, "NPLIB1", args.output_dataset, args, splits=splits,
                           extra_audit={"nplib1_ce": args.nplib1_ce})
    write_nplib1_candidates(src, args.output_dataset, labels)
    return labels


def merge_spectra(peak_arrays, tol=0.01):
    """Merge several spectra (each max-normalized) by max intensity within an m/z tolerance."""
    peak_arrays = [p for p in peak_arrays if len(p)]
    if not peak_arrays:
        return np.zeros((0, 2))
    allp = np.concatenate([np.column_stack([p[:, 0], p[:, 1] / p[:, 1].max()]) for p in peak_arrays])
    allp = allp[np.argsort(allp[:, 0])]
    merged = [allp[0].copy()]
    for mz, inten in allp[1:]:
        if mz - merged[-1][0] <= tol:
            if inten > merged[-1][1]:
                merged[-1] = np.array([mz, inten])
        else:
            merged.append(np.array([mz, inten]))
    return np.array(merged)


def write_nplib1_candidates(src: Path, out_dir: Path, labels: pd.DataFrame):
    """Copy the 50-candidate PubChem retrieval tables, adding precursor and collision energy columns."""
    lab = labels.set_index("spec")
    (out_dir / "retrieval").mkdir(exist_ok=True)
    for path in sorted((src / "retrieval").glob("cands_df_split_*_50.tsv")):
        cands = pd.read_csv(path, sep="\t")
        cands = cands[cands["spec"].isin(lab.index)].copy()
        cands["ionization"] = cands["spec"].map(lab["ionization"])
        cands["precursor"] = cands["spec"].map(lab["precursor"])
        cands["collision_energies"] = cands["spec"].map(lab["collision_energies"])
        cands.to_csv(out_dir / "retrieval" / path.name, sep="\t", index=False)


def main():
    args = get_args()
    {"msnlib": build_msnlib, "massbank": build_massbank, "nplib1": build_nplib1}[args.source](args)


if __name__ == "__main__":
    main()
