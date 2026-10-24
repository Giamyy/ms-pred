#!/usr/bin/env python3
"""Convert MassSpecGym1.5.tsv into the repository's spec_files.hdf5 layout.

Each spectrum is stored as one SIRIUS-style ``.ms`` string under the key
``{identifier}.ms``, which is what ``common.HDF5Dataset.read_str`` and
``common.parse_spectra`` expect (see run_magma.py and
data_scripts/forms/01_assign_subformulae.py). Example entry:

    >compound MassSpecGymID0000001
    >formula C16H17NO4
    >parentmass 288.1225
    >ionization [M+H]+
    >instrumentation Orbitrap
    >smiles CC(=O)N[C@@H](CC1=CC=CC=C1)C2=CC(=CC(=O)O2)OC
    >InChIKey VFMQMACUYWGDOJ

    >collision 30
    91.0542 0.2316
    ...

Rows without a known collision energy are written under ``>collision nan``
(the key ICEBERG uses for missing energies) unless ``--missing-ce skip`` is set.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import pandas as pd


DEFAULT_SOURCE_TABLE = (
    "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/"
    "resolve/main/data/MassSpecGym1.5.tsv"
)
DEFAULT_OUTPUT = Path("data/spec_datasets/msg/spec_files.hdf5")

META_COLUMNS = {
    "formula": "formula",
    "parentmass": "precursor_mz",
    "ionization": "adduct",
    "instrumentation": "instrument_type",
    "smiles": "smiles",
    "InChIKey": "inchikey",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE,
                        help="Path or URL of MassSpecGym1.5.tsv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--labels", type=Path, default=None,
                        help="Optional labels.tsv; only specs in its 'spec' column are written")
    parser.add_argument("--missing-ce", choices=["nan", "skip"], default="nan",
                        help="How to handle rows without a collision energy")
    parser.add_argument("--id-col", default="identifier")
    parser.add_argument("--mz-col", default="mzs")
    parser.add_argument("--inten-col", default="intensities")
    parser.add_argument("--ce-col", default="collision_energy")
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after writing this many spectra (for quick checks)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-out", type=Path, default=None)
    return parser.parse_args()


def format_collision_energy(value) -> str | None:
    """Return the energy label used in '>collision {label}', or None if unknown."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val):
        return None
    return str(int(val)) if val.is_integer() else f"{val:g}"


def split_peaks(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def build_ms_string(row: pd.Series, args, ce_label: str) -> str | None:
    """Build one SIRIUS .ms string, or None if the peak lists are unusable."""
    mzs = split_peaks(row[args.mz_col])
    intens = split_peaks(row[args.inten_col])
    if len(mzs) == 0 or len(mzs) != len(intens):
        return None
    try:
        peaks = [(float(m), float(i)) for m, i in zip(mzs, intens)]
    except ValueError:
        return None

    header = [f">compound {row[args.id_col]}"]
    for key, col in META_COLUMNS.items():
        val = row.get(col)
        if val is not None and not pd.isna(val) and str(val).strip():
            header.append(f">{key} {str(val).strip()}")

    # The blank line separates metadata from the spectrum block; parse_spectra
    # relies on it to tell the two groups apart.
    peak_lines = [f"{m} {i}" for m, i in peaks]
    return "\n".join(header) + f"\n\n>collision {ce_label}\n" + "\n".join(peak_lines) + "\n"


def write_str(h5: h5py.File, name: str, data: str):
    """Same on-disk layout as common.HDF5Dataset.write_str."""
    dt = h5py.special_dtype(vlen=str)
    ds = h5.create_dataset(name, (1,), dtype=dt, compression="gzip")
    ds[0] = data


def main():
    args = get_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    keep_specs = None
    if args.labels is not None:
        keep_specs = set(pd.read_csv(args.labels, sep="\t", usecols=["spec"])["spec"].astype(str))

    required = {args.id_col, args.mz_col, args.inten_col, args.ce_col}
    audit = {
        "source_table": str(args.source_table),
        "source_rows": 0,
        "written": 0,
        "written_missing_ce": 0,
        "skipped_not_in_labels": 0,
        "skipped_missing_ce": 0,
        "skipped_bad_peaks": 0,
        "skipped_duplicate_id": 0,
    }
    seen = set()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_name(args.output.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    done = False
    with h5py.File(tmp_path, "w") as h5:
        for chunk in pd.read_csv(args.source_table, sep="\t", chunksize=args.chunksize):
            missing = required - set(chunk.columns)
            if missing:
                raise KeyError(f"Missing columns {sorted(missing)}; available: {list(chunk.columns)}")

            for _, row in chunk.iterrows():
                audit["source_rows"] += 1
                spec = str(row[args.id_col])

                if keep_specs is not None and spec not in keep_specs:
                    audit["skipped_not_in_labels"] += 1
                    continue
                if spec in seen:
                    audit["skipped_duplicate_id"] += 1
                    continue

                ce_label = format_collision_energy(row[args.ce_col])
                if ce_label is None:
                    if args.missing_ce == "skip":
                        audit["skipped_missing_ce"] += 1
                        continue
                    ce_label = "nan"

                ms_str = build_ms_string(row, args, ce_label)
                if ms_str is None:
                    audit["skipped_bad_peaks"] += 1
                    continue

                write_str(h5, f"{spec}.ms", ms_str)
                seen.add(spec)
                audit["written"] += 1
                if ce_label == "nan":
                    audit["written_missing_ce"] += 1

                if args.limit is not None and audit["written"] >= args.limit:
                    done = True
                    break
            if done:
                break

    os.replace(tmp_path, args.output)

    if keep_specs is not None:
        audit["labels_specs"] = len(keep_specs)
        audit["labels_specs_missing_from_output"] = len(keep_specs - seen)

    print(json.dumps(audit, indent=2))
    if args.audit_out is not None:
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        args.audit_out.write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
