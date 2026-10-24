#!/usr/bin/env python3
"""Merge MAGMA and subformula outputs computed on dataset shards.

Each shard dir (see make_magma_shard.py) holds
``magma_outputs/magma_tree.hdf5`` and ``subformulae/no_subform.hdf5``. Shards
cover disjoint spectra, so merging copies every top-level entry into one file:

* magma_tree.hdf5 is a PredSpecDB: root attributes are copied from the first
  shard (all shards must agree) and the embedded manifest is rebuilt from the
  merged groups.
* no_subform.hdf5 holds one ``{spec}_collision {ce}.json`` string per entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from ms_pred.common.misc_utils import (
    _MANIFEST_GROUP,
    _build_manifest_from_old_structure,
    _embed_manifest_into_h5,
)

MAGMA_REL = Path("magma_outputs/magma_tree.hdf5")
SUBFORM_REL = Path("subformulae/no_subform.hdf5")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Dataset dir that receives magma_outputs/ and subformulae/")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_entries(shard_files: list[Path], out_path: Path, skip: set[str]) -> dict:
    written, duplicates = 0, []
    with h5py.File(out_path, "a") as dst:
        for path in shard_files:
            with h5py.File(path, "r") as src:
                for key in src.keys():
                    if key in skip:
                        continue
                    if key in dst:
                        duplicates.append(key)
                        continue
                    src.copy(key, dst, name=key)
                    written += 1
    if duplicates:
        raise ValueError(f"{len(duplicates)} entries appear in more than one shard, e.g. {duplicates[:5]}")
    return {"entries": written}


def merge_magma(shard_files: list[Path], out_path: Path) -> dict:
    with h5py.File(shard_files[0], "r") as first:
        root_attrs = dict(first.attrs)
    for path in shard_files[1:]:
        with h5py.File(path, "r") as f:
            if dict(f.attrs) != root_attrs:
                raise ValueError(f"Root attributes of {path} differ from {shard_files[0]}")

    with h5py.File(out_path, "w") as dst:
        dst.attrs.update(root_attrs)
    summary = copy_entries(shard_files, out_path, skip={_MANIFEST_GROUP})

    with h5py.File(out_path, "a") as dst:
        manifest = _build_manifest_from_old_structure(dst)
        _embed_manifest_into_h5(dst, manifest, out_path)
    summary["manifest_entries"] = len(manifest["leaf_path"])
    return summary


def merge_subformulae(shard_files: list[Path], out_path: Path) -> dict:
    with h5py.File(out_path, "w"):
        pass
    return copy_entries(shard_files, out_path, skip=set())


def main():
    args = get_args()

    outputs = {MAGMA_REL: args.output_dir / MAGMA_REL, SUBFORM_REL: args.output_dir / SUBFORM_REL}
    for rel in outputs:
        missing = [str(d) for d in args.shard_dirs if not (d / rel).exists()]
        if missing:
            raise FileNotFoundError(f"{rel} missing in shard dirs: {missing}")
    for out in outputs.values():
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        out.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "shards": [str(d) for d in args.shard_dirs],
        "magma_tree": merge_magma([d / MAGMA_REL for d in args.shard_dirs], outputs[MAGMA_REL]),
        "no_subform": merge_subformulae([d / SUBFORM_REL for d in args.shard_dirs], outputs[SUBFORM_REL]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
