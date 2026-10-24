#!/usr/bin/env python3
"""Cut one shard out of a dataset so MAGMA can run on it in a separate job.

Shard ``i`` of ``k`` takes rows ``i, i+k, i+2k, ...`` of the source labels.tsv
and copies the matching ``{spec}.ms`` entries of spec_files.hdf5. Run
``bash data_scripts/dag/run_magma.sh <output dataset name>`` on the result,
then combine the shards with data_scripts/dag/merge_magma_shards.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True,
                        help="Dataset dir with labels.tsv and spec_files.hdf5")
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"--shard-id must be in [0, {args.num_shards})")

    labels = pd.read_csv(args.source_dataset / "labels.tsv", sep="\t")
    shard = labels.iloc[args.shard_id::args.num_shards]

    args.output_dataset.mkdir(parents=True, exist_ok=True)
    shard.to_csv(args.output_dataset / "labels.tsv", sep="\t", index=False)

    missing = []
    with h5py.File(args.source_dataset / "spec_files.hdf5", "r") as src, \
            h5py.File(args.output_dataset / "spec_files.hdf5", "w") as dst:
        for spec in shard["spec"].astype(str):
            key = f"{spec}.ms"
            if key in src:
                src.copy(key, dst, name=key)
            else:
                missing.append(spec)

    summary = {
        "source_specs": int(len(labels)),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "shard_specs": int(len(shard)),
        "missing_in_spec_files": len(missing),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
