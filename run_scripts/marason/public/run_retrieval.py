"""Retrieval among PubChem isomers for a MARASON model trained by run_scripts/marason/public/run_all.sh.

Mirrors run_scripts/marason/run_retrieval_nist.py, parameterized by dataset.
Expects data/spec_datasets/<dataset>/retrieval/cands_df_<split>_<max_k>.tsv
(shipped for nplib1 by data_scripts/public/create_public_dataset.py).
"""
import argparse
import os
import subprocess
from pathlib import Path

import yaml

pred_file = "src/ms_pred/marason/predict_smis.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_benchmark.py"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-split", default="split_1_rnd1")
    parser.add_argument("--test-split", default="split_1")
    parser.add_argument("--max-k", default=50, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-cpu-workers", default=10, type=int)
    parser.add_argument("--add-ref", action="store_true", help="Use retrieved reference spectra at prediction time.")
    return parser.parse_args()


def main():
    args = get_args()
    dataset, split = args.dataset, args.test_split
    inten_dir = Path(f"results/marason_{dataset}") / args.train_split
    inten_model = inten_dir / "ckpt/inten/best.ckpt"
    if not inten_model.exists():
        raise FileNotFoundError(inten_model)

    labels = Path(f"data/spec_datasets/{dataset}/retrieval/cands_df_{split}_{args.max_k}.tsv")
    if not labels.exists():
        raise FileNotFoundError(labels)
    true_labels = Path(f"data/spec_datasets/{dataset}/labels.tsv")
    ref_dir = Path(f"data/spec_datasets/{dataset}/closest_neighbors/{split}")

    save_dir = inten_dir / f"retrieval_{dataset}_{split}_{args.max_k}"
    save_dir.mkdir(exist_ok=True, parents=True)

    train_args = yaml.safe_load(open(inten_dir / "args.yaml"))
    gen_model = Path(train_args["magma_dag_folder"]).parent / "ckpt/gen/best.ckpt"

    cmd = f"""python {pred_file} \\
    --batch-size {args.batch_size} \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --max-nodes 100 \\
    --split-name {split}.tsv \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {inten_model} \\
    --save-dir {save_dir} \\
    --ref-dir {ref_dir} \\
    --dataset-labels {labels} \\
    --num-cpu-workers {args.num_cpu_workers} \\
    --num-gpu-workers 0 \\
    --magma-dag-folder {train_args["magma_dag_folder"]} \\
    --adduct-shift \\
    --gpu \\
    --embed-elem-group \\
    --binned-out \\
    """
    if args.add_ref:
        cmd += "--add-ref --max-ref-count 3 "
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
    print(cmd + "\n")
    subprocess.run(cmd, shell=True, check=True, env=env)

    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --true-labels {true_labels} \\
    --full-labels {labels} \\
    --formula-dir-name no_subform.hdf5 \\
    --pred-file {save_dir / "binned_preds.hdf5"} \\
    --dist-fn cos \\
    --binned-pred \\
    """
    print(cmd + "\n")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
