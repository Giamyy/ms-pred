#!/usr/bin/env bash
# Train and evaluate MARASON on a public dataset built by data_scripts/public/build_public_datasets.sh.
#
# Usage: bash run_scripts/marason/public/run_all.sh <msnlib|massbank|nplib1>
# Env overrides: workers (CPU workers, default 32), CUDA_VISIBLE_DEVICES (default 0)
#
# Stages are skipped when their output already exists, so the script can be re-run after a failure.
set -euo pipefail

dataset=${1:?usage: run_all.sh <msnlib|massbank|nplib1>}
workers=${workers:-32}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

data_dir=data/spec_datasets/$dataset
res_dir=results/marason_$dataset/split_1_rnd1
ref_dir=$data_dir/closest_neighbors/split_1
config_dir=configs/marason/public

for f in labels.tsv spec_files.hdf5 splits/split_1.tsv magma_outputs subformulae/no_subform.hdf5; do
  [ -e "$data_dir/$f" ] || { echo "Missing $data_dir/$f; run data_scripts/public/build_public_datasets.sh $dataset first"; exit 1; }
done

# 1. Fragment generator (same as the ICEBERG generator)
if [ ! -f "$res_dir/ckpt/gen/best.ckpt" ]; then
  python launcher_scripts/run_from_config.py $config_dir/marason_train_$dataset.yaml
fi

# 2. Predict fragmentation DAGs for every spectrum, then attach observed intensities
if [ ! -f "$res_dir/preds_train_100/tree_preds.hdf5" ]; then
  python launcher_scripts/run_from_config.py $config_dir/marason_gen_predict_train_$dataset.yaml
fi
if [ ! -f "$res_dir/preds_train_100_inten.hdf5" ]; then
  python data_scripts/dag/add_dag_intens.py \
    --pred-dag-path "$res_dir/preds_train_100/tree_preds.hdf5" \
    --true-dag-path "$data_dir/subformulae/no_subform.hdf5" \
    --out-dag-path "$res_dir/preds_train_100_inten.hdf5" \
    --num-workers "$workers" \
    --add-raw
fi

# 3. Retrieval-augmented intensity model; reference spectra are searched in the train fold and saved to $ref_dir
mkdir -p "$ref_dir"
if [ ! -f "$res_dir/ckpt/inten/best.ckpt" ]; then
  python launcher_scripts/run_from_config.py $config_dir/marason_inten_train_$dataset.yaml
fi

# 4. Predict the test fold and report spectrum similarity (cosine, entropy, ...)
mkdir -p "$res_dir/preds"
python src/ms_pred/marason/predict_inten.py \
  --batch-size 64 \
  --dataset-name "$dataset" \
  --split-name split_1.tsv \
  --checkpoint-pth "$res_dir/ckpt/inten/best.ckpt" \
  --save-dir "$res_dir/preds" \
  --gpu \
  --num-workers 0 \
  --magma-dag-folder "$res_dir/preds_train_100/tree_preds.hdf5" \
  --inten-folder "$res_dir/preds_train_100_inten.hdf5" \
  --subset-datasets test_only \
  --binned-out \
  --add-ref \
  --max-ref-count 3 \
  --ref-dir "$ref_dir"

python analysis/spec_pred_eval.py \
  --binned-pred-file "$res_dir/preds/binned_preds.hdf5" \
  --max-peaks 100 \
  --min-inten 0 \
  --formula-dir-name no_subform.hdf5 \
  --dataset "$dataset"

# 5. Retrieval among PubChem isomers (only nplib1 ships candidate tables)
if [ -f "$data_dir/retrieval/cands_df_split_1_50.tsv" ]; then
  python run_scripts/marason/public/run_retrieval.py --dataset "$dataset"
fi
