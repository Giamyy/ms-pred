#!/usr/bin/env bash
# Download public MS/MS libraries and build MARASON-ready datasets, independent of MassSpecGym.
#
# Usage: bash data_scripts/public/build_public_datasets.sh [msnlib] [massbank] [nplib1]   (default: all)
# Env overrides: workers (default 32), skip_magma=1, plus the version pins in download_public_datasets.sh
#
# Produces data/spec_datasets/<dataset>/ with labels.tsv, spec_files.hdf5, splits/,
# magma_outputs/ and subformulae/no_subform.hdf5 (and retrieval/ for nplib1).
set -euo pipefail

workers=${workers:-32}
raw_dir=${raw_dir:-data/raw}
datasets=("$@")
[ ${#datasets[@]} -eq 0 ] && datasets=(msnlib massbank nplib1)

raw_dir=$raw_dir bash data_scripts/public/download_public_datasets.sh "${datasets[@]}"

for dataset in "${datasets[@]}"; do
  case $dataset in
    msnlib) inputs=("$raw_dir"/msnlib/*_pos_ms2.mgf) ;;
    massbank) inputs=("$raw_dir/massbank/MassBank_NISTformat.msp") ;;
    nplib1) inputs=("$raw_dir/nplib1/canopus_train_public") ;;
  esac
  python data_scripts/public/create_public_dataset.py "$dataset" \
    --input "${inputs[@]}" \
    --output-dataset "data/spec_datasets/$dataset" \
    --num-workers "$workers" \
    --overwrite

  # MAGMa fragment annotations + no_subform.hdf5 (the slow step)
  if [ "${skip_magma:-0}" != "1" ]; then
    workers=$workers bash data_scripts/dag/run_magma.sh "$dataset"
  fi
done
