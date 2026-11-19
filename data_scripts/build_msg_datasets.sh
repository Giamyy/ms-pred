#!/usr/bin/env bash
# Rebuild every MassSpecGym dataset folder used by the repo from the public
# Hugging Face release (https://huggingface.co/datasets/roman-bushuiev/MassSpecGym).
#
#   data/spec_datasets/msg_all         all entries (+ magma_outputs, subformulae)
#   data/spec_datasets/msg_known_ce    entries with known collision energy (links msg_all resources)
#   data/spec_datasets/msg             simulation-challenge subset (links msg_all resources)
#   data/spec_datasets/msg_simulation  official simulation subset built from msg
#
# Usage: bash data_scripts/build_msg_datasets.sh
# Env overrides: workers, source_table, skip_magma=1
set -euo pipefail

workers=${workers:-32}
source_table=${source_table:-https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/MassSpecGym1.5.tsv}
data_root=data/spec_datasets

# 1. Full MassSpecGym: labels, splits, spec_files.hdf5, retrieval candidates
python data_scripts/create_msg_dataset.py \
  --source-table "$source_table" \
  --subset all \
  --output-dataset $data_root/msg_all \
  --num-workers $workers \
  --overwrite

# 2. MAGMa fragment annotations + no_subform.hdf5 (slow; computed once on msg_all)
if [ "${skip_magma:-0}" != "1" ]; then
  workers=$workers bash data_scripts/dag/run_magma.sh msg_all
fi

# 3. Known-CE subset for the first stage of run_scripts/iceberg/msg_all
python data_scripts/create_msg_dataset.py \
  --source-table "$source_table" \
  --subset known_ce \
  --output-dataset $data_root/msg_known_ce \
  --link-resources-from $data_root/msg_all \
  --formula-candidates '' --mass-candidates '' \
  --overwrite

# 4. Base msg dataset (GLACIER / MARASON configs, source of msg_simulation)
python data_scripts/create_msg_dataset.py \
  --source-table "$source_table" \
  --subset simulation \
  --output-dataset $data_root/msg \
  --link-resources-from $data_root/msg_all \
  --num-workers $workers \
  --overwrite

# 5. Official simulation-challenge dataset
python data_scripts/create_msg_simulation_dataset.py \
  --source-table "$source_table" \
  --source-dataset $data_root/msg \
  --output-dataset $data_root/msg_simulation \
  --force-output \
  --overwrite \
  --audit-out $data_root/msg_simulation/audit.json
