#!/usr/bin/env bash
# Build a public dataset and train / evaluate MARASON inside a Kaggle notebook.
#
# Usage (from the repository root, inside /kaggle/working):
#   bash run_scripts/marason/public/kaggle_run.sh <msnlib|massbank|nplib1> [setup|data|train|all]   (default: all)
#
# Kaggle settings: Accelerator "GPU T4 x2" (or P100), Internet on.
# Env overrides:
#   MAX_EPOCHS   epochs for both training stages (default 50; the repo configs use 200)
#   RESUME_FROM  repository folder from an earlier notebook output added as input,
#                e.g. /kaggle/input/<your-notebook>/ms-pred; its data and results are copied back first
#   MARASON_OVERRIDES  extra config overrides, see run_all.sh
#
# Data and results stay in the repository (/kaggle/working is kept as notebook output, 20 GB max).
# The Python environment and raw downloads go to /tmp, which is not kept.
set -euo pipefail

dataset=${1:?usage: kaggle_run.sh <msnlib|massbank|nplib1> [setup|data|train|all]}
step=${2:-all}
venv=${VENV:-/tmp/ms-pred-venv}
export workers=${workers:-$(nproc)}
export raw_dir=${raw_dir:-/tmp/raw}

setup() {
  if [ ! -x "$venv/bin/python" ]; then
    pip install -q uv
    # Same environment as the README install, placed outside /kaggle/working
    UV_PROJECT_ENVIRONMENT=$venv uv sync --extra cu124 --python 3.11
  fi
}

activate() {
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
  python -c "import torch; assert torch.cuda.is_available(), 'No GPU: set Accelerator to GPU in the notebook settings'; print('GPU:', torch.cuda.get_device_name(0))"
}

resume() {
  [ -n "${RESUME_FROM:-}" ] || return 0
  for path in data/spec_datasets/$dataset results/marason_$dataset; do
    if [ -e "$RESUME_FROM/$path" ] && [ ! -e "$path" ]; then
      echo "Restoring $path from $RESUME_FROM"
      mkdir -p "$(dirname "$path")"
      cp -r "$RESUME_FROM/$path" "$path"
    fi
  done
}

data() {
  if [ -d "data/spec_datasets/$dataset/magma_outputs" ] && [ -f "data/spec_datasets/$dataset/subformulae/no_subform.hdf5" ]; then
    echo "data/spec_datasets/$dataset is ready"
  else
    bash data_scripts/public/build_public_datasets.sh "$dataset"
  fi
}

train() {
  export MARASON_OVERRIDES="num-workers=$workers num-cpu-workers=$workers num-gpu-workers=1 max-epochs=${MAX_EPOCHS:-50} ${MARASON_OVERRIDES:-}"
  bash run_scripts/marason/public/run_all.sh "$dataset"
  du -sh "data/spec_datasets/$dataset" "results/marason_$dataset" || true
}

case $step in
  setup) setup ;;
  data) setup; activate; resume; data ;;
  train) setup; activate; resume; train ;;
  all) setup; activate; resume; data; train ;;
  *) echo "Unknown step $step (expected setup, data, train or all)"; exit 1 ;;
esac
