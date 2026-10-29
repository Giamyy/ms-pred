#!/usr/bin/env bash
# Train and evaluate MARASON on data/spec_datasets/msg_simulation on a local GPU machine.
#
# Same steps as run_scripts/marason/run_all.sh (without MAGMA, which the dataset already has):
#   B1 train fragment generator      configs/marason/marason_train_msg.yaml
#   B2 predict DAGs                  configs/marason/marason_gen_predict_train_msg.yaml
#   B3 add true intensities          data_scripts/dag/add_dag_intens.py
#   B4 references + train intensity  configs/marason/marason_inten_train_msg.yaml
#   B5 predict + evaluate test       as run_scripts/marason/05_predict_marason_inten.py
#   B6 retrieval                     as run_scripts/marason/run_retrieval_msg.py
#
# The configs are copied to results/marason_msg_simulation/local_configs/ with this machine's
# GPUs / workers / batch sizes; the files in configs/ stay unchanged. Each finished step writes
# a .done_B* marker, so re-running the script resumes after the last finished step.
#
#   bash scripts/train_marason_msg.sh                 # all steps
#   STEPS="B5 B6" bash scripts/train_marason_msg.sh   # only some steps
#   GPUS=0,1 GEN_BATCH=16 bash scripts/train_marason_msg.sh
#   FORCE=1 STEPS=B4 bash scripts/train_marason_msg.sh   # redo a finished step

set -euo pipefail

export GPUS="${GPUS:-0}"                           # CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1
NCPU="$(nproc)"
NUM_WORKERS="${NUM_WORKERS:-$(( NCPU < 16 ? NCPU : 16 ))}"
GEN_BATCH="${GEN_BATCH:-32}"                       # config default 32
INTEN_BATCH="${INTEN_BATCH:-16}"                   # config default 16
GEN_PRED_BATCH="${GEN_PRED_BATCH:-8}"              # config default 8
INTEN_PRED_BATCH="${INTEN_PRED_BATCH:-64}"         # 05_predict_marason_inten.py default 64
RETRIEVAL_BATCH="${RETRIEVAL_BATCH:-10}"           # run_retrieval_msg.py default 10
MAX_EPOCHS="${MAX_EPOCHS:-200}"                    # config default 200 (early stopping usually ends sooner)
STEPS="${STEPS:-B1 B2 B3 B4 B5 B6}"
FORCE="${FORCE:-0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

DATASET=msg_simulation
DATA=data/spec_datasets/$DATASET
RES=results/marason_$DATASET
RUN=$RES/split_rnd1
REF_DIR=data/$DATASET/closest_neighbors/infinite
CFG_DIR=$RES/local_configs
LOG_DIR=$RES/logs
mkdir -p "$CFG_DIR" "$LOG_DIR" "$RUN"

for f in labels.tsv spec_files.hdf5 splits/split.tsv magma_outputs/magma_tree.hdf5 subformulae/no_subform.hdf5; do
    [[ -e "$DATA/$f" ]] || { echo "Thiếu $DATA/$f" >&2; exit 1; }
done
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'torch không thấy GPU (kiểm tra driver / môi trường)')"
echo "GPU: $GPUS | workers: $NUM_WORKERS | batch gen/inten: $GEN_BATCH/$INTEN_BATCH | epochs: $MAX_EPOCHS"
echo "Các bước: $STEPS"

want() { [[ " $STEPS " == *" $1 "* ]]; }

# run_step NAME COMMAND...: run unless done, log to $LOG_DIR/NAME.log, mark done on success
run_step() {
    local name=$1; shift
    want "$name" || return 0
    if [[ -f "$RUN/.done_$name" && "$FORCE" != 1 ]]; then
        echo "== $name: đã xong trước đó, bỏ qua (FORCE=1 để chạy lại)"
        return 0
    fi
    # Later steps depend on this one: their old results are no longer valid
    local later=0 s
    for s in B1 B2 B3 B4 B5 B6; do
        (( later )) && rm -f "$RUN/.done_$s"
        [[ $s == "$name" ]] && later=1
    done
    echo "== $name: bắt đầu $(date '+%F %T'), log: $LOG_DIR/$name.log"
    local t0=$SECONDS
    "$@" 2>&1 | tee "$LOG_DIR/$name.log"
    touch "$RUN/.done_$name"
    echo "== $name: xong sau $(( (SECONDS - t0) / 60 )) phút"
}

# local_config SRC DST [key=value ...]: copy a launcher config with this machine's settings
local_config() {
    python - "$@" <<'EOF'
import os, sys, yaml
src, dst, *pairs = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg["launcher_args"]["visible_devices"] = [int(g) for g in os.environ["GPUS"].split(",")]
for pair in pairs:
    key, value = pair.split("=", 1)
    cfg["universal_args"][key] = [yaml.safe_load(value)]
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF
}

b1() {
    local_config configs/marason/marason_train_msg.yaml "$CFG_DIR/train_gen.yaml" \
        num-workers="$NUM_WORKERS" batch-size="$GEN_BATCH" max-epochs="$MAX_EPOCHS"
    python launcher_scripts/run_from_config.py "$CFG_DIR/train_gen.yaml"
    [[ -f "$RUN/ckpt/gen/best.ckpt" ]] || { echo "Không thấy $RUN/ckpt/gen/best.ckpt" >&2; return 1; }
}

b2() {
    local_config configs/marason/marason_gen_predict_train_msg.yaml "$CFG_DIR/gen_predict.yaml" \
        num-cpu-workers="$NUM_WORKERS" num-gpu-workers="$(( NUM_WORKERS < 4 ? NUM_WORKERS : 4 ))" \
        batch-size="$GEN_PRED_BATCH"
    python launcher_scripts/run_from_config.py "$CFG_DIR/gen_predict.yaml"
    [[ -f "$RUN/preds_train_100/tree_preds.hdf5" ]] || { echo "Không thấy tree_preds.hdf5" >&2; return 1; }
}

b3() {
    python data_scripts/dag/add_dag_intens.py \
        --pred-dag-path "$RUN/preds_train_100/tree_preds.hdf5" \
        --true-dag-path "$DATA/subformulae/no_subform.hdf5" \
        --out-dag-path "$RUN/preds_train_100_inten.hdf5" \
        --num-workers "$NUM_WORKERS" --add-raw
}

b4() {
    # References (RAG neighbours) are computed and saved on the first run, loaded afterwards
    local load=true save=false
    if [[ ! -f "$REF_DIR/train_ref.npy" || ! -f "$REF_DIR/db_specs.npz" ]]; then
        load=false save=true
        mkdir -p "$REF_DIR"
        echo "Chưa có phổ tham chiếu trong $REF_DIR: tính và lưu trong lần train này"
    fi
    local_config configs/marason/marason_inten_train_msg.yaml "$CFG_DIR/train_inten.yaml" \
        num-workers="$NUM_WORKERS" batch-size="$INTEN_BATCH" max-epochs="$MAX_EPOCHS" \
        load-reference="$load" save-reference="$save"
    python launcher_scripts/run_from_config.py "$CFG_DIR/train_inten.yaml"
    [[ -f "$RUN/ckpt/inten/best.ckpt" ]] || { echo "Không thấy $RUN/ckpt/inten/best.ckpt" >&2; return 1; }
}

b5() {
    CUDA_VISIBLE_DEVICES=$GPUS python src/ms_pred/marason/predict_inten.py \
        --batch-size "$INTEN_PRED_BATCH" \
        --dataset-name "$DATASET" \
        --split-name split.tsv \
        --checkpoint-pth "$RUN/ckpt/inten/best.ckpt" \
        --save-dir "$RUN/preds" \
        --gpu \
        --num-workers 0 \
        --magma-dag-folder "$RUN/preds_train_100/tree_preds.hdf5" \
        --inten-folder "$RUN/preds_train_100_inten.hdf5" \
        --subset-datasets test_only \
        --binned-out \
        --add-ref \
        --max-ref-count 3 \
        --ref-dir "$REF_DIR"
    python analysis/spec_pred_eval.py \
        --binned-pred-file "$RUN/preds/binned_preds.hdf5" \
        --max-peaks 100 \
        --min-inten 0 \
        --formula-dir-name no_subform.hdf5 \
        --dataset "$DATASET"
}

b6() {
    local ran=0
    for kind in formula mass; do
        local table="$DATA/retrieval/cands_df_test_${kind}_256.tsv"
        if [[ ! -f "$table" ]]; then
            echo "Không có $table, bỏ qua retrieval theo $kind"
            continue
        fi
        local out="$RUN/retrieval_${DATASET}_test_${kind}_256"
        mkdir -p "$out"
        CUDA_VISIBLE_DEVICES=$GPUS python src/ms_pred/marason/predict_smis.py \
            --batch-size "$RETRIEVAL_BATCH" \
            --dataset-name "$DATASET" \
            --sparse-out \
            --sparse-k 250 \
            --max-nodes 250 \
            --split-name split.tsv \
            --gen-checkpoint "$RUN/ckpt/gen/best.ckpt" \
            --inten-checkpoint "$RUN/ckpt/inten/best.ckpt" \
            --save-dir "$out" \
            --ref-dir "$REF_DIR" \
            --dataset-labels "$table" \
            --num-cpu-workers "$(( NUM_WORKERS < 4 ? NUM_WORKERS : 4 ))" \
            --num-gpu-workers 0 \
            --magma-dag-folder "$RUN/preds_train_100_inten.hdf5" \
            --adduct-shift \
            --gpu \
            --embed-elem-group \
            --add-ref --max-ref-count 3 \
            --binned-out
        python src/ms_pred/retrieval/retrieval_benchmark.py \
            --dataset "$DATASET" \
            --true-labels "$DATA/labels.tsv" \
            --full-labels "$table" \
            --formula-dir-name no_subform.hdf5 \
            --pred-file "$out/binned_preds.hdf5" \
            --dist-fn entropy \
            --binned-pred
        ran=1
    done
    (( ran )) || { echo "Không có bảng retrieval nào trong $DATA/retrieval" >&2; return 1; }
}

run_step B1 b1
run_step B2 b2
run_step B3 b3
run_step B4 b4
run_step B5 b5
run_step B6 b6
echo "Hoàn tất. Kết quả trong $RUN, log trong $LOG_DIR"
