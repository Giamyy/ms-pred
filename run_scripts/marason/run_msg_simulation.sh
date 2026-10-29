#!/usr/bin/env bash
# Run the MARASON pipeline on MassSpecGym msg_simulation (3 seeds) with the authors' scripts.
# Usage (from the repo root, inside the activated environment):
#     bash run_scripts/marason/run_msg_simulation.sh
# A step is skipped when all of its outputs already exist, so re-running the
# command continues from the first unfinished step.

set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"

DATA=data/spec_datasets/msg_simulation
RES=results/marason_msg_simulation
SEEDS="1 2 3"

say() { echo; echo "========== $* =========="; }

all_exist() {  # all_exist <path pattern with SEED>
    for s in $SEEDS; do
        [ -e "${1//SEED/$s}" ] || return 1
    done
}

say "Kiem tra du lieu"
for f in labels.tsv spec_files.hdf5 splits/split.tsv retrieval/cands_df_test_formula_256.tsv; do
    if [ ! -e "$DATA/$f" ]; then
        echo "THIEU FILE: $DATA/$f"
        exit 1
    fi
done
python -c "import torch; n = torch.cuda.device_count(); print('So GPU:', n); assert n > 0, 'Khong thay GPU'"

say "Buoc 0: MAGMa + subformula"
if [ -e "$DATA/magma_outputs/magma_tree.hdf5" ] && [ -e "$DATA/subformulae/no_subform.hdf5" ]; then
    echo "Da co, bo qua."
else
    dataset=msg_simulation workers=16 bash data_scripts/dag/run_magma.sh
fi

say "Buoc 1: train mo hinh sinh manh"
if all_exist "$RES/split_rndSEED/ckpt/gen/best.ckpt"; then
    echo "Da co, bo qua."
else
    bash run_scripts/marason/01_run_marason_gen_train.sh
fi

say "Buoc 3: du doan DAG + gan cuong do"
if all_exist "$RES/split_rndSEED/preds_train_100_inten.hdf5"; then
    echo "Da co, bo qua."
else
    bash run_scripts/marason/03_run_marason_gen_predict.sh
fi

say "Buoc 4: train mo hinh cuong do"
if all_exist "$RES/split_rndSEED/ckpt/inten/best.ckpt"; then
    echo "Da co, bo qua."
else
    bash run_scripts/marason/04_train_marason_inten.sh
fi

say "Buoc 5: du doan + cham diem pho test"
if all_exist "$RES/split_rndSEED/preds/pred_eval.yaml"; then
    echo "Da co, bo qua."
else
    python run_scripts/marason/05_predict_marason_inten.py
fi

say "Buoc 6: retrieval"
RET="$RES/split_rndSEED/retrieval_msg_simulation_test_formula_256"
if all_exist "$RET/rerank_eval_entropy.yaml"; then
    echo "Da co, bo qua."
else
    bash run_scripts/marason/06_run_retrieval.sh
fi

say "Buoc 7: khoang tin cay bootstrap"
for s in $SEEDS; do
    python analysis/msg_bootstrap.py --path "${RET//SEED/$s}/rerank_eval_entropy.yaml" --dist-fn entropy
done

say "XONG. Ket qua:"
for s in $SEEDS; do
    echo "--- seed $s"
    grep -E "^avg_(cos_sim|entropy_sim):" "$RES/split_rnd$s/preds/pred_eval.yaml" || true
    grep -E "^avg_top_(1|5|20):" "${RET//SEED/$s}/rerank_eval_entropy.yaml" || true
done
