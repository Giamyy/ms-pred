# Assign intensities to prediction for next training run
# (NIST20 entries are left commented out; only MassSpecGym msg_simulation, 3 seeds)

# python launcher_scripts/run_from_config.py configs/marason/marason_gen_predict_train_nist20.yaml
python launcher_scripts/run_from_config.py configs/marason/marason_gen_predict_train_msg.yaml

for rnd in 1 2 3; do
python data_scripts/dag/add_dag_intens.py \
	--pred-dag-path  results/marason_msg_simulation/split_rnd${rnd}/preds_train_100/tree_preds.hdf5 \
	--true-dag-path data/spec_datasets/msg_simulation/subformulae/no_subform.hdf5 \
	--out-dag-path results/marason_msg_simulation/split_rnd${rnd}/preds_train_100_inten.hdf5  \
	--num-workers 32 \
	--add-raw
done
