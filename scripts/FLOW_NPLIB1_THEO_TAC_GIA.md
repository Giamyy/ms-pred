# Chạy MARASON trên NPLIB1 theo flow của tác giả

Flow chạy các script MARASON của tác giả (`run_scripts/marason/`, `launcher_scripts/run_from_config.py`) trên tập
**NPLIB1** (`canopus_train_public`, tập con GNPS dùng trong bài ICEBERG), từng bước bằng tay. Cùng 6 bước như
`run_scripts/marason/run_all.sh`.

> **Cần biết trước:**
> - Bài MARASON **không** báo cáo kết quả trên NPLIB1, và nhánh `main` của repo gốc đã ngừng hỗ trợ NPLIB1
>   (README: *"We are retiring the support of the NPLIB1 dataset … in the main branch"*). Repo gốc không có config
>   MARASON cho NPLIB1.
> - Config dùng ở đây (`configs/marason/public/*_nplib1.yaml`) **chép từ config MARASON cho MassSpecGym của tác giả**,
>   chỉ đổi tên dataset, split và thư mục. Siêu tham số giữ nguyên (GGNN, hidden 512, softmax matching, loss entropy…).
> - NPLIB1 **không có năng lượng va chạm**. Các khối MS2 của một phổ được gộp thành một phổ mang nhãn `nan`, như trong
>   bài ICEBERG. Code có một chỉnh sửa nhỏ (`train_inten.py`) để phổ không có năng lượng va chạm tham chiếu lẫn nhau;
>   code gốc sẽ không tìm được tham chiếu nào cho các phổ này.
>
> Vì vậy kết quả là **baseline MARASON tự dựng trên NPLIB1**, không phải tái lập một con số trong bài báo.

---

## 0. Yêu cầu
Linux, GPU NVIDIA (driver hỗ trợ CUDA 12.4 hoặc 11.8), `git`, `gcc`, mạng ra được GitHub, PyPI, `zenodo.org`,
`download.pytorch.org`, `data.dgl.ai`, `data.pyg.org`.

---

## 1. Lấy code và cài môi trường
```bash
cd ~
git clone -b claude/code-explanation-frix3g https://github.com/Giamyy/ms-pred.git ms-pred
cd ~/ms-pred
git log --oneline -1                 # ghi lại commit
bash scripts/setup_env_uv.sh         # = uv sync --extra cu124 --python 3.11 (tự chọn bản CUDA)
source .venv/bin/activate
```
Nhánh này là repo gốc `coleygroup/ms-pred`, cộng thêm phần dựng NPLIB1 (`data_scripts/public/`), config
`configs/marason/public/` và các script chạy. Code mô hình giữ như repo gốc, trừ chỉnh sửa về năng lượng va chạm `nan`
nêu ở trên.

---

## 2. Dựng dữ liệu NPLIB1

**Cách A, trên server (khoảng 1 giờ, phần lớn là MAGMa):**
```bash
workers=$(nproc) bash data_scripts/public/build_public_datasets.sh nplib1
```
Script làm ba việc:
1. Tải `canopus_train_public.zip` từ Zenodo của tác giả ms-pred (record 8433354), lấy `labels.tsv`, `spec_files/`,
   `splits/` và các bảng ứng viên `retrieval/cands_df_split_*_50.tsv`.
2. `create_public_dataset.py nplib1`: chuyển sang `labels.tsv` + `spec_files.hdf5`, **giữ nguyên split chính thức**
   (`split_1/2/3.tsv`) và bảng ứng viên (thêm cột `precursor`, `collision_energies`).
3. `data_scripts/dag/run_magma.sh nplib1` (script gốc): MAGMa (`--max-peaks 50 --ppm-diff 20`) và
   `subformulae/no_subform.hdf5` (`--use-all`).

**Cách B, dùng dữ liệu đã dựng trên Kaggle** (notebook `notebooks/kaggle/kaggle_build_public_data.ipynb`,
`DATASET = "nplib1"`): tải output về rồi chép thư mục `ms-pred/data/spec_datasets/nplib1/` vào
`~/ms-pred/data/spec_datasets/nplib1/`.

**Kiểm tra:**
```bash
ls data/spec_datasets/nplib1 data/spec_datasets/nplib1/splits data/spec_datasets/nplib1/retrieval
cat data/spec_datasets/nplib1/audit.json
```
Phải có `labels.tsv`, `spec_files.hdf5`, `splits/split_1.tsv`, `magma_outputs/magma_tree.hdf5`,
`subformulae/no_subform.hdf5`, `retrieval/cands_df_split_1_50.tsv`.

---

## 3. Chỉnh cho máy
Config nplib1 đã đặt `visible_devices: [0]` (1 GPU) và tự tính phổ tham chiếu ở lần train đầu
(`load-reference: false`, `save-reference: true`). Chỉ cần sửa số worker nếu máy dưới 16 nhân CPU:
```bash
N=$(nproc)
sed -i "s/num-workers: \[16\]/num-workers: [$N]/" configs/marason/public/marason_train_nplib1.yaml configs/marason/public/marason_inten_train_nplib1.yaml
sed -i "s/num-cpu-workers: \[32\]/num-cpu-workers: [$N]/" configs/marason/public/marason_gen_predict_train_nplib1.yaml
mkdir -p data/spec_datasets/nplib1/closest_neighbors/split_1
```
Máy 2 GPU: đổi `visible_devices: [0]` thành `[0, 1]` trong 3 file config. Nếu gặp `CUDA out of memory`: giảm
`batch-size` trong config tương ứng.

---

## 4. Chạy 6 bước (như `run_scripts/marason/run_all.sh`)
Chạy trong `tmux` (`tmux new -s nplib1`), sau `cd ~/ms-pred && source .venv/bin/activate`.

### Bước 1. Train bộ sinh mảnh
```bash
python launcher_scripts/run_from_config.py configs/marason/public/marason_train_nplib1.yaml
```
Đầu ra: `results/marason_nplib1/split_1_rnd1/ckpt/gen/best.ckpt` (seed 1, tối đa 200 epoch, early stopping).

### Bước 2. Dò số mảnh (`02_sweep_gen_thresh.py`): bỏ qua
Bước tuỳ chọn, cần `subformulae/magma_subform_50.hdf5` (chưa dựng cho nplib1). Các bước sau không phụ thuộc bước này.

### Bước 3. Sinh DAG cho tập train và gắn phổ thật
```bash
python launcher_scripts/run_from_config.py configs/marason/public/marason_gen_predict_train_nplib1.yaml

python data_scripts/dag/add_dag_intens.py \
  --pred-dag-path results/marason_nplib1/split_1_rnd1/preds_train_100/tree_preds.hdf5 \
  --true-dag-path data/spec_datasets/nplib1/subformulae/no_subform.hdf5 \
  --out-dag-path results/marason_nplib1/split_1_rnd1/preds_train_100_inten.hdf5 \
  --num-workers $(nproc) \
  --add-raw
```

### Bước 4. Train bộ cường độ (tính phổ tham chiếu lần đầu)
```bash
python launcher_scripts/run_from_config.py configs/marason/public/marason_inten_train_nplib1.yaml
```
- Phổ tham chiếu được tính trong tập train và lưu vào `data/spec_datasets/nplib1/closest_neighbors/split_1/`.
- Đầu ra: `results/marason_nplib1/split_1_rnd1/ckpt/inten/best.ckpt`.
- Nếu chạy lại bước này, đổi sang đọc phổ tham chiếu đã lưu:
  ```bash
  sed -i 's/load-reference: \[false\]/load-reference: [true]/; s/save-reference: \[true\]/save-reference: [false]/' configs/marason/public/marason_inten_train_nplib1.yaml
  ```

### Bước 5. Dự đoán và đánh giá phổ trên tập test
Cùng tham số với `05_predict_marason_inten.py` của tác giả:
```bash
R=results/marason_nplib1/split_1_rnd1
python src/ms_pred/marason/predict_inten.py \
  --batch-size 64 --dataset-name nplib1 --split-name split_1.tsv \
  --checkpoint-pth $R/ckpt/inten/best.ckpt --save-dir $R/preds --gpu --num-workers 0 \
  --magma-dag-folder $R/preds_train_100/tree_preds.hdf5 \
  --inten-folder $R/preds_train_100_inten.hdf5 \
  --subset-datasets test_only --binned-out \
  --add-ref --max-ref-count 3 --ref-dir data/spec_datasets/nplib1/closest_neighbors/split_1

python analysis/spec_pred_eval.py \
  --binned-pred-file $R/preds/binned_preds.hdf5 \
  --max-peaks 100 --min-inten 0 --formula-dir-name no_subform.hdf5 --dataset nplib1
```
Kết quả: `results/marason_nplib1/split_1_rnd1/preds/pred_eval.yaml` (cosine, entropy similarity).

### Bước 6. Retrieval (50 ứng viên PubChem cùng công thức)
```bash
python run_scripts/marason/public/run_retrieval.py --dataset nplib1 --gpu
```
- Script làm theo `run_scripts/marason/run_retrieval_nist.py` của tác giả: khoảng cách `cos`, tối đa 100 mảnh,
  **không dùng phổ tham chiếu** khi xếp hạng (`add_ref = False`, như script NIST).
- Muốn retrieval dùng RAG như script MassSpecGym của tác giả: thêm `--add-ref`. Chọn cách nào thì phải giữ nguyên khi
  so với bản cải tiến.
- Kết quả: `results/marason_nplib1/split_1_rnd1/retrieval_nplib1_split_1_50/rerank_eval_cos.yaml` (top-1/5/10).

---

## 5. Xem kết quả
```bash
grep -E "^avg_(cos_sim|entropy_sim|cos_sim_zero_pep|coverage):" results/marason_nplib1/split_1_rnd1/preds/pred_eval.yaml
grep -E "^avg_top_(1|5|10):" results/marason_nplib1/split_1_rnd1/retrieval_nplib1_split_1_50/rerank_eval_cos.yaml
```

---

## 6. Chạy thêm seed
Thêm vào `iterative_args` của 3 config các mục seed 2, 3 (giống các mục đã comment trong config NIST của tác giả):
- `marason_train_nplib1.yaml`: `save-dir: [split_1_rnd2]`, `seed: [2]`;
- `marason_gen_predict_train_nplib1.yaml`: `checkpoint-pth: [results/marason_nplib1/split_1_rnd2/ckpt/gen/best.ckpt]`,
  `save-dir: [split_1_rnd2/preds_train_100]`;
- `marason_inten_train_nplib1.yaml`: `save-dir: [split_1_rnd2]`, `magma-dag-folder: [results/marason_nplib1/split_1_rnd2/preds_train_100_inten.hdf5]`,
  `seed: [2]` (và đặt `load-reference: [true]`, `save-reference: [false]`);

rồi lặp lại bước 3, 5, 6 với `split_1_rnd2` thay cho `split_1_rnd1`.

Cách nhanh hơn, cùng các lệnh trên: `python scripts/run_marason_baseline.py --datasets nplib1 --seeds 1 2 3`,
tổng hợp bằng `python scripts/summarize_marason_baseline.py --datasets nplib1`.

---

## Ghi chú cho báo cáo
| Điểm | Nội dung |
|---|---|
| Dữ liệu | `canopus_train_public` (Zenodo 8433354 của tác giả ms-pred), split chính thức `split_1.tsv`, bảng 50 ứng viên của tác giả |
| Năng lượng va chạm | Không có; phổ MS2 gộp thành một phổ (`nan`), như bài ICEBERG |
| Config | Config MARASON cho MassSpecGym của tác giả, đổi sang nplib1 |
| Chỉnh code | Phổ năng lượng `nan` được tham chiếu tới nhau (`train_inten.py`) |
| Retrieval | Theo script NIST của tác giả (cos, không RAG), trừ khi dùng `--add-ref` |
| So sánh tài liệu | Bài ICEBERG (Goldman et al. 2024) có số liệu trên NPLIB1; bài MARASON thì không |
