# Chạy MARASON trên MassSpecGym theo quy trình của tác giả

Tài liệu này đi theo `README.md` của repo gốc https://github.com/coleygroup/ms-pred (mục *Install & setup*,
*MassSpecGym Data Processing*, *ICEBERG and GLACIER Processing* và *Experiments → MARASON*). Chỉ dùng script của
tác giả, chạy cho tập MassSpecGym `msg_simulation`.

Mỗi bước ghi rõ: lệnh theo README, những chỗ phải sửa để chạy được khi chỉ có MassSpecGym, và đầu ra.
Tập NIST'20 là dữ liệu thương mại nên không nằm trong tài liệu này.

> Phiên bản README dùng để viết tài liệu: commit `bccbed6` của `coleygroup/ms-pred`
> ("Support all MassSpecGym dataset strategies from Hugging Face").

---

## 0. Tổng quan

Theo README, MARASON gồm hai phần: **bộ sinh mảnh** (giống ICEBERG) và **bộ dự đoán cường độ dùng RAG**.
Quy trình đầy đủ được tác giả gói trong `run_scripts/marason/run_all.sh`:

```bash
. data_scripts/dag/run_magma.sh                     # MAGMa + subformula
. run_scripts/marason/01_run_marason_gen_train.sh   # 1. train bộ sinh mảnh
python run_scripts/marason/02_sweep_gen_thresh.py   # 2. dò số mảnh (tuỳ chọn)
. run_scripts/marason/03_run_marason_gen_predict.sh # 3. sinh DAG cho tập train + gắn phổ thật
. run_scripts/marason/04_train_marason_inten.sh     # 4. train bộ cường độ
python run_scripts/marason/05_predict_marason_inten.py  # 5. dự đoán + đánh giá phổ
. run_scripts/marason/06_run_retrieval.sh           # 6. retrieval
```

**Không chạy nguyên `run_all.sh` khi chỉ có MassSpecGym.** Lý do:
- `run_magma.sh` không truyền tên dataset thì mặc định chạy `nist23`.
- `01_…sh`, `03_…sh` và `04_…sh` chạy cả config NIST lẫn MSG. Lệnh NIST sẽ lỗi vì thiếu dữ liệu.

Vì vậy, các mục dưới đây chạy **từng lệnh phần MSG** của các script đó.

---

## 1. Phần cứng

README ghi: *"You need two GPUs with at least 24GB RAM to train MARASON (we used NVIDIA A5000 for development).
If you are trying to train the model on a smaller GPU, try cutting down the hidden_size to 256. Note that changing
training parameters will affect the model performance."*

- Các config MSG đặt sẵn `visible_devices: [0, 1]` (2 GPU). Nếu máy chỉ có 1 GPU, sửa thành `[0]` trong
  `configs/marason/marason_train_msg.yaml`, `marason_gen_predict_train_msg.yaml`, `marason_inten_train_msg.yaml`.
- `num-workers: 16` và `num-cpu-workers: 32` được đặt cho máy nhiều nhân. Nếu máy ít nhân hơn, giảm xuống bằng số nhân (`nproc`).
- Mọi thay đổi so với config gốc (số GPU, worker, batch size, `hidden-size`) cần ghi vào báo cáo.

---

## 2. Cài đặt (README: *Install & setup*)

```bash
git clone https://github.com/coleygroup/ms-pred.git
cd ms-pred
curl -LsSf https://astral.sh/uv/install.sh | sh     # cài uv nếu chưa có
uv sync --extra cu124                               # CUDA 12.4; CUDA 11.8 thì dùng --extra cu118
source .venv/bin/activate
```
- Cần có `gcc`. README ghi thời gian cài khoảng 10 phút.
- Các extra `cpu`, `cu118`, `cu124` loại trừ nhau, chỉ chọn một.
- Kiểm tra: `python -c "import torch, dgl; print(torch.__version__, torch.cuda.is_available())"`.

---

## 3. Dữ liệu

### 3.1. Dựng dataset từ Hugging Face (README: *Build directly from Hugging Face*)
```bash
python data_scripts/create_msg_simulation_dataset.py --strategy msg_simulation \
  --audit-out data/spec_datasets/msg_simulation/audit.json
```
- Script đọc MassSpecGym 1.5 TSV từ Hugging Face, chọn phổ có `simulation_challenge == True` và đã biết năng lượng va chạm,
  rồi giữ nguyên `fold` chính thức.
- **Đầu ra** trong `data/spec_datasets/msg_simulation/`: `labels.tsv`, `spec_files.hdf5`, `splits/split.tsv`, các file debug,
  và `audit.json`.
- **Kiểm tra `audit.json`:** theo README, kỳ vọng **119.029 phổ duy nhất** và **0 nhãn imputed**.
  `pending_resources` liệt kê những gì còn thiếu (MAGMa, subformula, bảng ứng viên), được làm ở các mục 3.2–3.4.
- Muốn cố định phiên bản dữ liệu, tải TSV về rồi thêm `--source-table /đường/dẫn/MassSpecGym1.5.tsv`.

### 3.2. MAGMa + subformula (README: *ICEBERG and GLACIER Processing*)
```bash
workers=32 bash data_scripts/dag/run_magma.sh msg_simulation
```
- **Bắt buộc truyền `msg_simulation`**, nếu không script chạy mặc định cho `nist23`.
- Script chạy `src/ms_pred/magma/run_magma.py` (`--max-peaks 50 --ppm-diff 20`), ra `magma_outputs/magma_tree.hdf5`.
  Sau đó, nếu chưa có `subformulae/no_subform.hdf5`, script chạy
  `data_scripts/forms/01_assign_subformulae.py --use-all` để tạo file đó.
- `workers` là số tiến trình CPU, mặc định 32. Đây là bước CPU nặng: trên 4 nhân, đo được khoảng 24 giờ cho 119.029 phổ.
- Khoảng 0,2% số phổ bị bỏ qua (MAGMa không sinh được mảnh hoặc phổ không có đỉnh dùng được). Đây là hành vi bình thường.

### 3.3. Subformula `magma_subform_50` (chỉ cần cho bước 2, tuỳ chọn)
Bước 2 (`02_sweep_gen_thresh.py`) đánh giá với `subformulae/magma_subform_50.hdf5`. `run_magma.sh` không tạo file này.
Tạo theo lệnh thứ hai trong `data_scripts/all_assign_subform.sh`, đổi dataset sang `msg_simulation`:
```bash
python data_scripts/forms/01_assign_subformulae.py \
  --data-dir data/spec_datasets/msg_simulation/ \
  --labels-file data/spec_datasets/msg_simulation/labels.tsv \
  --use-magma --mass-diff-thresh 20 \
  --output-dir magma_subform_50.hdf5 \
  --num-workers 32
```

### 3.4. Bảng ứng viên retrieval
- `run_scripts/marason/run_retrieval_msg.py` dùng bảng `cands_df_test_formula_256.tsv`: mỗi phổ test có tối đa 256 phân tử
  cùng công thức. Tải bảng này từ Dropbox của tác giả (link trong README, mục *Pretrained ICEBERG 2.1 Model Weights on
  MassSpecGym*), rồi đặt vào `data/spec_datasets/msg/retrieval/`.
- Bảng của tác giả được dựng cho toàn bộ `msg`. Lọc lại chỉ giữ các phổ của `msg_simulation`, đặt vào thư mục của dataset này:
```bash
mkdir -p data/spec_datasets/msg_simulation/retrieval
python - <<'EOF'
import pandas as pd
specs = set(pd.read_csv("data/spec_datasets/msg_simulation/labels.tsv", sep="\t")["spec"].astype(str))
df = pd.read_csv("data/spec_datasets/msg/retrieval/cands_df_test_formula_256.tsv", sep="\t")
df = df[df["spec"].astype(str).isin(specs)]
df.to_csv("data/spec_datasets/msg_simulation/retrieval/cands_df_test_formula_256.tsv", sep="\t", index=False)
print(len(df), "dòng,", df["spec"].nunique(), "phổ")
EOF
```

### 3.5. Kết quả sau mục 3
```
data/spec_datasets/msg_simulation/
├── labels.tsv
├── spec_files.hdf5
├── splits/split.tsv
├── magma_outputs/magma_tree.hdf5
├── subformulae/no_subform.hdf5            (+ magma_subform_50.hdf5 nếu làm mục 3.3)
├── retrieval/cands_df_test_formula_256.tsv
└── audit.json
```

---

## 4. Train và đánh giá (README: *Experiments → MARASON*)

Mọi lệnh chạy từ thư mục gốc `ms-pred`, sau khi đã `source .venv/bin/activate`. `launcher_scripts/run_from_config.py`
đọc file YAML, ghép tham số thành lệnh `python src/ms_pred/marason/...`, đặt `CUDA_VISIBLE_DEVICES` theo `visible_devices`,
rồi chạy. Thêm `--dry` để chỉ in lệnh mà không chạy. Kết quả nằm trong `results/marason_msg_simulation/`.

### Bước 1. Train bộ sinh mảnh (`01_run_marason_gen_train.sh`, dòng MSG)
```bash
python launcher_scripts/run_from_config.py configs/marason/marason_train_msg.yaml
```
- Gọi `src/ms_pred/marason/train_gen.py`. Mô hình học từ cây MAGMa `magma_outputs/`.
- Siêu tham số chính trong config: GGNN 6 lớp, `hidden-size 512`, `pe-embed-k 14`, `learning-rate 0.000996`,
  `lr-decay-rate 0.7214`, `dropout 0.2`, `batch-size 32`, `max-epochs 200`, `seed 1`.
- Early stopping theo `val_loss`, patience 5.
- **Đầu ra:** `results/marason_msg_simulation/split_rnd1/ckpt/gen/best.ckpt`

### Bước 2. Dò số mảnh cần sinh (`02_sweep_gen_thresh.py`, tuỳ chọn)
```bash
python run_scripts/marason/02_sweep_gen_thresh.py
```
- Với mỗi `max_nodes` trong {10, 20, 30, 40, 50, 100, 200, 300, 500, 1000}, script sinh DAG cho tập test, chuyển DAG sang
  subformula (`data_scripts/dag/dag_to_subform.py`), rồi so với `magma_subform_50.hdf5` bằng `analysis/form_pred_eval.py`.
  Mục đích là xem nên giữ bao nhiêu mảnh. Các bước sau dùng 100.
- **Cần sửa trong file:** `list_devices = [0, 1]` theo số GPU, `workers = 32` theo số nhân.
  Cần đã làm mục 3.3.
- **Đầu ra:** `split_rnd1/inten_thresh_sweep/summary.tsv`
- `run_all.sh` ghi chú bước này có thể bỏ nếu không cần phần đánh giá.

### Bước 3. Sinh DAG cho tập train và gắn phổ thật (`03_run_marason_gen_predict.sh`, phần MSG)
```bash
python launcher_scripts/run_from_config.py configs/marason/marason_gen_predict_train_msg.yaml

python data_scripts/dag/add_dag_intens.py \
  --pred-dag-path results/marason_msg_simulation/split_rnd1/preds_train_100/tree_preds.hdf5 \
  --true-dag-path data/spec_datasets/msg_simulation/subformulae/no_subform.hdf5 \
  --out-dag-path results/marason_msg_simulation/split_rnd1/preds_train_100_inten.hdf5 \
  --num-workers 32 \
  --add-raw
```
- Lệnh 1 gọi `predict_gen.py`, sinh DAG tối đa 100 mảnh (`max-nodes 100`, `threshold 0.0`) cho mọi phân tử.
  Đầu ra: `split_rnd1/preds_train_100/tree_preds.hdf5`.
- Lệnh 2 gắn phổ thật (danh sách đỉnh `[m/z, cường độ]` trong `no_subform.hdf5`) vào từng DAG dưới khoá `raw_spec`.
  Đầu ra: `split_rnd1/preds_train_100_inten.hdf5`. File này vừa là đáp án để train bước 4, vừa là kho phổ tham chiếu.
- **Lưu ý khi chạy nguyên file `.sh`:** khối `add_dag_intens.py` của `nist20/split_1_rnd1` trong file chỉ comment dòng
  đầu (`# python ... \`). Các dòng tham số phía sau không được comment, nên bash sẽ báo `command not found`.
  Lỗi này không làm hỏng phần MSG, nhưng chạy riêng hai lệnh trên thì gọn hơn.

### Bước 4. Train bộ cường độ (`04_train_marason_inten.sh`, dòng MSG)
**Lần chạy đầu phải tính phổ tham chiếu.** Config đặt sẵn `load-reference: [true]`, nghĩa là đọc phổ tham chiếu đã tính
trước ở `data/msg_simulation/closest_neighbors/infinite`. Lần đầu chưa có thư mục này, nên phải làm như README hướng dẫn:

1. Tạo thư mục (`train_inten.py` ghi file vào đó nhưng không tự tạo thư mục):
   ```bash
   mkdir -p data/msg_simulation/closest_neighbors/infinite
   ```
2. Trong `configs/marason/marason_inten_train_msg.yaml`, đặt:
   ```yaml
   load-reference: [false]
   save-reference: [true]
   ```
3. Chạy:
   ```bash
   python launcher_scripts/run_from_config.py configs/marason/marason_inten_train_msg.yaml
   ```
4. Từ lần sau, đặt lại `load-reference: [true]` và `save-reference: [false]` để bỏ qua bước tìm tham chiếu.

- Gọi `src/ms_pred/marason/train_inten.py` với `magma-dag-folder = split_rnd1/preds_train_100_inten.hdf5`.
- **Phổ tham chiếu:** với mỗi phổ, tìm các phân tử **trong tập train** gần nhất theo khoảng cách Jaccard của Morgan
  fingerprint. Ứng viên khác adduct, khác máy đo, hoặc trùng hệt cấu trúc bị loại. Khi bằng nhau thì ưu tiên năng lượng
  va chạm gần hơn. `infinite` nghĩa là không giới hạn chênh lệch năng lượng va chạm.
- **Mô hình:** GGNN 4 lớp, `hidden-size 512`, `multi-gnn true` (GNN riêng cho phân tử tham chiếu), `matching-method softmax`
  (`softmax-tau 0.02`), `max-ref-count 3`, `frag-set-layers 3`, `loss-fn entropy`, `learning-rate 0.0002`,
  `lr-decay-rate 0.825`, `dropout 0.1`, `batch-size 16`, `max-epochs 200`. Early stopping patience 5.
- **Đầu ra:** `split_rnd1/ckpt/inten/best.ckpt`, `split_rnd1/args.yaml`, và các file tham chiếu trong
  `data/msg_simulation/closest_neighbors/infinite/` (`train_ref.npy`, `db_specs.npz`, `train.csv`, …).
- Đây là bước lâu nhất và tốn VRAM nhất. Nếu GPU nhỏ, README khuyên giảm `hidden-size` xuống 256.

### Bước 5. Dự đoán và đánh giá phổ (`05_predict_marason_inten.py`)
```bash
python run_scripts/marason/05_predict_marason_inten.py
```
- **Cần sửa trong file:** `devices = ",".join(["0", "1", "2"])` theo số GPU, ví dụ `["0"]`. Mục `nist20` trong
  `test_entries` tự bỏ qua khi không thấy checkpoint.
- Với `msg_simulation`, script gọi `src/ms_pred/marason/predict_inten.py` (`--subset-datasets test_only --binned-out
  --add-ref --max-ref-count 3 --ref-dir data/msg_simulation/closest_neighbors/infinite`), rồi gọi
  `analysis/spec_pred_eval.py` (`--max-peaks 100 --min-inten 0 --formula-dir-name no_subform.hdf5`).
- **Đầu ra:** `split_rnd1/preds/binned_preds.hdf5` và **`split_rnd1/preds/pred_eval.yaml`** (cosine / entropy similarity
  trên tập test, kèm các bảng chia nhóm `pred_eval_grouped_*.tsv`).

### Bước 6. Retrieval (`06_run_retrieval.sh` → `run_retrieval_msg.py`)
`06_run_retrieval.sh` chạy `run_retrieval_msg.py` và `run_retrieval_nist.py`. Chỉ cần chạy bản MSG.

**`run_retrieval_msg.py` phải sửa trước khi chạy.** File tìm checkpoint và phổ tham chiếu ở các đường dẫn không khớp với
nơi các bước 1 và 4 lưu:

| Trong script | Nơi bước 1/4 thực sự lưu |
|---|---|
| `results/marason_inten_msg/split_rnd1/version_0/best.ckpt` | `results/marason_msg_simulation/split_rnd1/ckpt/inten/best.ckpt` |
| bộ sinh mảnh: `<thư mục chứa preds_train_100_inten.hdf5>/version_0/best.ckpt` | `results/marason_msg_simulation/split_rnd1/ckpt/gen/best.ckpt` |
| `ref_dir: data/msg/closest_neighbors/infinite` | `data/msg_simulation/closest_neighbors/infinite` |
| `dataset: msg` (cần `data/spec_datasets/msg/subformulae/…`) | `msg_simulation` |

Các chỗ sửa trong `run_scripts/marason/run_retrieval_msg.py`:
```python
devices = [0]          # theo số GPU (gốc: [0, 1])

test_entries = [
    {
        "dataset": "msg_simulation",
        "results_name": "msg_simulation",
        "test_split": "test_formula",
        "split_name": "split.tsv",
        "candidate_table": "data/spec_datasets/msg_simulation/retrieval/cands_df_test_formula_256.tsv",
        "true_labels": "data/spec_datasets/msg_simulation/labels.tsv",
        "inten_folder": "split_rnd1",
        "max_k": 256,
        "ref_dir": "data/msg_simulation/closest_neighbors/infinite",
    },
]
```
và trong vòng lặp:
```python
    inten_dir = Path(f"results/marason_{results_name}")                 # gốc: f"results/marason_inten_{results_name}"
    inten_model = inten_dir / inten_split / "ckpt/inten/best.ckpt"      # gốc: "version_0/best.ckpt"
    ...
    gen_model = form_folder.parent / "ckpt/gen/best.ckpt"               # gốc: "version_0/best.ckpt"
```
Rồi chạy:
```bash
python run_scripts/marason/run_retrieval_msg.py
```
- Script gọi `src/ms_pred/marason/predict_smis.py`: với mỗi ứng viên, sinh DAG tối đa 250 mảnh rồi dự đoán phổ
  (`--sparse-out --sparse-k 250 --binned-out --add-ref --max-ref-count 3`). Sau đó gọi
  `src/ms_pred/retrieval/retrieval_benchmark.py` để xếp hạng ứng viên theo khoảng cách `entropy` so với phổ thật.
- **Đầu ra:** `results/marason_msg_simulation/split_rnd1/retrieval_msg_simulation_test_formula_256/`, gồm
  `binned_preds.hdf5`, **`rerank_eval_entropy.yaml`** (top-k accuracy) và các bảng chia nhóm `*_grouped_*.tsv`.
- Phân tích bootstrap cho retrieval MassSpecGym: `analysis/msg_bootstrap.py`. Xem `--help`, vì đường dẫn mặc định của
  nó trỏ tới một kết quả khác.

---

## 5. Tái lập đúng kết quả bài báo (3 seed)

README: *"The above scripts will only run for split_1_rnd1 … If you want to replicate our reported result with random +
scaffold splits and 3 random seeds, please uncomment all entries in the following files."*

Với MassSpecGym, bỏ comment các mục `split_rnd2` và `split_rnd3` trong:
- `configs/marason/marason_train_msg.yaml`, `marason_gen_predict_train_msg.yaml`, `marason_inten_train_msg.yaml`
  (seed 2, 3);
- `run_scripts/marason/02_sweep_gen_thresh.py`;
- hai khối `add_dag_intens.py` của `msg_simulation/split_rnd2`, `split_rnd3` trong `03_run_marason_gen_predict.sh`;
- `run_scripts/marason/05_predict_marason_inten.py`;
- thêm mục `split_rnd2`, `split_rnd3` vào `test_entries` của `run_retrieval_msg.py`.

Báo cáo trung bình ± độ lệch chuẩn trên 3 seed.

---

## 6. Tóm tắt đầu ra và số liệu cần báo cáo

| Bước | File | Số liệu |
|---|---|---|
| 3.1 | `data/spec_datasets/msg_simulation/audit.json` | số phổ (kỳ vọng 119.029), số train/val/test |
| 1 | `split_rnd1/ckpt/gen/best.ckpt` | số epoch, `val_loss` tốt nhất |
| 2 | `split_rnd1/inten_thresh_sweep/summary.tsv` | độ phủ subformula theo số mảnh |
| 4 | `split_rnd1/ckpt/inten/best.ckpt` | số epoch, `val_loss` tốt nhất |
| 5 | `split_rnd1/preds/pred_eval.yaml` | cosine similarity, entropy similarity (tập test) |
| 6 | `split_rnd1/retrieval_msg_simulation_test_formula_256/rerank_eval_entropy.yaml` | top-1 / 5 / 10 accuracy |

(Mọi đường dẫn `split_rnd1/…` nằm trong `results/marason_msg_simulation/`.)

**Những thay đổi so với repo gốc cần ghi vào báo cáo:**
1. Chỉ chạy phần MSG của các script `01`, `03`, `04`, `06`. Bỏ phần NIST.
2. `run_magma.sh` chạy với tham số `msg_simulation`.
3. Bước 4, lần đầu: `load-reference false`, `save-reference true`, và tự tạo thư mục tham chiếu.
4. `run_retrieval_msg.py`: sửa đường dẫn checkpoint, phổ tham chiếu và dataset như mục Bước 6.
5. Số GPU, worker, batch size, `hidden-size`, nếu có thay đổi so với config.
