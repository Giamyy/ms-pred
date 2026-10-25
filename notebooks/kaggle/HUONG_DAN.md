# Tái lập dữ liệu MassSpecGym và chạy MARASON trên Kaggle

Tài liệu này mô tả toàn bộ trình tự: dựng lại dữ liệu `msg_simulation` bằng notebook
`notebooks/kaggle/msg_iceberg_kaggle.ipynb`, rồi dùng dữ liệu đó để chạy pipeline MARASON gốc
(`run_scripts/marason/`).

```
Phần A: dựng dữ liệu                         Phần B: chạy MARASON
────────────────────────────────            ─────────────────────────────────────────
A0 chuẩn bị (notebook, bảng formula)         B0 notebook GPU + môi trường + dữ liệu
A1 chạy thử 200 phổ (tuỳ chọn)               B1 train bộ sinh mảnh (gen)
A2 4 shard MAGMA, song song (~6 giờ)         B2 dự đoán DAG cho tập train
A3 dựng bảng mass (tuỳ chọn, ~1 giờ)          B3 gán cường độ thật vào DAG
A4 merge → msg_simulation (~1 giờ)           B4 phổ tham chiếu + train cường độ (inten)
A5 lưu thành dataset msg-spec-datasets       B5 dự đoán và đánh giá phổ
                                             B6 đánh giá retrieval
```

---

## Phần A: Dựng lại dữ liệu

### Vì sao phải chia nhiều notebook

MAGMA trên 119,029 phổ mất khoảng **24 giờ với 4 CPU** (đo được 23.7 phút cho 2000 phổ), trong khi
một phiên Kaggle tối đa 12 giờ. Vì vậy notebook có tham số `STAGE` ở ô 0:

| `STAGE` | Làm gì |
|---|---|
| `"all"` | Làm hết trong một phiên. Chỉ dùng để chạy thử với `N_SPECS` nhỏ. |
| `"shard"` | Chạy MAGMA cho phần `SHARD_ID` trong `NUM_SHARDS` phần |
| `"retrieval"` | Chỉ dựng bảng ứng viên còn thiếu (bảng mass) |
| `"merge"` | Gộp các phần, chạy script chính thức, tạo `msg_simulation` |

### A0. Chuẩn bị (một lần)

1. Kaggle → *Create → New Notebook → File → Import Notebook* → chọn `msg_iceberg_kaggle.ipynb`
   bản mới nhất. Đặt tên `msg-base`, dùng làm bản gốc để *Copy & Edit*.
2. Tải `retrieval/cands_df_test_formula_256.tsv` từ thư mục Dropbox của tác giả (link trong README,
   mục "Pretrained ICEBERG 2.1 Model Weights on MassSpecGym"). *Datasets → New Dataset* → đặt tên
   `msg-retrieval`. File `..._no_stereo_clean_tauts.tsv` cùng thư mục không cần cho MARASON.

### A1. Chạy thử (tuỳ chọn, ~10 phút)

Copy `msg-base`, ô 0: `STAGE = "all"`, `N_SPECS = 200`, `RETRIEVAL_DIR = None`. Ô 8 phải in
`SẴN SÀNG cho MARASON`. Bản này chỉ để kiểm tra notebook, không dùng để train.

### A2. Bốn shard MAGMA (song song, ~6 giờ mỗi shard)

Với `i` = 0, 1, 2, 3: copy `msg-base` thành `msg-shard-i`, ô 0:

```python
STAGE      = "shard"
NUM_SHARDS = 4
SHARD_ID   = i
N_SPECS    = None
```

*Settings*: Accelerator **None** (MAGMA chỉ dùng CPU), Internet **On** → *Save Version → Save & Run All*.

Kết quả mỗi shard: ô 8 in 4 dòng `OK` cho `msg_shard<i>/...`. Để kiểm tra chéo, mỗi shard có khoảng
29,757 phổ trong `labels`, khoảng 99.8% có cây MAGMA, 100% có subformula.

### A3. Dựng bảng mass (tuỳ chọn, ~1 giờ)

Dropbox chỉ có bảng formula. Bảng mass chỉ cần cho đánh giá retrieval theo khối lượng;
`run_retrieval_msg.py` của MARASON chỉ dùng bảng formula.

Copy `msg-base` thành `msg-mass`, *Add Input* `msg-retrieval`, ô 0:

```python
STAGE           = "retrieval"
N_SPECS         = None
RETRIEVAL_DIR   = "/kaggle/input/datasets/<user>/msg-retrieval"
BUILD_RETRIEVAL = True
```

Accelerator **None**. Ô 4b in tiến độ: `queries + candidate-list keys: normalising ~34,000 SMILES`, rồi
`candidates: normalising ~600,000 SMILES` kèm dòng `50000/…` khoảng 4–5 phút một lần.
Output: `retrieval/cands_df_test_mass_256.tsv`.

### A4. Merge (~1 giờ)

Copy `msg-base` thành `msg-merge`. *Add Input*: 4 notebook shard, `msg-retrieval`, và `msg-mass` nếu có.
Chạy tạm `!find /kaggle/input -name "cands_df*"` để lấy đường dẫn các bảng. Ô 0:

```python
STAGE           = "merge"
NUM_SHARDS      = 4
N_SPECS         = None
RETRIEVAL_DIR   = ["/kaggle/input/datasets/<user>/msg-retrieval",     # bảng formula của tác giả
                   "/kaggle/input/<...>/msg-mass/retrieval"]          # bảng mass, bỏ dòng này nếu không có
BUILD_RETRIEVAL = False
```

**Chạy tương tác ô 0 → 4b trước**, phải thấy `quy trình: CHÍNH THỨC`, rồi mới *Save & Run All*.
Nếu thấy `DỰ PHÒNG`, `RETRIEVAL_DIR` đang sai đường dẫn.

Kết quả đúng:
- ô 7: `Số phổ: 119029 | imputed: {... 0 ...}`;
- ô 8: tất cả `OK`, `Số phổ trong labels.tsv: 119029`, `SẴN SÀNG cho MARASON`;
- Output chỉ có `spec_datasets/` với `msg/` và `msg_simulation/`.

### A5. Lưu thành dataset

Tab **Output** của `msg-merge` → **New Dataset** → đặt tên `msg-spec-datasets`.

### Dữ liệu thu được

```
spec_datasets/
├── msg/
│   ├── labels.tsv                      (có sẵn trong repo)
│   ├── spec_files.hdf5
│   ├── magma_outputs/magma_tree.hdf5
│   ├── subformulae/no_subform.hdf5
│   └── retrieval/cands_df_test_formula_256.tsv  (+ ..._mass_256.tsv)
└── msg_simulation/                     (MARASON dùng thư mục này để train)
    ├── labels.tsv, splits/split.tsv
    ├── spec_files.hdf5, magma_outputs/, subformulae/
    ├── retrieval/                      (bảng đã lọc theo phổ của msg_simulation)
    ├── labels_debug.tsv, splits/split_debug.tsv, spec_files_debug.hdf5
    └── audit.json
```

### Phần nào dùng script gốc, phần nào tự viết

| Bước | Script | Nguồn |
|---|---|---|
| Phổ → `spec_files.hdf5` | `data_scripts/create_msg_spec_files.py` | Tự viết (repo không có) |
| MAGMA | `data_scripts/dag/run_magma.sh` → `src/ms_pred/magma/run_magma.py` | Gốc |
| Subformula | `data_scripts/forms/01_assign_subformulae.py` | Gốc |
| Tạo `msg_simulation` | `data_scripts/create_msg_simulation_dataset.py` | Gốc |
| Chia và gộp shard | `data_scripts/dag/make_magma_shard.py`, `merge_magma_shards.py` | Tự viết; đã kiểm chứng bằng MAGMA thật cho kết quả như chạy một lần |
| Bảng formula | Dropbox của tác giả | Gốc |
| Bảng mass | `data_scripts/create_msg_candidate_tables.py` | Tự viết, dựng từ danh sách ứng viên công khai của MassSpecGym |

### Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Ô 2: `libcudart.so.12: cannot open shared object file` | Dùng bản notebook mới nhất, hoặc chọn GPU T4 |
| Ô 2: `Kernel đang dùng torch …` | *Run → Restart kernel* rồi chạy lại từ ô 0 |
| Shard chạy quá 12 giờ | Tăng `NUM_SHARDS` (vd. 6) ở **mọi** notebook shard và merge, chạy lại các shard |
| Merge: `Cần đủ 4 shard …` | Kiểm tra đã Add Input đủ 4 shard và cả 4 đều Successful |
| Ô 4b: `KHÔNG CÓ (kiểm tra lại đường dẫn)` hoặc `DỰ PHÒNG` | Sửa `RETRIEVAL_DIR` theo kết quả `find` |
| Bấm Interrupt nhưng CPU vẫn bận | Bản mới tự dừng tiến trình con; bản cũ: `!pkill -f <tên script>` |
| `SyntaxWarning: invalid escape sequence` | Cảnh báo từ code gốc với Python 3.13, bỏ qua được |

---

## Phần B: Sau khi có dữ liệu, chạy MARASON

Các bước dưới theo `run_scripts/marason/run_all.sh`, phần MassSpecGym. Đây là phần **train**,
cần GPU (*Accelerator: GPU T4 x2*).

### B0. Chuẩn bị notebook train

1. Tạo notebook mới, *Add Input* `msg-spec-datasets`, Accelerator **GPU T4 x2**, Internet **On**.
2. Cài môi trường: chép **ô 1 (cài đặt)** và **ô 2 (kiểm tra)** của `msg_iceberg_kaggle.ipynb`.
   Trên Kaggle, Python 3.13 không có wheel torch-scatter nên notebook dùng bản thay thế bằng torch thuần;
   kết quả tính toán đúng, có thể chậm hơn.
3. Clone repo rồi trỏ dữ liệu vào đúng chỗ code mong đợi:
   ```python
   !git clone --depth 1 https://github.com/coleygroup/ms-pred.git /kaggle/working/ms-pred
   %cd /kaggle/working/ms-pred
   !rm -rf data/spec_datasets
   !ln -s /kaggle/input/<...>/msg-spec-datasets/spec_datasets data/spec_datasets
   ```
   `/kaggle/input` chỉ đọc. Nếu bước nào báo lỗi ghi vào `data/spec_datasets`, chép ra thay vì link:
   `!cp -r /kaggle/input/<...>/spec_datasets data/`.
4. Sửa config cho Kaggle (4 CPU, 2 GPU T4) trong các file `configs/marason/*_msg.yaml`:
   `num-workers` (và `num-cpu-workers`) xuống 4; `visible_devices: [0, 1]` giữ được với GPU T4 x2.

### B1. Train bộ sinh mảnh

```bash
python launcher_scripts/run_from_config.py configs/marason/marason_train_msg.yaml
```
Đầu ra: `results/marason_msg_simulation/split_rnd1/ckpt/gen/best.ckpt`.
Config đặt 200 epoch nhưng có early stopping (patience 5) nên thường dừng sớm hơn.

`run_scripts/marason/02_sweep_gen_thresh.py` (dò số mảnh) là tuỳ chọn; nó dùng
`magma_subform_50.hdf5`, file mà quy trình dữ liệu ở trên không tạo, nên cần sửa `subform_name`
thành `no_subform.hdf5` nếu muốn chạy.

### B2. Dự đoán DAG cho tập train

```bash
python launcher_scripts/run_from_config.py configs/marason/marason_gen_predict_train_msg.yaml
```
Đầu ra: `results/marason_msg_simulation/split_rnd1/preds_train_100/tree_preds.hdf5`.

### B3. Gán cường độ thật vào DAG

```bash
python data_scripts/dag/add_dag_intens.py \
  --pred-dag-path results/marason_msg_simulation/split_rnd1/preds_train_100/tree_preds.hdf5 \
  --true-dag-path data/spec_datasets/msg_simulation/subformulae/no_subform.hdf5 \
  --out-dag-path results/marason_msg_simulation/split_rnd1/preds_train_100_inten.hdf5 \
  --num-workers 4 --add-raw
```

### B4. Phổ tham chiếu và train mô hình cường độ

`configs/marason/marason_inten_train_msg.yaml` đọc phổ tham chiếu (phần RAG của MARASON) đã tính sẵn ở
`data/msg_simulation/closest_neighbors/infinite` (`load-reference: true`). Lần đầu phải tạo:

```bash
mkdir -p data/msg_simulation/closest_neighbors/infinite
```
rồi sửa trong config: `load-reference: [false]`, `save-reference: [true]`, và chạy:

```bash
python launcher_scripts/run_from_config.py configs/marason/marason_inten_train_msg.yaml
```
Lần chạy này vừa tính và lưu phổ tham chiếu vừa train. Các lần sau đặt lại `load-reference: [true]`,
`save-reference: [false]`. Phổ tham chiếu gắn với đúng tập dữ liệu; đổi dữ liệu thì phải tính lại.

Đầu ra: `results/marason_msg_simulation/split_rnd1/ckpt/inten/best.ckpt`.

### B5. Dự đoán và đánh giá phổ trên tập test

```bash
python run_scripts/marason/05_predict_marason_inten.py
```
Script đã có sẵn mục `msg_simulation` (`split_rnd1`). Sửa `devices = "0,1,2"` thành `"0,1"`
cho GPU T4 x2.

### B6. Đánh giá retrieval

```bash
python run_scripts/marason/run_retrieval_msg.py
```
**Script này cần sửa đường dẫn trước khi chạy.** Nó tìm mô hình ở cấu trúc thư mục khác với nơi các
config train ở trên lưu checkpoint:

| Script đang dùng | Nơi B1/B4 thực sự lưu |
|---|---|
| `results/marason_inten_msg/split_rnd1/version_0/best.ckpt` | `results/marason_msg_simulation/split_rnd1/ckpt/inten/best.ckpt` |
| bộ sinh mảnh: `<thư mục magma_dag_folder>/../version_0/best.ckpt` | `results/marason_msg_simulation/split_rnd1/ckpt/gen/best.ckpt` |
| `ref_dir: data/msg/closest_neighbors/infinite` | phổ tham chiếu tính ở B4: `data/msg_simulation/closest_neighbors/infinite` |

Bảng ứng viên script dùng là `data/spec_datasets/msg/retrieval/cands_df_test_formula_256.tsv`, đã có
sẵn từ Phần A. Script đặt sẵn `devices = [0, 1]`, khớp với GPU T4 x2.

### Lưu ý khi train trên Kaggle

- Mỗi phiên tối đa 12 giờ, output tối đa 20 GB. Sau mỗi bước, *Save Version* rồi biến output
  (`results/...`) thành dataset để bước sau dùng tiếp.
- B1 và B4 là hai bước lâu nhất. Nếu một phiên không đủ, giảm `max-epochs` hoặc chia bước ra
  nhiều phiên.
- Mọi `results/...` nằm trong thư mục repo đã clone ở `/kaggle/working/ms-pred`.
