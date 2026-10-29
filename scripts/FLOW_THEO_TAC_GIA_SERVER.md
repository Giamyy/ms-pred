# Flow chạy MARASON theo repo gốc của tác giả, trên dữ liệu MassSpecGym đã dựng sẵn

Flow chạy đúng các script MARASON của tác giả trong https://github.com/coleygroup/ms-pred
(README, mục *Experiments → MARASON*), trên tập MassSpecGym `msg_simulation` (119.029 phổ).

Dữ liệu đã được xử lý sẵn và để trên Kaggle: https://www.kaggle.com/datasets/giamyy/msg-simulation.
Dataset gồm `labels.tsv`, `spec_files.hdf5`, `splits/`, `magma_outputs/magma_tree.hdf5`,
`subformulae/no_subform.hdf5` và `retrieval/`. Vì vậy **bỏ qua phần xử lý dữ liệu** (dựng từ Hugging Face, chạy
MAGMa, gán subformula) và đi thẳng vào train.

---

## 0. Yêu cầu
- Linux, GPU NVIDIA, driver hỗ trợ CUDA 12.4 (hoặc 11.8). README: *"You need two GPUs with at least 24GB RAM to train
  MARASON (we used NVIDIA A5000)… on a smaller GPU, try cutting down the hidden_size to 256."*
- `git`, `gcc`; ổ đĩa trống vài chục GB.

---

## 1. Lấy code gốc và cài môi trường (README: *Install & setup*)
```bash
cd ~
git clone https://github.com/coleygroup/ms-pred.git
cd ~/ms-pred
git log --oneline -1                               # ghi lại commit đã dùng
curl -LsSf https://astral.sh/uv/install.sh | sh    # nếu chưa có uv
source ~/.bashrc
uv sync --extra cu124                              # CUDA 11.8 thì dùng --extra cu118
source .venv/bin/activate
python -c "import torch, dgl; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```
Dòng cuối phải in `True`.

---

## 2. Tải dữ liệu từ Kaggle
```bash
cd ~/ms-pred
mkdir -p ~/msg_dl
curl -L -o ~/msg_dl/msg-simulation.zip https://www.kaggle.com/api/v1/datasets/download/giamyy/msg-simulation
unzip -q ~/msg_dl/msg-simulation.zip -d ~/msg_dl/x

# chép thư mục chứa labels.tsv vào đúng chỗ code đọc
SRC=$(dirname "$(find ~/msg_dl/x -name labels.tsv | head -1)")
mkdir -p data/spec_datasets/msg_simulation
cp -r "$SRC"/* data/spec_datasets/msg_simulation/
ls data/spec_datasets/msg_simulation data/spec_datasets/msg_simulation/subformulae data/spec_datasets/msg_simulation/retrieval
```
- Phải thấy đủ: `labels.tsv`, `spec_files.hdf5`, `splits/`, `magma_outputs/`, `subformulae/no_subform.hdf5`,
  `retrieval/cands_df_test_formula_256.tsv`.
- Nếu `curl` báo lỗi quyền truy cập: `pip install kaggle`, đặt token vào `~/.kaggle/kaggle.json`, rồi chạy
  `kaggle datasets download giamyy/msg-simulation -p ~/msg_dl`.
- Chép xong có thể xoá `~/msg_dl` cho đỡ tốn chỗ.

---

## 3. Chỉnh cho máy và cho dữ liệu MassSpecGym

Config gốc được viết cho 2–3 GPU, và có vài đường dẫn không khớp với nhau (chi tiết ở cuối file). Chạy các lệnh sau
**một lần** trong `~/ms-pred`:

```bash
cd ~/ms-pred

# (a) Máy 1 GPU: chỉ dùng GPU 0. Máy có 2 GPU thì bỏ qua 2 dòng này.
sed -i 's/visible_devices: \[0, 1\]/visible_devices: [0]/' configs/marason/marason_train_msg.yaml configs/marason/marason_gen_predict_train_msg.yaml configs/marason/marason_inten_train_msg.yaml
sed -i 's/^devices = ",".join(\["0", "1", "2"\])/devices = "0"/' run_scripts/marason/05_predict_marason_inten.py

# (b) Lần train đầu phải tự tính phổ tham chiếu (RAG) thay vì đọc file có sẵn
sed -i 's/load-reference: \[true\]/load-reference: [false]/; s/save-reference: \[false\]/save-reference: [true]/' configs/marason/marason_inten_train_msg.yaml
mkdir -p data/msg_simulation/closest_neighbors/infinite

# (c) Sửa đường dẫn trong script retrieval cho khớp nơi các bước train lưu kết quả
python - <<'EOF'
from pathlib import Path
p = Path("run_scripts/marason/run_retrieval_msg.py")
s = p.read_text()
edits = [
    ('devices = [0, 1]', 'devices = [0]'),           # máy 2 GPU thì xoá dòng này
    ('"dataset": "msg",', '"dataset": "msg_simulation",'),
    ('"results_name": "msg",', '"results_name": "msg_simulation",'),
    ('"data/spec_datasets/msg/retrieval/cands_df_test_formula_256.tsv"',
     '"data/spec_datasets/msg_simulation/retrieval/cands_df_test_formula_256.tsv"'),
    ('"data/spec_datasets/msg/labels.tsv"', '"data/spec_datasets/msg_simulation/labels.tsv"'),
    ('"data/msg/closest_neighbors/infinite"', '"data/msg_simulation/closest_neighbors/infinite"'),
    ('Path(f"results/marason_inten_{results_name}")', 'Path(f"results/marason_{results_name}")'),
    ('inten_dir / inten_split / "version_0/best.ckpt"', 'inten_dir / inten_split / "ckpt/inten/best.ckpt"'),
    ('form_folder.parent / "version_0/best.ckpt"', 'form_folder.parent / "ckpt/gen/best.ckpt"'),
]
for old, new in edits:
    assert s.count(old) == 1, f"không tìm thấy: {old}"
    s = s.replace(old, new)
p.write_text(s)
print("Đã sửa", p)
EOF

git diff --stat     # kiểm tra: 5 file thay đổi
```
- Nếu máy có ít hơn 16 nhân CPU, giảm `num-workers: [16]` trong `marason_train_msg.yaml` và
  `marason_inten_train_msg.yaml`, cùng `num-cpu-workers: [32]` trong `marason_gen_predict_train_msg.yaml`,
  xuống bằng số nhân (`nproc`).
- Nếu GPU dưới 24 GB và gặp `CUDA out of memory`: giảm `batch-size`, hoặc làm theo README là giảm `hidden-size` xuống 256
  trong config tương ứng.

---

## 4. Train và đánh giá (README: *Experiments → MARASON*, 6 bước)

Chạy trong `tmux` để ngắt SSH không làm dừng job:
```bash
tmux new -s marason
cd ~/ms-pred && source .venv/bin/activate
```
(`Ctrl+B` rồi `D` để thoát mà job vẫn chạy; `tmux attach -t marason` để xem lại.)

**Không chạy nguyên `run_scripts/marason/run_all.sh`.** Script đó chạy cả MAGMa và các config NIST'20 (dữ liệu thương
mại, không có), nên sẽ lỗi. Chỉ chạy các lệnh phần MassSpecGym dưới đây, theo đúng thứ tự.

### Bước 1. Train bộ sinh mảnh (`01_run_marason_gen_train.sh`, dòng MSG)
```bash
python launcher_scripts/run_from_config.py configs/marason/marason_train_msg.yaml
```
Đầu ra: `results/marason_msg_simulation/split_rnd1/ckpt/gen/best.ckpt`. Tối đa 200 epoch, có early stopping.

### Bước 2. Dò số mảnh (`02_sweep_gen_thresh.py`, tuỳ chọn, **bỏ qua**)
Bước này cần `subformulae/magma_subform_50.hdf5`, mà dataset chưa có. Các bước sau không phụ thuộc bước này.
`run_all.sh` của tác giả cũng ghi chú có thể bỏ.

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
Đầu ra: `split_rnd1/preds_train_100/tree_preds.hdf5` và `split_rnd1/preds_train_100_inten.hdf5`.
(`--num-workers` nên bằng số nhân CPU.)

### Bước 4. Train bộ cường độ (`04_train_marason_inten.sh`, dòng MSG)
```bash
python launcher_scripts/run_from_config.py configs/marason/marason_inten_train_msg.yaml
```
- Lần này vừa tính và lưu phổ tham chiếu vào `data/msg_simulation/closest_neighbors/infinite/`, vừa train.
- Đầu ra: `split_rnd1/ckpt/inten/best.ckpt`. Đây là bước lâu nhất.
- Nếu phải chạy lại bước này, đổi lại config để đọc phổ tham chiếu đã lưu thay vì tính lại:
  ```bash
  sed -i 's/load-reference: \[false\]/load-reference: [true]/; s/save-reference: \[true\]/save-reference: [false]/' configs/marason/marason_inten_train_msg.yaml
  ```

### Bước 5. Dự đoán và đánh giá phổ trên tập test (`05_predict_marason_inten.py`)
```bash
python run_scripts/marason/05_predict_marason_inten.py
```
Mục `nist20` trong script tự bỏ qua vì không có checkpoint.
Kết quả: `split_rnd1/preds/pred_eval.yaml` (cosine và entropy similarity).

### Bước 6. Retrieval (`06_run_retrieval.sh`, chỉ phần MSG)
```bash
python run_scripts/marason/run_retrieval_msg.py
```
Kết quả: `split_rnd1/retrieval_msg_simulation_test_formula_256/rerank_eval_entropy.yaml` (top-k accuracy, 256 ứng
viên cùng công thức).

(Mọi đường dẫn `split_rnd1/…` nằm trong `~/ms-pred/results/marason_msg_simulation/`.)

---

## 5. Xem kết quả
```bash
cd ~/ms-pred/results/marason_msg_simulation/split_rnd1
cat preds/pred_eval.yaml
cat retrieval_msg_simulation_test_formula_256/rerank_eval_entropy.yaml
```

---

## Ghi chú: những gì đã chỉnh so với repo gốc và lý do
| Chỉnh | Lý do |
|---|---|
| Không chạy MAGMa / subformula | Dữ liệu trên Kaggle đã có sẵn, được tạo bằng chính `data_scripts/dag/run_magma.sh` của tác giả |
| Chỉ chạy phần MSG của `01`, `03`, `04`, `06` | Phần NIST cần dữ liệu thương mại |
| `visible_devices`, `devices` | Config gốc giả định 2–3 GPU |
| `load-reference false`, `save-reference true` (lần đầu) và tạo thư mục | Config gốc đọc phổ tham chiếu đã tính sẵn mà repo không kèm; `train_inten.py` không tự tạo thư mục |
| `run_retrieval_msg.py` | Script tìm checkpoint ở `results/marason_inten_msg/…/version_0/`, nhưng config train lưu ở `results/marason_msg_simulation/split_rnd1/ckpt/{gen,inten}/`; phổ tham chiếu và dataset đổi sang `msg_simulation` |
| Bỏ bước 2 | Cần `magma_subform_50.hdf5`; bước tuỳ chọn |
| Chỉ chạy seed 1 (`split_rnd1`) | Để tái lập số liệu 3 seed trong bài báo, bỏ comment các mục `split_rnd2`, `split_rnd3` trong 3 config MSG, trong `03_…sh` (khối `add_dag_intens.py`) và trong `05_…py`, rồi chạy lại từ bước 1 |
