# Hướng dẫn chạy lại MARASON trên MassSpecGym (`msg_simulation`)

Tài liệu mô tả quy trình train và đánh giá mô hình MARASON (repo `coleygroup/ms-pred`) trên tập
MassSpecGym 1.5, chạy trên server Linux có GPU NVIDIA. Toàn bộ quy trình được gói trong hai script:

| Script | Việc |
|---|---|
| `scripts/setup_env_uv.sh` | Cài môi trường Python (một lần) |
| `scripts/train_marason_msg.py` | Tải dữ liệu từ Kaggle, train, dự đoán, đánh giá (B0–B6, thêm S2 nếu dùng `--sweep`) |

- Mã nguồn: https://github.com/Giamyy/ms-pred, nhánh `claude/code-explanation-frix3g` (fork của `coleygroup/ms-pred`)
- Dữ liệu đã xử lý: https://www.kaggle.com/datasets/giamyy/msg-simulation
  (bản `giamyy/msg-simulation-v2` có thêm `subformulae/magma_subform_50.hdf5`, cần cho bước S2)

---

## 1. Tổng quan

MARASON dự đoán phổ MS/MS từ cấu trúc phân tử bằng hai mô hình nối tiếp:

1. **Bộ sinh mảnh (gen):** từ phân tử, dự đoán cây/DAG phân mảnh.
2. **Bộ cường độ (inten):** dự đoán cường độ của từng mảnh. Mô hình dùng thêm **phổ tham chiếu (RAG)**: tìm các phân tử
   giống nhất trong tập train, rồi dùng **neural graph matching** ghép mảnh của phân tử cần dự đoán với mảnh của phân tử
   tham chiếu để mượn thông tin cường độ.

Quy trình (theo `run_scripts/marason/run_all.sh` của tác giả):

```
Kaggle dataset ──B0──▶ data/spec_datasets/msg_simulation/
                              │
magma_tree ──B1──▶ gen.ckpt ──B2──▶ DAG dự đoán ──B3 (+ phổ thật)──▶ DAG + raw_spec
                       │
                       └──S2 (tuỳ chọn): dò số mảnh, so với magma_subform_50
                                                                       │
                              B4: phổ tham chiếu + train ──────────────┘──▶ inten.ckpt
                                                                               │
                              B5: dự đoán + đánh giá phổ tập test ◀────────────┤
                              B6: retrieval (xếp hạng 256 ứng viên) ◀───────────┘
```

---

## 2. Dữ liệu

Dữ liệu được dựng trên Kaggle bằng notebook `notebooks/kaggle/kaggle_build_msg_data.ipynb`:

| Bước | Cách làm | Nguồn |
|---|---|---|
| `labels.tsv`, `spec_files.hdf5`, `splits/split.tsv` | `data_scripts/create_msg_simulation_dataset.py --strategy msg_simulation`, đọc MassSpecGym 1.5 từ Hugging Face | Script gốc |
| Cây phân mảnh `magma_outputs/magma_tree.hdf5` | `src/ms_pred/magma/run_magma.py` (`--max-peaks 50 --ppm-diff 20`), chạy chia 4 phần rồi gộp | Script gốc; chia/gộp tự viết |
| Subformula `subformulae/no_subform.hdf5` | `data_scripts/forms/01_assign_subformulae.py --use-all` | Script gốc |
| Bảng ứng viên `retrieval/cands_df_test_formula_256.tsv` | Dropbox của tác giả | Gốc |
| Bảng ứng viên `retrieval/cands_df_test_mass_256.tsv` | `data_scripts/create_msg_candidate_tables.py`, từ danh sách ứng viên công khai của MassSpecGym | Tự dựng lại |
| Subformula `subformulae/magma_subform_50.hdf5` (chỉ cho S2) | `data_scripts/forms/01_assign_subformulae.py --use-magma --mass-diff-thresh 20` (lệnh trong `data_scripts/all_assign_subform.sh`), chạy chia shard bằng notebook `notebooks/kaggle/kaggle_magma_subform50.ipynb`, lưu thành dataset `msg-simulation-v2` | Script gốc; chia/gộp tự viết |

- Số phổ: **119.029**. Chia train/val/test theo cột `fold` chính thức của MassSpecGym.
- MAGMa sinh được cây cho khoảng 99,8% số phổ. Phần còn lại bị bỏ qua vì không sinh được mảnh hoặc không có đỉnh dùng được.
- Đã kiểm chứng bằng MAGMa thật trên mẫu 30 phân tử: kết quả chia 4 phần rồi gộp **giống hệt** chạy một lần (chỉ khác thứ tự lưu).

---

## 3. Yêu cầu hệ thống

| | Yêu cầu |
|---|---|
| Hệ điều hành | Linux (hoặc WSL2 trên Windows). DGL không có bản CUDA cho Windows. |
| GPU | NVIDIA; driver hỗ trợ CUDA 12.x (hoặc 11.8) |
| Phần mềm | `git`, `gcc` (biên dịch phần Cython) |
| Ổ đĩa | vài chục GB; lúc tải dữ liệu cần thêm khoảng 2 lần kích thước dataset |
| Mạng | truy cập GitHub, PyPI, download.pytorch.org, data.dgl.ai, data.pyg.org, kaggle.com |

Kiểm tra:
```bash
nvidia-smi        # tên GPU, VRAM, "CUDA Version"
nproc             # số nhân CPU
df -h ~           # dung lượng trống
gcc --version
```

---

## 4. Cài đặt (một lần)

### 4.1. Lấy mã nguồn
```bash
cd ~
git clone -b claude/code-explanation-frix3g https://github.com/Giamyy/ms-pred.git ms-pred
cd ~/ms-pred
git log --oneline -1          # ghi lại commit đã dùng
```

### 4.2. Cài môi trường
```bash
bash scripts/setup_env_uv.sh
```
Script cài `uv` nếu chưa có, đọc phiên bản CUDA từ `nvidia-smi` để chọn bản `cu124` / `cu118` / `cpu`, rồi chạy
`uv sync --extra <bản> --python 3.11` theo `pyproject.toml`: Python 3.11, torch 2.6.0, DGL, torch-scatter, rdkit 2025.03,
PyTorch Lightning, pygmtools… Môi trường nằm ở `~/ms-pred/.venv`. Không cần quyền sudo.

Kết quả đúng: dòng `CUDA: True` kèm tên GPU.

### 4.3. Token Kaggle (chỉ khi dataset để private)
```bash
mkdir -p ~/.kaggle && nano ~/.kaggle/kaggle.json   # Kaggle → Settings → API → Create New Token
chmod 600 ~/.kaggle/kaggle.json
```

---

## 5. Chạy

### 5.1. Chạy thử (2 epoch)
Để kiểm tra toàn bộ quy trình, đo thời gian mỗi epoch, và chọn batch size vừa VRAM:
```bash
cd ~/ms-pred && source .venv/bin/activate
python scripts/train_marason_msg.py --max-epochs 2
rm -rf results/marason_msg_simulation        # xoá kết quả thử; giữ dữ liệu và phổ tham chiếu
```
Nếu gặp `CUDA out of memory`, thêm `--gen-batch 16 --inten-batch 8`.

### 5.2. Chạy chính thức

**Server không có Slurm** (dùng `tmux` để ngắt SSH không làm dừng job):
```bash
tmux new -s marason
cd ~/ms-pred && source .venv/bin/activate
python scripts/train_marason_msg.py --gpus 0
# Ctrl+B rồi D: thoát, job vẫn chạy.   tmux attach -t marason: xem lại.
```

**Server có Slurm**, file `job.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=marason
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=marason_%j.log
cd ~/ms-pred && source .venv/bin/activate
python scripts/train_marason_msg.py --gpus 0 --num-workers 8
```
Gửi job bằng `sbatch job.sh`, theo dõi bằng `squeue -u $USER`, huỷ bằng `scancel <mã job>`.

### 5.3. Tuỳ chọn của `train_marason_msg.py`

| Tuỳ chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--gpus` | `0` | GPU dùng (`CUDA_VISIBLE_DEVICES`), ví dụ `0,1` |
| `--num-workers` | số nhân CPU, tối đa 16 | tiến trình đọc dữ liệu / xử lý song song |
| `--gen-batch` / `--inten-batch` | 32 / 16 | batch size khi train B1 / B4 |
| `--gen-pred-batch` / `--inten-pred-batch` / `--retrieval-batch` | 8 / 64 / 10 | batch size khi dự đoán B2 / B5 / B6 |
| `--max-epochs` | 200 | số epoch tối đa của B1, B4 (có early stopping) |
| `--steps` | `B1 … B6` | chỉ chạy một số bước (có thể gồm `S2`) |
| `--sweep` | | chạy thêm S2 (dò số mảnh); cần `subformulae/magma_subform_50.hdf5` |
| `--force` | | chạy lại bước đã xong |
| `--kaggle-dataset` | `giamyy/msg-simulation` | dataset Kaggle cần tải |
| `--skip-download` / `--keep-download` | | không tải / giữ file zip sau khi giải nén |

Config gốc trong `configs/marason/` được giữ nguyên. Script chép chúng sang
`results/marason_msg_simulation/local_configs/`, chỉ sửa GPU, số worker, batch size và số epoch.
**Siêu tham số mô hình giữ như tác giả.**

---

## 6. Các bước trong script

Mỗi bước ghi log vào `results/marason_msg_simulation/logs/B*.log`. Khi bước thành công, script tạo file đánh dấu
`split_rnd1/.done_B*`. Nếu job bị dừng, chạy lại đúng lệnh cũ: các bước đã xong được bỏ qua. Khi một bước được chạy lại,
các bước sau nó cũng được đánh dấu là cần chạy lại. Thư mục kết quả: `results/marason_msg_simulation/split_rnd1/`.

### B0. Tải dữ liệu
Nếu `data/spec_datasets/msg_simulation/` chưa đủ file, script tải zip của dataset Kaggle qua
`https://www.kaggle.com/api/v1/datasets/download/giamyy/msg-simulation`. Sau đó giải nén, tìm thư mục chứa `labels.tsv`
và `magma_outputs/magma_tree.hdf5`, chuyển vào `data/spec_datasets/msg_simulation/`, rồi xoá zip.

### B1. Train bộ sinh mảnh
- **Script:** `src/ms_pred/marason/train_gen.py` (config `marason_train_msg.yaml`)
- **Đầu vào:** `labels.tsv`, `splits/split.tsv`, `magma_outputs/magma_tree.hdf5` (cây MAGMa làm đáp án)
- **Mô hình:** GNN 6 lớp (GGNN, hidden 512, positional encoding random-walk k=14). Ở mỗi bước phân mảnh, mô hình học
  dự đoán nguyên tử nào bị tách ra. Learning rate 0,000996, giảm 0,7214 mỗi epoch.
- **Dừng:** tối đa 200 epoch; early stopping khi `val_loss` không giảm trong 5 epoch liên tiếp.
- **Đầu ra:** `ckpt/gen/best.ckpt`

### S2. Dò số mảnh (tuỳ chọn, `--sweep`)
- **Tương ứng:** bước 2 của tác giả, `run_scripts/marason/02_sweep_gen_thresh.py`.
- **Việc:** với mỗi số mảnh tối đa trong {10, 20, 30, 40, 50, 100, 200, 300, 500, 1000}, dùng bộ sinh mảnh của B1 sinh
  DAG cho tập test (`predict_gen.py --subset-datasets test_only --threshold 0 --max-nodes N`, batch 8). Sau đó so các mảnh
  với `subformulae/magma_subform_50.hdf5` bằng `analysis/form_pred_eval.py` (độ phủ, Jaccard…). Kết quả cho biết nên
  giữ bao nhiêu mảnh; các bước sau dùng 100 như config của tác giả.
- **Vì sao không gọi thẳng script gốc:** `02_sweep_gen_thresh.py` của MARASON truyền `--num-workers` cho `predict_gen.py`
  và `--tree-pred-folder` cho `form_pred_eval.py`, nhưng hai script này không nhận các tham số đó, nên sẽ lỗi. S2 làm theo
  bản cập nhật của tác giả cho ICEBERG (`run_scripts/iceberg/nist23/02_sweep_gen_thresh.py`): đánh giá trực tiếp
  `tree_preds.hdf5`.
- **Đầu ra:** `inten_thresh_sweep/<N>/pred_eval.yaml` và bảng tổng hợp `inten_thresh_sweep/summary.tsv`.
- S2 không ảnh hưởng B2–B6. Chạy lại B1 thì kết quả S2 cũ bị xoá.

### B2. Sinh DAG phân mảnh
- **Script:** `src/ms_pred/marason/predict_gen.py` (config `marason_gen_predict_train_msg.yaml`)
- **Việc:** sinh DAG tối đa 100 mảnh (`max-nodes 100`) cho mọi phân tử. Bộ cường độ được train trên DAG do mô hình sinh ra
  (thay vì cây MAGMa) để khớp với lúc dự đoán thật.
- **Đầu ra:** `preds_train_100/tree_preds.hdf5`

### B3. Gắn phổ thật vào DAG
- **Script:** `data_scripts/dag/add_dag_intens.py --add-raw` (chỉ dùng CPU)
- **Việc:** với mỗi phổ, đọc danh sách đỉnh thật `[m/z, cường độ]` từ `subformulae/no_subform.hdf5`, rồi gắn vào DAG dự đoán
  tương ứng dưới khoá `raw_spec`.
- **Đầu ra:** `preds_train_100_inten.hdf5`. File này dùng làm đáp án khi train B4, và làm kho phổ tham chiếu (phần tập train).

### B4. Phổ tham chiếu + train bộ cường độ
- **Script:** `src/ms_pred/marason/train_inten.py` (config `marason_inten_train_msg.yaml`)
- **Phổ tham chiếu (lần đầu tính, các lần sau đọc lại):** với mỗi phổ, tính khoảng cách Jaccard giữa Morgan fingerprint
  của nó và mọi phân tử **trong tập train**. Loại ứng viên khác adduct, khác loại máy đo, hoặc trùng hệt cấu trúc. Giữ các
  láng giềng gần nhất, ưu tiên năng lượng va chạm gần hơn. Kết quả lưu ở
  `data/msg_simulation/closest_neighbors/infinite/` (`infinite`: không giới hạn chênh lệch năng lượng va chạm).
- **Mô hình** (mỗi phổ dùng tối đa 3 láng giềng):
  1. GNN 4 lớp mã hoá từng mảnh của phân tử cần dự đoán và của láng giềng (hai GNN riêng), kèm adduct, năng lượng va chạm
     và máy đo.
  2. Với mỗi mảnh của láng giềng, đọc cường độ trong phổ thật của láng giềng tại m/z của mảnh đó.
  3. **Graph matching:** độ tương đồng cosine giữa mọi cặp mảnh, rồi softmax với τ = 0,02. Mỗi mảnh của phân tử cần dự đoán
     được ghép với mảnh giống nhất của láng giềng.
  4. Nối thông tin mượn được (vector mảnh láng giềng, cường độ, điểm ghép, khoảng cách Tanimoto), cho qua 3 lớp Transformer.
  5. Dự đoán cường độ cho từng mảnh và từng biến thể lệch H, rồi gộp theo bin m/z thành phổ.
- **Loss:** entropy giữa phổ dự đoán và phổ thật. Learning rate 0,0002, giảm 0,825 mỗi epoch; early stopping 5 epoch.
- **Đầu ra:** `ckpt/inten/best.ckpt`. Đây là bước lâu nhất và tốn VRAM nhất.

### B5. Dự đoán và đánh giá phổ trên tập test
- **Script:** `src/ms_pred/marason/predict_inten.py` (`--subset-datasets test_only --add-ref --max-ref-count 3`),
  rồi `analysis/spec_pred_eval.py`
- **Việc:** dự đoán phổ cho các phân tử test (láng giềng vẫn chỉ lấy từ tập train), so với phổ thật.
- **Đầu ra:** `preds/binned_preds.hdf5`; **cosine similarity** và **entropy similarity** ở cuối `logs/B5.log`.

### B6. Retrieval
- **Script:** `src/ms_pred/marason/predict_smis.py` rồi `src/ms_pred/retrieval/retrieval_benchmark.py`
- **Việc:** mỗi phổ test có tối đa 256 phân tử ứng viên. Dự đoán phổ cho từng ứng viên (DAG tối đa 250 mảnh), so với phổ
  thật bằng khoảng cách entropy, rồi xếp hạng. Chạy cho bảng `formula` (ứng viên cùng công thức), và bảng `mass` (ứng viên
  cùng khối lượng) nếu có.
- **Đầu ra:** `retrieval_msg_simulation_test_{formula,mass}_256/`; **top-1 / top-5 / top-10** ở cuối `logs/B6.log`.

---

## 7. Theo dõi và xử lý sự cố

| Muốn | Lệnh |
|---|---|
| Xem log trực tiếp | `tail -f results/marason_msg_simulation/logs/B4.log` |
| Kiểm tra GPU đang chạy | `nvidia-smi` |
| Các bước đã xong | `ls -a results/marason_msg_simulation/split_rnd1/ \| grep done` |
| Thời gian từng bước | `grep "xong sau" results/marason_msg_simulation/logs/*.log` |
| Chạy tiếp sau khi dừng | chạy lại đúng lệnh cũ |
| Chạy lại từ B4 | `python scripts/train_marason_msg.py --steps B4 B5 B6 --force` |

| Lỗi | Cách xử lý |
|---|---|
| `torch không thấy GPU` | chưa `source .venv/bin/activate`, hoặc driver không hỗ trợ CUDA; chạy lại `setup_env_uv.sh` |
| `CUDA out of memory` | giảm `--gen-batch` (B1) hoặc `--inten-batch` (B4) |
| B0: `HTTP 401/403/404` | dataset private hoặc sai tên: đặt `~/.kaggle/kaggle.json` (mục 4.3) |
| Server không ra được internet | tải zip về máy, chép lên server, giải nén vào `data/spec_datasets/msg_simulation/`, chạy với `--skip-download` |
| Slurm dừng job vì hết thời gian | `sbatch job.sh` lại; script chạy tiếp, bước đang dở chạy lại từ đầu bước đó |

---

## 8. Kết quả cần báo cáo

```bash
tail -40 results/marason_msg_simulation/logs/B5.log
tail -40 results/marason_msg_simulation/logs/B6.log
cat results/marason_msg_simulation/split_rnd1/inten_thresh_sweep/summary.tsv    # nếu chạy S2
grep "xong sau" results/marason_msg_simulation/logs/*.log
```

| Mục | Giá trị |
|---|---|
| Server / GPU / VRAM / CUDA | |
| Commit mã nguồn | |
| Số phổ | 119.029 (train / val / test theo `fold` của MassSpecGym) |
| Batch size gen / inten; số epoch thực tế (early stopping) | |
| Thời gian B1 / B4 / toàn bộ | |
| Cosine similarity (test) | |
| Entropy similarity (test) | |
| Retrieval top-1 / 5 / 10, bảng formula | |
| Retrieval top-1 / 5 / 10, bảng mass | |
| S2: độ phủ subformula theo số mảnh (nếu chạy) | |
| So với bài báo MARASON | |

### Khác biệt so với quy trình của tác giả
- MAGMa chạy chia 4 phần trên Kaggle (giới hạn 12 giờ mỗi phiên) rồi gộp. Đã kiểm chứng kết quả giống chạy một lần.
- Bảng ứng viên `mass` được dựng lại từ danh sách công khai của MassSpecGym, nên có thể lệch nhẹ so với bảng của tác giả.
  Bảng `formula` là của tác giả.
- Config chỉ chỉnh GPU, số worker, batch size cho vừa máy. Nếu phải giảm batch size, cần ghi rõ trong báo cáo vì có thể
  ảnh hưởng nhẹ tới kết quả.
- Script `run_retrieval_msg.py` gốc dùng đường dẫn checkpoint và phổ tham chiếu không khớp với nơi các config train lưu.
  B6 gọi trực tiếp `predict_smis.py` và `retrieval_benchmark.py` với cùng tham số như script gốc, nhưng dùng đường dẫn đúng.
- Bước 2 (dò số mảnh) không dùng nguyên `02_sweep_gen_thresh.py` của MARASON vì script không khớp code hiện tại; S2 làm
  theo bản ICEBERG cập nhật của tác giả (xem mục S2).
- Chỉ chạy một seed (`split_rnd1`). Các config gốc để sẵn `split_rnd2` và `split_rnd3` nhưng đã comment lại.
