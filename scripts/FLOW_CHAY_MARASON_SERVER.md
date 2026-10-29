# Flow chạy MARASON trên server (dữ liệu MassSpecGym đã dựng sẵn trên Kaggle)

Nhờ anh/chị chạy giúp em quy trình train và đánh giá MARASON trên tập MassSpecGym `msg_simulation` (119.029 phổ).
Dữ liệu em đã dựng sẵn trên Kaggle, nên **không cần xử lý dữ liệu**. Script tự tải dữ liệu về, rồi train và đánh giá.

- Code: https://github.com/Giamyy/ms-pred, nhánh `claude/code-explanation-frix3g` (fork của `coleygroup/ms-pred`)
- Dữ liệu: https://www.kaggle.com/datasets/giamyy/msg-simulation

---

## Yêu cầu
- Linux, GPU NVIDIA (driver hỗ trợ CUDA 12.x hoặc 11.8). Tác giả dùng 2 × A5000 24 GB; 1 GPU cũng chạy được.
- Có `git` và `gcc`. Không cần quyền sudo.
- Ổ đĩa trống vài chục GB (nên để sẵn khoảng 50 GB).
- Mạng ra được GitHub, PyPI, download.pytorch.org, data.dgl.ai, data.pyg.org, kaggle.com.

---

## Các bước

### 1. Lấy code và cài môi trường (khoảng 10–20 phút, một lần)
```bash
cd ~
git clone -b claude/code-explanation-frix3g https://github.com/Giamyy/ms-pred.git ms-pred
cd ~/ms-pred
bash scripts/setup_env_uv.sh
```
Script tự chọn bản CUDA theo driver và tạo môi trường `.venv` trong thư mục repo.
Kết quả đúng là dòng có `CUDA: True` kèm tên GPU.

### 2. Chạy thử 2 epoch (để kiểm tra lỗi và đo thời gian)
```bash
cd ~/ms-pred && source .venv/bin/activate
python scripts/train_marason_msg.py --gpus 0 --max-epochs 2
```
- Lần đầu, script tự tải dataset Kaggle `giamyy/msg-simulation` vào `data/spec_datasets/msg_simulation/`.
- Nếu báo `CUDA out of memory`: chạy lại với `--gen-batch 16 --inten-batch 8`.
- Chạy thử xong thì xoá kết quả thử, **giữ lại** thư mục dữ liệu:
  ```bash
  rm -rf results/marason_msg_simulation
  ```

### 3. Chạy chính thức

**Không có Slurm:**
```bash
tmux new -s marason
cd ~/ms-pred && source .venv/bin/activate
python scripts/train_marason_msg.py --gpus 0
# Ctrl+B rồi D để thoát (job vẫn chạy);  tmux attach -t marason để xem lại
```

**Có Slurm**, file `job.sh`:
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
Gửi job bằng `sbatch job.sh`.

- Nếu ở bước 2 phải giảm batch, thêm đúng các tham số đó vào lệnh.
- Nếu job bị dừng (mất kết nối, hết giờ Slurm…): chạy lại **đúng lệnh cũ**. Script tự bỏ qua các bước đã xong.
- Có 2 GPU thì dùng `--gpus 0,1`.

Script chạy lần lượt:

| Bước | Việc | Ghi chú |
|---|---|---|
| B0 | Tải dữ liệu từ Kaggle | bỏ qua nếu đã có |
| B1 | Train bộ sinh mảnh | lâu |
| B2 | Sinh DAG phân mảnh | |
| B3 | Gắn phổ thật vào DAG | chỉ CPU |
| B4 | Tính phổ tham chiếu + train bộ cường độ | lâu nhất |
| B5 | Dự đoán và đánh giá phổ trên tập test | |
| B6 | Đánh giá retrieval | |

Theo dõi tiến độ:
```bash
tail -f ~/ms-pred/results/marason_msg_simulation/logs/B1.log      # đổi B1 thành bước đang chạy
grep "xong sau" ~/ms-pred/results/marason_msg_simulation/logs/*.log
```

---

## 4. Gửi lại cho em
Khi màn hình in `Hoàn tất`:
```bash
cd ~/ms-pred
tar czf marason_ketqua.tar.gz \
  results/marason_msg_simulation/logs \
  results/marason_msg_simulation/local_configs \
  results/marason_msg_simulation/split_rnd1/preds/pred_eval.yaml \
  results/marason_msg_simulation/split_rnd1/retrieval_msg_simulation_test_*_256/rerank_eval_entropy.yaml \
  results/marason_msg_simulation/split_rnd1/ckpt
nvidia-smi > thong_tin_gpu.txt; git log --oneline -1 >> thong_tin_gpu.txt
```
Gửi em 2 file `marason_ketqua.tar.gz` và `thong_tin_gpu.txt`.
Gói kết quả có log, config đã dùng, điểm đánh giá phổ, điểm retrieval và 2 checkpoint mô hình.

Nếu bước nào báo lỗi, gửi em file log của bước đó (`results/marason_msg_simulation/logs/B*.log`).

---

## Lỗi thường gặp
| Lỗi | Cách xử lý |
|---|---|
| `torch không thấy GPU` | chưa `source .venv/bin/activate`, hoặc driver quá cũ |
| `CUDA out of memory` | thêm `--gen-batch 16 --inten-batch 8` (giảm tiếp nếu vẫn lỗi) |
| B0 báo `HTTP 401/403/404` | báo em để em mở quyền dataset, hoặc đặt token Kaggle vào `~/.kaggle/kaggle.json` |
| Server không ra được kaggle.com | tải zip dataset ở máy khác, giải nén vào `~/ms-pred/data/spec_datasets/msg_simulation/`, rồi chạy thêm `--skip-download` |

Em cảm ơn anh/chị!
