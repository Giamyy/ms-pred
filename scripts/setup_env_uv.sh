#!/usr/bin/env bash
# Create the ms-pred environment (.venv in the repo root) with uv, once.
#
#   bash scripts/setup_env_uv.sh            # pick cu124 / cu118 / cpu from nvidia-smi
#   bash scripts/setup_env_uv.sh cu118      # or force an extra: cu124 | cu118 | cpu
#
# Linux or WSL2 only: DGL has no CUDA wheels for torch 2.6 on native Windows.
# Re-running is safe; uv only installs what changed (e.g. after a git pull).
# Afterwards: source .venv/bin/activate

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ ! -f pyproject.toml ]]; then
    echo "Không thấy pyproject.toml trong $REPO_DIR" >&2
    exit 1
fi

# gcc is needed to build the Cython extensions of ms_pred
if ! command -v gcc >/dev/null 2>&1; then
    echo "Chưa có gcc. Cài bằng: sudo apt update && sudo apt install -y build-essential git" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "==> Cài uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

EXTRA="${1:-}"
if [[ -z "$EXTRA" ]]; then
    CUDA_VER=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_VER="$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' || true)"
    fi
    if [[ -z "$CUDA_VER" ]]; then
        EXTRA=cpu
    else
        major="${CUDA_VER%%.*}"
        minor="${CUDA_VER##*.}"
        if (( major >= 12 )); then
            EXTRA=cu124
        elif (( major == 11 && minor >= 8 )); then
            EXTRA=cu118
        else
            EXTRA=cpu
        fi
    fi
    echo "==> Driver hỗ trợ CUDA: ${CUDA_VER:-không có GPU} -> dùng --extra $EXTRA"
fi

case "$EXTRA" in
    cu124|cu118|cpu) ;;
    *) echo "Extra không hợp lệ: $EXTRA (chọn cu124 | cu118 | cpu)" >&2; exit 1 ;;
esac

echo "==> uv sync --extra $EXTRA --python $PYTHON_VERSION"
for attempt in 1 2 3; do
    if uv sync --extra "$EXTRA" --python "$PYTHON_VERSION"; then
        break
    fi
    if (( attempt == 3 )); then
        echo "uv sync thất bại sau 3 lần" >&2
        exit 1
    fi
    echo "uv sync lỗi (lần $attempt/3), thử lại sau 10 giây..."
    sleep 10
done

echo "==> Kiểm tra"
.venv/bin/python - <<'EOF'
import sys
import torch, dgl, torch_scatter, rdkit, ms_pred
print("python", sys.version.split()[0])
print("torch ", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  ", i, torch.cuda.get_device_name(i))
print("dgl   ", dgl.__version__, "| rdkit", rdkit.__version__)
EOF

echo
echo "Xong. Mỗi lần dùng: cd $REPO_DIR && source .venv/bin/activate"
