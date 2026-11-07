#!/usr/bin/env bash
# 在 Pod 上安装环境。每次新建或重新启动 Pod 后运行一次，重复运行是安全的。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

apt-get update
apt-get install -y tmux git curl

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi

mkdir -p "$WORKSPACE"
if [ ! -d "$VERL_DIR/.git" ]; then
    git clone https://github.com/verl-project/verl.git "$VERL_DIR"
fi

cd "$VERL_DIR"
git fetch origin
git checkout "$VERL_COMMIT"

nvidia-smi

# 首次运行会根据 uv.lock 创建 .venv，需要一些时间。
"${UV_RUN[@]}" python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch 找不到 GPU。请检查 nvidia-smi 显示的 CUDA 版本是否不低于 13.0。")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "环境已就绪。"
