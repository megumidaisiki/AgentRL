#!/usr/bin/env bash
# Install the environment on the Pod. Run once after creating or restarting a Pod; safe to re-run.
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

# The first run creates .venv from uv.lock, which takes a while.
"${UV_RUN[@]}" python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see a GPU. Check that nvidia-smi reports CUDA 13.0 or newer.")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Environment ready."
