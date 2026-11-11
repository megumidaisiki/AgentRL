#!/usr/bin/env bash
# Download the model and prepare the GSM8K and AIME 2026 data. Everything is stored under /workspace.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_path "$VERL_DIR" "Run scripts/setup_pod.sh first."
cd "$VERL_DIR"

"${UV_RUN[@]}" hf download Qwen/Qwen3.5-2B --local-dir "$MODEL_PATH"

"${UV_RUN[@]}" python examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR/gsm8k"

"${UV_RUN[@]}" python "$PROJECT_DIR/scripts/prepare_aime26.py" --output_dir "$DATA_DIR/aime26"

echo "Model and data ready."
