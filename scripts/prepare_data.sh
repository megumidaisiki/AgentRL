#!/usr/bin/env bash
# 下载模型并准备 GSM8K 和 AIME 2026 数据。文件都保存在 /workspace。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_path "$VERL_DIR" "请先运行 scripts/setup_pod.sh。"
cd "$VERL_DIR"

"${UV_RUN[@]}" hf download Qwen/Qwen3.5-2B --local-dir "$MODEL_PATH"

"${UV_RUN[@]}" python examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR/gsm8k"

"${UV_RUN[@]}" python "$PROJECT_DIR/scripts/prepare_aime26.py" --output_dir "$DATA_DIR/aime26"

echo "模型和数据已就绪。"
