#!/usr/bin/env bash
# 所有脚本共用的路径和默认参数。不要直接运行，由其他脚本 source。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export WORKSPACE="${WORKSPACE:-/workspace}"
export VERL_COMMIT="${VERL_COMMIT:-b17218b}"
export VERL_DIR="${VERL_DIR:-$WORKSPACE/verl}"
export DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
export CKPT_DIR="${CKPT_DIR:-$WORKSPACE/ckpts}"
export LOG_DIR="${LOG_DIR:-$WORKSPACE/logs}"

# 缓存和 Python 都放持久存储。容器盘在 Pod 停止后会清空，会让 .venv 失效。
export HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORKSPACE/.uv-python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [ -f "$HOME/.local/bin/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi

UV_RUN=(uv run --frozen --all-packages --extra vllm --extra fsdp)

# 以下变量由 verl 的 run_qwen3_8b_fsdp.sh 读取，运行时可以用环境变量覆盖。
export MODEL_PATH="${MODEL_PATH:-$WORKSPACE/models/Qwen3.5-2B}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.3}"
export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export PROJECT_NAME="${PROJECT_NAME:-qwen3_5_2b_lora_grpo}"
# 每张卡一次前向最多处理的 token 数。Qwen3.5 词表很大，logits 很占显存。
# 必须不小于单条样本的最大长度（AIME 评测是 1024 + 4096）。
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}"

require_path() {
    if [ ! -e "$1" ]; then
        echo "找不到 $1。$2" >&2
        exit 1
    fi
}

# 用法：run_verl <实验名> [Hydra 覆盖参数...]
# 后传入的参数优先，所以调用方可以覆盖这里的默认值。
run_verl() {
    local name="$1"
    shift

    require_path "$VERL_DIR" "请先运行 scripts/setup_pod.sh。"
    require_path "$MODEL_PATH" "请先运行 scripts/prepare_data.sh。"
    require_path "$DATA_DIR/gsm8k/train.parquet" "请先运行 scripts/prepare_data.sh。"

    export EXPERIMENT_NAME="$name"
    mkdir -p "$LOG_DIR" "$CKPT_DIR"
    local log_file="$LOG_DIR/$name-$(date +%Y%m%d_%H%M%S).log"
    echo "日志文件：$log_file"

    cd "$VERL_DIR"
    bash examples/tuning/lora/run_qwen3_8b_fsdp.sh \
        "data.train_files=$DATA_DIR/gsm8k/train.parquet" \
        "data.val_files=$DATA_DIR/gsm8k/test.parquet" \
        actor_rollout_ref.model.use_remove_padding=True \
        'actor_rollout_ref.model.exclude_modules=.*visual.*' \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.ref.strategy=fsdp2 \
        actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
        actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
        actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
        trainer.log_val_generations=5 \
        "trainer.default_local_dir=$CKPT_DIR/$name" \
        "$@" 2>&1 | tee "$log_file"
}
