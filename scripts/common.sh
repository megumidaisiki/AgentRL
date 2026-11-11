#!/usr/bin/env bash
# Shared paths and default parameters. Do not run directly; other scripts source this file.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export WORKSPACE="${WORKSPACE:-/workspace}"
export VERL_COMMIT="${VERL_COMMIT:-b17218b}"
export VERL_DIR="${VERL_DIR:-$WORKSPACE/verl}"
export DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
export CKPT_DIR="${CKPT_DIR:-$WORKSPACE/ckpts}"
export LOG_DIR="${LOG_DIR:-$WORKSPACE/logs}"

# Keep caches and the Python install on persistent storage. The container disk is wiped
# when a Pod stops, which would break .venv.
export HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORKSPACE/.uv-python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [ -f "$HOME/.local/bin/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi

UV_RUN=(uv run --frozen --all-packages --extra vllm --extra fsdp)

# The variables below are read by verl's run_qwen3_8b_fsdp.sh and can be overridden via environment variables.
export MODEL_PATH="${MODEL_PATH:-$WORKSPACE/models/Qwen3.5-2B}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.3}"
export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export PROJECT_NAME="${PROJECT_NAME:-qwen3_5_2b_lora_grpo}"
# Max tokens per GPU in one forward pass. Qwen3.5 has a very large vocabulary, so logits use a lot of memory.
# Must be at least the longest single sample; run_verl raises it automatically when needed
# (e.g. AIME evaluation in thinking mode).
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}"
# verl starts many helper actors (8 each for TransferQueue storage, reward, and rollout agents), and Ray
# reserves 1 CPU for each by default. On Pods with few cores the remaining actors queue forever and the job
# hangs without an error (an A40 Pod had only 7 cores). So we start fewer actors and tell Ray there are
# 32 CPUs (these actors are mostly idle and do not actually saturate the CPU).
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export TQ_STORAGE_UNITS="${TQ_STORAGE_UNITS:-2}"
export REWARD_WORKERS="${REWARD_WORKERS:-2}"
export AGENT_WORKERS="${AGENT_WORKERS:-2}"

# Thinking mode: THINKING=1 enables Qwen3.5's thinking mode. Training and evaluation must use the same mode.
# Thinking makes responses much longer, so the response limits are raised. Validation uses the sampling
# parameters the Qwen3.5 model card recommends for text tasks in thinking mode (temperature 1.0, top_p 0.95,
# top_k 20), because greedy decoding tends to loop in thinking mode.
# The model card also recommends presence_penalty 1.5, but verl's validation config does not expose it.
# The model card recommends 32768 output tokens (81920 for hard problems); budget limits force much shorter
# limits here, and truncation lowers the scores.
export THINKING="${THINKING:-0}"
if [ "$THINKING" = "1" ]; then
    export GSM8K_RESPONSE_LENGTH="${GSM8K_RESPONSE_LENGTH:-4096}"
    export AIME_RESPONSE_LENGTH="${AIME_RESPONSE_LENGTH:-8192}"
    MODE_SUFFIX="-think"
    MODE_ARGS=(
        +data.apply_chat_template_kwargs.enable_thinking=True
        actor_rollout_ref.rollout.val_kwargs.do_sample=True
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95
        actor_rollout_ref.rollout.val_kwargs.top_k=20
    )
else
    export GSM8K_RESPONSE_LENGTH="${GSM8K_RESPONSE_LENGTH:-1024}"
    export AIME_RESPONSE_LENGTH="${AIME_RESPONSE_LENGTH:-4096}"
    MODE_SUFFIX=""
    MODE_ARGS=()
fi

require_path() {
    if [ ! -e "$1" ]; then
        echo "Not found: $1. $2" >&2
        exit 1
    fi
}

# Usage: run_verl <experiment name> [Hydra overrides...]
# Every validation response is written to $LOG_DIR/val_generations/<experiment>/<step>.jsonl.
# With only the console logger, log_val_generations prints no samples, so this file is how we read outputs.
# Later arguments take precedence, so callers can override the defaults here.
run_verl() {
    local name="$1"
    shift

    require_path "$VERL_DIR" "Run scripts/setup_pod.sh first."
    require_path "$MODEL_PATH" "Run scripts/prepare_data.sh first."
    require_path "$DATA_DIR/gsm8k/train.parquet" "Run scripts/prepare_data.sh first."

    export EXPERIMENT_NAME="$name"
    # The per-GPU token limit must be at least the longest sample, otherwise verl errors out.
    local need=$(( ${MAX_PROMPT_LENGTH:-512} + ${MAX_RESPONSE_LENGTH:-1024} ))
    if [ "$PPO_MAX_TOKEN_LEN_PER_GPU" -lt "$need" ]; then
        export PPO_MAX_TOKEN_LEN_PER_GPU="$need"
    fi
    echo "Thinking mode: $THINKING, max response length: ${MAX_RESPONSE_LENGTH:-1024}, max tokens per GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"
    mkdir -p "$LOG_DIR" "$CKPT_DIR"
    local log_file="$LOG_DIR/$name-$(date +%Y%m%d_%H%M%S).log"
    echo "Log file: $log_file"

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
        "ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS" \
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=$TQ_STORAGE_UNITS" \
        "reward.num_workers=$REWARD_WORKERS" \
        "actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS" \
        trainer.log_val_generations=5 \
        "trainer.validation_data_dir=$LOG_DIR/val_generations/$name" \
        "trainer.default_local_dir=$CKPT_DIR/$name" \
        "${MODE_ARGS[@]}" \
        "$@" 2>&1 | tee "$log_file"
}
