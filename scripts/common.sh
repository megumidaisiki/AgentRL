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
# 必须不小于单条样本的最大长度。run_verl 会在需要时自动调大（例如思考模式下的 AIME 评测）。
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}"
# verl 会开很多辅助进程（TransferQueue 存储、奖励、rollout 调度各 8 个），Ray 默认每个分 1 个 CPU。
# Pod 的 CPU 核数不够时，剩下的进程会一直排队，任务卡住不报错（A40 Pod 只有 7 核）。
# 所以减少进程数，并让 Ray 按 32 个 CPU 分配（这些进程大多空闲，不会真的占满 CPU）。
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export TQ_STORAGE_UNITS="${TQ_STORAGE_UNITS:-2}"
export REWARD_WORKERS="${REWARD_WORKERS:-2}"
export AGENT_WORKERS="${AGENT_WORKERS:-2}"

# 思考模式：THINKING=1 时打开 Qwen3.5 的思考模式，训练和评测必须一致。
# 思考会让回答长很多，所以回答上限跟着调大。验证改用 Qwen 推荐的采样参数，
# 因为思考模式下用贪心解码容易陷入重复。
export THINKING="${THINKING:-0}"
if [ "$THINKING" = "1" ]; then
    export GSM8K_RESPONSE_LENGTH="${GSM8K_RESPONSE_LENGTH:-4096}"
    export AIME_RESPONSE_LENGTH="${AIME_RESPONSE_LENGTH:-8192}"
    MODE_SUFFIX="-think"
    MODE_ARGS=(
        +data.apply_chat_template_kwargs.enable_thinking=True
        actor_rollout_ref.rollout.val_kwargs.do_sample=True
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6
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
        echo "找不到 $1。$2" >&2
        exit 1
    fi
}

# 用法：run_verl <实验名> [Hydra 覆盖参数...]
# 每次验证的全部回答写到 $LOG_DIR/val_generations/<实验名>/<步数>.jsonl。
# 只用控制台日志时，log_val_generations 不会打印样例，所以要靠这个文件看模型输出。
# 后传入的参数优先，所以调用方可以覆盖这里的默认值。
run_verl() {
    local name="$1"
    shift

    require_path "$VERL_DIR" "请先运行 scripts/setup_pod.sh。"
    require_path "$MODEL_PATH" "请先运行 scripts/prepare_data.sh。"
    require_path "$DATA_DIR/gsm8k/train.parquet" "请先运行 scripts/prepare_data.sh。"

    export EXPERIMENT_NAME="$name"
    # 每卡 token 上限不能小于一条样本的最大长度，否则 verl 会报错。
    local need=$(( ${MAX_PROMPT_LENGTH:-512} + ${MAX_RESPONSE_LENGTH:-1024} ))
    if [ "$PPO_MAX_TOKEN_LEN_PER_GPU" -lt "$need" ]; then
        export PPO_MAX_TOKEN_LEN_PER_GPU="$need"
    fi
    echo "思考模式：$THINKING，回答上限：${MAX_RESPONSE_LENGTH:-1024}，每卡 token 上限：$PPO_MAX_TOKEN_LEN_PER_GPU"
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
