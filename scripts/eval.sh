#!/usr/bin/env bash
# 只评测不训练。不传 CHECKPOINT 时，测的是训练前的基线。
#
# 用法：
#   bash scripts/eval.sh gsm8k
#   bash scripts/eval.sh aime26
#   CHECKPOINT=/workspace/ckpts/<实验名>/global_step_<步数> bash scripts/eval.sh aime26
#   THINKING=1 bash scripts/eval.sh gsm8k    # 打开思考模式，训练时开了，评测也要开
#
# AIME 只有 30 题，单次回答的分数波动很大。思考模式下默认每题采样 AIME_N=8 次取平均。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

task="${1:-}"
if [ "$#" -gt 0 ]; then
    shift
fi

task_args=()
case "$task" in
    gsm8k)
        val_file="$DATA_DIR/gsm8k/test.parquet"
        export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
        export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$GSM8K_RESPONSE_LENGTH}"
        ;;
    aime26)
        val_file="$DATA_DIR/aime26/test.parquet"
        export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
        export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$AIME_RESPONSE_LENGTH}"
        if [ "$THINKING" = "1" ]; then
            task_args+=("actor_rollout_ref.rollout.val_kwargs.n=${AIME_N:-8}")
        fi
        ;;
    *)
        echo "用法：bash scripts/eval.sh <gsm8k|aime26> [Hydra 覆盖参数...]" >&2
        exit 1
        ;;
esac
require_path "$val_file" "请先运行 scripts/prepare_data.sh。"

if [ -n "${CHECKPOINT:-}" ]; then
    require_path "$CHECKPOINT" "请检查 CHECKPOINT 路径。"
    resume=(trainer.resume_mode=resume_path "trainer.resume_from_path=$CHECKPOINT")
    label=trained
else
    resume=(trainer.resume_mode=disable)
    label=baseline
fi

run_verl "eval-$task-$label$MODE_SUFFIX" \
    "data.val_files=$val_file" \
    'trainer.logger=["console"]' \
    trainer.val_before_train=True \
    trainer.val_only=True \
    "${resume[@]}" \
    "${task_args[@]}" \
    "$@"
