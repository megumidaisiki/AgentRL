#!/usr/bin/env bash
# Evaluate only, no training. Without CHECKPOINT this measures the pre-training baseline.
#
# Usage:
#   bash scripts/eval.sh gsm8k
#   bash scripts/eval.sh aime26
#   CHECKPOINT=/workspace/ckpts/<experiment>/global_step_<step> bash scripts/eval.sh aime26
#   THINKING=1 bash scripts/eval.sh gsm8k    # thinking mode; must match the mode used in training
#
# AIME has only 30 problems, so a single response per problem is very noisy. In thinking mode each
# problem is sampled AIME_N=8 times by default and the scores are averaged.
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
        echo "Usage: bash scripts/eval.sh <gsm8k|aime26> [Hydra overrides...]" >&2
        exit 1
        ;;
esac
require_path "$val_file" "Run scripts/prepare_data.sh first."

if [ -n "${CHECKPOINT:-}" ]; then
    require_path "$CHECKPOINT" "Check the CHECKPOINT path."
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
