#!/usr/bin/env bash
# Full GRPO training on GSM8K. Defaults to 1 epoch; check the timing before running more:
#   TOTAL_EPOCHS=3 RUN_NAME=gsm8k-r32-e3 bash scripts/train.sh
#   THINKING=1 TOTAL_EPOCHS=3 bash scripts/train.sh    # thinking mode
#
# Every training rollout is written to $LOG_DIR/rollout_generations/<experiment>/<step>.jsonl;
# scripts/group_stats.py reports the share of all-correct / all-wrong GRPO groups from these files.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-25}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$GSM8K_RESPONSE_LENGTH}"

if [ -n "${WANDB_API_KEY:-}" ]; then
    logger='trainer.logger=["console","wandb"]'
else
    echo "WANDB_API_KEY is not set; metrics go to the console and the log file only."
    logger='trainer.logger=["console"]'
fi

name="${RUN_NAME:-gsm8k-r${LORA_RANK}${MODE_SUFFIX}-$(date +%m%d_%H%M)}"
run_verl "$name" \
    "$logger" \
    "trainer.rollout_data_dir=$LOG_DIR/rollout_generations/$name" \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=2 \
    "$@"
