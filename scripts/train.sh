#!/usr/bin/env bash
# 在 GSM8K 上做正式训练。默认只跑 1 个 epoch，看清耗时后再加量：
#   TOTAL_EPOCHS=3 RUN_NAME=gsm8k-r32-e3 bash scripts/train.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-25}"

if [ -n "${WANDB_API_KEY:-}" ]; then
    logger='trainer.logger=["console","wandb"]'
else
    echo "没有设置 WANDB_API_KEY，本次只把指标输出到控制台和日志文件。"
    logger='trainer.logger=["console"]'
fi

run_verl "${RUN_NAME:-gsm8k-r${LORA_RANK}-$(date +%m%d_%H%M)}" \
    "$logger" \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=2 \
    "$@"
