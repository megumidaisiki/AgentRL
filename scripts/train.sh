#!/usr/bin/env bash
# 在 GSM8K 上做正式训练。默认只跑 1 个 epoch，看清耗时后再加量：
#   TOTAL_EPOCHS=3 RUN_NAME=gsm8k-r32-e3 bash scripts/train.sh
#   THINKING=1 TOTAL_EPOCHS=3 bash scripts/train.sh    # 打开思考模式
#
# 训练中每一步的全部回答写到 $LOG_DIR/rollout_generations/<实验名>/<步数>.jsonl，
# 可以用 scripts/group_stats.py 统计 GRPO 组内全对、全错的比例。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-25}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$GSM8K_RESPONSE_LENGTH}"

if [ -n "${WANDB_API_KEY:-}" ]; then
    logger='trainer.logger=["console","wandb"]'
else
    echo "没有设置 WANDB_API_KEY，本次只把指标输出到控制台和日志文件。"
    logger='trainer.logger=["console"]'
fi

name="${RUN_NAME:-gsm8k-r${LORA_RANK}${MODE_SUFFIX}-$(date +%m%d_%H%M)}"
run_verl "$name" \
    "$logger" \
    "trainer.rollout_data_dir=$LOG_DIR/rollout_generations/$name" \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=2 \
    "$@"
