#!/usr/bin/env bash
# 小规模试跑：确认训练、验证和保存 checkpoint 能跑通，分数没有参考价值。
# 换了设置（例如 THINKING=1）后先跑一次，看清每步耗时和回答长度再开正式训练。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TOTAL_EPOCHS=1
export SAVE_FREQ=2
export TEST_FREQ=2
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$GSM8K_RESPONSE_LENGTH}"

name="smoke$MODE_SUFFIX"
run_verl "$name" \
    "trainer.rollout_data_dir=$LOG_DIR/rollout_generations/$name" \
    data.train_max_samples=64 \
    data.val_max_samples=32 \
    'trainer.logger=["console"]' \
    trainer.val_before_train=True \
    trainer.resume_mode=disable \
    "$@"
