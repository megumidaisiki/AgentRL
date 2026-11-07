#!/usr/bin/env bash
# 小规模试跑：确认训练、验证和保存 checkpoint 能跑通，分数没有参考价值。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TOTAL_EPOCHS=1
export SAVE_FREQ=2
export TEST_FREQ=2

run_verl smoke \
    data.train_max_samples=64 \
    data.val_max_samples=32 \
    'trainer.logger=["console"]' \
    trainer.val_before_train=True \
    trainer.resume_mode=disable \
    "$@"
