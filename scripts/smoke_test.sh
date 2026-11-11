#!/usr/bin/env bash
# Small trial run: checks that training, validation, and checkpointing work. The scores are meaningless.
# After changing a setting (e.g. THINKING=1), run this first to see per-step time and response length.
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
