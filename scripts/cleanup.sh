#!/usr/bin/env bash
# Kill verl, Ray, and vLLM processes left over from a previous run.
# After a failed run these processes may not exit and keep holding RAM and GPU memory.
# Warning: this kills ALL verl jobs, including a training run in another tmux window.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ -d "$VERL_DIR" ]; then
    (cd "$VERL_DIR" && "${UV_RUN[@]}" ray stop --force) || true
fi

for pattern in "verl.trainer.main_ppo" "multiprocessing.spawn" "multiprocessing.resource_tracker" "VLLM::" "ray::"; do
    pkill -9 -f "$pattern" || true
done
sleep 2

echo "== RAM =="
free -g
echo "== GPU memory =="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "== Remaining Python processes =="
ps aux --sort=-rss | grep -E "python|VLLM|ray::" | grep -v grep | head -5 || echo "none"
