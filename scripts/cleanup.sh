#!/usr/bin/env bash
# 清理上一次运行留下的 verl、Ray 和 vLLM 进程。
# 训练失败后，这些进程可能不退出，继续占用内存和显存。
# 注意：会结束所有 verl 任务，包括另一个 tmux 窗口里正在跑的训练。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ -d "$VERL_DIR" ]; then
    (cd "$VERL_DIR" && "${UV_RUN[@]}" ray stop --force) || true
fi

for pattern in "verl.trainer.main_ppo" "multiprocessing.spawn" "multiprocessing.resource_tracker" "VLLM::" "ray::"; do
    pkill -9 -f "$pattern" || true
done
sleep 2

echo "== 内存 =="
free -g
echo "== 显存 =="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "== 剩余的 Python 进程 =="
ps aux --sort=-rss | grep -E "python|VLLM|ray::" | grep -v grep | head -5 || echo "无"
