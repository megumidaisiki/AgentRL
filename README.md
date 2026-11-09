# Qwen3.5-2B LoRA + GRPO 入门项目

在 RunPod 单张 L40S 上，用 verl 训练 Qwen3.5-2B 的 LoRA 适配器，比较训练前后的评测分数。

> 状态：试跑和基线评测已在 48GB 显卡上跑通，结果见 [results/README.md](results/README.md)。

## 文件说明

| 文件 | 作用 |
|---|---|
| `scripts/common.sh` | 路径、固定的 verl 版本、默认参数 |
| `scripts/setup_pod.sh` | 在 Pod 上安装环境 |
| `scripts/prepare_data.sh` | 下载模型，准备 GSM8K 和 AIME 2026 |
| `scripts/prepare_aime26.py` | 把 AIME 2026 转成 verl 格式 |
| `scripts/smoke_test.sh` | 小规模试跑 |
| `scripts/eval.sh` | 只评测不训练 |
| `scripts/train.sh` | 正式训练 |
| `results/` | 实验记录和传回的小文件 |

## 1. 准备 RunPod

1. 把本机公钥加到 RunPod 的 **Settings → SSH Public Keys**。
2. 新建 100 GB 的 Network Volume，选有 L40S 库存的数据中心。
3. 部署 Pod：1 张 L40S，On-Demand，Runpod PyTorch 模板，容器盘 40 GB，挂载上面的 Volume。如果有 CUDA 筛选项，选 13.0 或以上。
4. 复制 Connect 页的 **SSH over exposed TCP** 命令，在本地终端连接。

## 2. 把代码放到 Pod 上

推荐先推到私有 GitHub 仓库，再在 Pod 上克隆：

```bash
cd /workspace
git clone https://github.com/<你的用户名>/<仓库名>.git project
cd project
```

## 3. 运行

先开一个 tmux 会话，SSH 断开后任务不会中断。第一次运行时 tmux 还没安装，所以先装好环境：

```bash
bash scripts/setup_pod.sh
tmux new -s train
```

断开后用 `tmux attach -t train` 回来。在 tmux 里依次运行：

```bash
bash scripts/prepare_data.sh
bash scripts/smoke_test.sh
bash scripts/eval.sh gsm8k
bash scripts/eval.sh aime26
bash scripts/train.sh
```

训练完成后，加载 checkpoint 再评测一次：

```bash
CHECKPOINT=/workspace/ckpts/<实验名>/global_step_<步数> bash scripts/eval.sh gsm8k
CHECKPOINT=/workspace/ckpts/<实验名>/global_step_<步数> bash scripts/eval.sh aime26
```

用同一个 `RUN_NAME` 再运行 `train.sh`，verl 会自动从最新的 checkpoint 继续。想重新开始就换一个名字。

## 4. 看结果

- **训练中看 GRPO 组内情况：** 训练回答会写到 `/workspace/logs/rollout_generations/<实验名>/`。另开一个窗口运行 `python3 scripts/group_stats.py /workspace/logs/rollout_generations/<实验名> --last 5`，可以看到全错组、全对组的比例，以及回答长度和疑似循环比例。全对组越来越多，说明题目对模型太简单。
- **传回本地后完整分析：** `python scripts/analyze_generations.py results/<目录>`（思考模式加 `--think`），会打印训练前后的对比表格，并生成训练曲线。

- **分数：** 日志里 `val` 开头的指标。AIME 的奖励是 1 或 -1，准确率看 `acc`。
- **模型输出：** `log_val_generations` 会打印几条样例，一定要读，分数可能掩盖格式问题。
- **W&B：** 在 RunPod 的环境变量或 Secrets 里设置 `WANDB_API_KEY`，`train.sh` 会自动启用。不要把密钥写进代码。

## 5. 传回结果并关机

在本地终端运行，端口和 IP 在 Connect 页：

```bash
scp -P <端口> -r root@<IP>:/workspace/logs F:/a_Agentic_RL/results/
```

如果要保留适配器，先在 Pod 上把 checkpoint 导出成标准格式，再用 `scp` 传回 `lora_adapter` 目录。这条命令还没有验证过，verl 的合并工具是否支持这里用的 `fsdp2` checkpoint，要实际跑一次才知道：

```bash
cd /workspace/verl
uv run --frozen --all-packages --extra vllm --extra fsdp python -m verl.model_merger merge --backend fsdp --local_dir /workspace/ckpts/<实验名>/global_step_<步数>/actor --target_dir /workspace/export/<实验名>
```

确认文件已传回后，**Terminate Pod**。Volume 上的数据会保留，也会继续收存储费。项目结束后记得删除它。

## 常见问题

- **脚本报 `$'\r': command not found`：** 文件变成了 CRLF 换行。在 Pod 上运行 `sed -i 's/\r$//' scripts/*.sh`。
- **运行失败后再跑就内存或显存不足：** 上一次的进程可能没有退出，先运行 `bash scripts/cleanup.sh` 清理。它会结束所有 verl 任务，有训练在跑时不要用。
- **显存不足：** 依次调小参数，例如 `ROLLOUT_GPU_MEM_UTIL=0.25 TRAIN_BATCH_SIZE=32 ROLLOUT_N=4 bash scripts/train.sh`。
- **torch 找不到 GPU：** `nvidia-smi` 显示的 CUDA 版本可能低于 13.0，换一台机器重新部署。
- **Pod 重启后找不到 `uv` 或 `tmux`：** 再运行一次 `setup_pod.sh`，`/workspace` 里的内容不用重新下载。

## 解读结果时注意

- GSM8K 的提示词要求 `####`，AIME 的提示词要求 `\boxed{}`，两者评分规则不同。
- AIME 2026 只有 30 题，每题约 3.3 分，要和 GSM8K 测试集及输出样例一起看。
- Qwen3.5-2B 默认不开启思考模式。所有脚本都支持 `THINKING=1` 打开思考模式（实际参数是 `+data.apply_chat_template_kwargs.enable_thinking=True`，前面的 `+` 不能省）。打开后 GSM8K 回答上限变为 4096、AIME 变为 8192，验证改用模型卡推荐的采样参数（temperature 1.0、top_p 0.95、top_k 20），AIME 每题采样 8 次。训练和评测必须用同一个模式。官方提示 2B 开启后容易陷入循环，换模式后先跑一次 `THINKING=1 bash scripts/smoke_test.sh`。
