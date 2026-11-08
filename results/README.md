# 实验记录

这里只放小文件：日志、分数和样例输出。每次实验记一行。

| 日期 | 实验名 | 硬件 | 配置变化 | GSM8K | AIME 2026 | 备注 |
|---|---|---|---|---|---|---|
| 2026-11-07 | smoke | RTX 6000 Ada 48GB | 64 条训练、32 条验证，4 步 | 25.0%（32 题，无参考价值） | — | 流程跑通，每步约 47 秒 |
| 2026-11-07 | eval-gsm8k-baseline | RTX 6000 Ada 48GB | 无训练 | **48.4%**（1319 题） | — | 回答上限 1024 token |
| 2026-11-07 | eval-aime26-baseline | RTX 6000 Ada 48GB | 无训练 | — | **3.3%**（1/30） | 回答上限 4096 token |

共同设置：verl `b17218b`，Qwen3.5-2B，LoRA rank 32 / alpha 32，关闭思考模式，验证时每题只生成一次回答。

日志在 Pod 的 `/workspace/logs/`：`smoke-20261108_*.log`、`eval-gsm8k-baseline-*.log`、`eval-aime26-baseline-*.log`（时间为 UTC）。

## 2026-11-07 试跑记录

### 结果

- **试跑：** 4 步训练、验证和保存 checkpoint 都完成。奖励在 0 和 1 之间变化，说明奖励函数正常工作。
  - 回答平均 392 token，约 9% 写满 1024 token 被截断。
  - 每步耗时：生成约 10 秒，更新参数约 7 秒，保存 checkpoint 约 24 秒。
- **GSM8K 基线 48.4%：** 可能低估了模型的实际能力，原因有三：
  - 评分只认 `#### 数字` 格式。
  - 一部分回答被截断。
  - 关闭了思考模式。

  训练初期的分数上涨可能主要来自格式和长度的改善，需要对照回答样例判断。
- **AIME 基线 3.3%：** 只答对 1 题，每题约占 3.3 分。训练后多对 1 题就会显得进步很大，要谨慎解读。
- **评测结束时的 SIGSEGV：** AIME 评测结束时，奖励计算进程报了 SIGSEGV，出现在指标打印之后，不影响结果。

### 排错经过

| 现象 | 原因 | 解决 |
|---|---|---|
| RTX 4090：vLLM 报 `No available memory for the cache blocks` | 24 GB 显存下，默认 `ROLLOUT_GPU_MEM_UTIL=0.3` 只给 vLLM 约 7 GB | 调高该值并开启 offload 后，又遇到下一行的内存不足 |
| RTX 4090：Ray 报内存不足（`OutOfMemoryError`） | Pod 内存上限约 38 GB，offload 把参数放进内存后超限 | 换成 48 GB 显卡 |
| 失败后重跑，内存或显存莫名不够 | 上次失败运行留下的进程没有退出，占用约 10 GB 内存 | 重跑前运行 `scripts/cleanup.sh` |
| RTX 6000 Ada：计算熵时 `CUDA out of memory` | Qwen3.5 词表很大；`use_remove_padding=False` 时不会分块计算熵，整块 logits 一次性展开 | 改为 `use_remove_padding=True`，开启分块计算熵，每卡 token 上限设为 6144（已写入 `scripts/common.sh`） |

### 下一步

- 推送本地修改，在 Pod 上 `git pull`。
- 用 `bash scripts/train.sh; runpodctl stop pod $RUNPOD_POD_ID` 跑 1 个 epoch 正式训练，粗估 4–6 小时。
- 训练后用 `CHECKPOINT=... bash scripts/eval.sh` 重新评测 GSM8K 和 AIME，并对比回答样例。
