# 项目说明（给 AI 助手）

用 verl 在单张 L40S 上对 Qwen3.5-2B 做 LoRA + GRPO 训练，比较训练前后的评测分数，并对照 DeepSeek-R1 论文尝试复现。

## 约定

- 本机是 Windows，只放代码和小型结果，不在本地训练。
- 训练在 RunPod 上运行，持久数据放 `/workspace`，容器盘的内容在 Pod 停止后会丢失。
- 所有路径和默认参数集中在 `scripts/common.sh`，不要在其他脚本里重复写。
- verl 固定在 `scripts/common.sh` 的 `VERL_COMMIT`，不要提交到本仓库。升级时记录原因并重新试跑。
- 模型、数据、checkpoint 和密钥不进 Git。`results/` 只放日志、分数和样例输出。
- 不要声称某个脚本已跑通，除非 `results/` 或实验记录里有对应日志。

## 实验顺序

`smoke_test.sh` → `eval.sh`（基线）→ `train.sh` → `eval.sh`（带适配器）

先跑通数学任务。LiveCodeBench 等数学流程稳定后再接入。
