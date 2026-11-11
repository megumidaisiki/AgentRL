# Experiment Log

This directory holds only small files: logs, scores, and sample outputs. One row per experiment. All results were produced on a single NVIDIA L40S (48 GB).

| Date | Run | Setup | GSM8K | AIME 2026 | Notes |
|---|---|---|---|---|---|
| 2026-11-08 | smoke | 64 train / 32 val prompts, 4 steps | 34.4% → 43.8% (32 problems, not meaningful) | — | Pipeline works; ~25 s per step once warmed up |
| 2026-11-08 | eval-*-baseline (1st) | No training | 48.4% | 13.3% (4/30) | Responses not saved |
| 2026-11-09 | eval-*-baseline (2nd, responses saved) | No training | **49.1%** | **3.3%** (1/30) | Pre-training reference |
| 2026-11-09 | gsm8k-r32-e1 | 1 epoch, 116 steps, ~2.5 h | 49.1% → 72.0% | — | In-training validation |
| 2026-11-09 | eval-*-trained (step 116) | Trained LoRA adapter | **72.0%** (strict) / 73.1% (lenient) | **3.3%** (1/30) | Gain is almost entirely format; see below |
| 2026-11-09 | smoke-think | Thinking mode, 4096-token limit | — | — | 64% of responses hit the 4096 limit |
| 2026-11-10 | eval-*-baseline-think | Thinking mode, no training, 8192-token limit | **44.7%** (strict) / 71.7% (lenient) | **0%** (8 samples each, 0/240) | See "Thinking-mode baseline" |

Shared setup for the non-thinking runs: verl `b17218b`, Qwen3.5-2B, LoRA rank 32 / alpha 32, thinking mode off, greedy decoding at validation with one response per problem. Response limits: 1024 tokens for GSM8K, 4096 for AIME.

- Logs from the 2026-11-08 smoke test and first baseline: [`l40s/`](l40s/).
- Training and before/after evaluation logs and responses from 2026-11-09: [`run1/`](run1/). Validation responses saved every 25 steps during training (~11 MB) are kept locally and not committed.
- Thinking-mode logs: [`think/`](think/).
- Ray DEBUG lines were stripped from the logs; timestamps are UTC. Lines printed by the scripts themselves may be in Chinese because the logs predate the English translation of the scripts.
- To reproduce the tables and curves: `python scripts/analyze_generations.py results/run1`.

## Run 1: GRPO on GSM8K (gsm8k-r32-e1)

Setup: GRPO, 64 prompts × 5 responses per step, 1 epoch (116 steps), learning rate 3e-6, KL loss coefficient 0.001 (`low_var_kl`). Training took ~2.5 h; with evaluation, ~5.3 h total at $1.09/h.

![Training curves](run1/training_curves.png)

### Main finding: the score gain came almost entirely from answer format

verl's GSM8K grader only accepts `#### <number>`, **with a space after the hashes**. The base model often wrote `####17` or `#### $18,000`, which score 0 even when the number is right.

| Category (1319 problems) | Before | After |
|---|---|---|
| Correct (strict grader) | 647 | 950 |
| Right answer, format rejected by grader | 349 | 0 |
| No `####`, but the last number is right | 4 | 14 |
| Format OK, wrong answer | 171 | 333 |
| Bad format, wrong answer | 86 | 2 |
| No `####`, wrong answer | 28 | 12 |
| No `####`, likely truncated | 34 | 8 |
| **Strict accuracy (verl grader)** | **49.1%** | **72.0%** |
| **Lenient accuracy (answer only)** | **75.8%** | **73.1%** |
| Response length (chars, mean / median) | 1158 / 1038 | 866 / 793 |
| Responses with reflection words ("wait", "verify", ...) | 9.2% | 3.7% |

- Within the first 25 steps, strict accuracy jumped from 49% to 72%, while "right answer, bad format" fell from 365 to 33. Strict accuracy barely moved over the remaining 90 steps.
- Judged on the answer alone, accuracy dropped slightly from 75.8% to 73.1%: 103 problems went from wrong to right and 139 from right to wrong. **The model did not get better at math.**
- Among the problems that flipped to wrong, a common pattern is compressed reasoning that drops a condition. In one example the reference answer is 36; the base model got it right, while the trained model forgot to multiply by "3 times a week" and answered 12.

### Validation during training (GSM8K test set)

| step | strict acc | lenient acc | right answer, bad format | likely truncated | mean length (chars) | reflection words |
|---|---|---|---|---|---|---|
| 0 | 49.1% | 76.7% | 365 | 41 | 1171 | 9.4% |
| 25 | 72.0% | 74.5% | 33 | 35 | 1131 | 8.1% |
| 50 | 72.3% | 72.6% | 3 | 32 | 1060 | 7.1% |
| 75 | 72.9% | 72.9% | 0 | 16 | 957 | 4.5% |
| 100 | 71.4% | 71.9% | 6 | 10 | 874 | 3.2% |
| 116 | 72.1% | 73.2% | 14 | 9 | 871 | 3.7% |

### Training dynamics

| Metric | First 5 steps → last 5 steps | Interpretation |
|---|---|---|
| Training reward | 0.36 → 0.70 | Rises mostly in the first 30 steps, matching when the format was learned |
| Response length (tokens) | 486 → 288 | Keeps shrinking, the **opposite** of the growing responses in the DeepSeek-R1 paper |
| Truncation rate | 17% → 1.7% | Responses hitting the 1024 limit almost disappear |
| Policy entropy | 0.43 → 0.14 | Keeps falling; outputs become more deterministic |
| KL to reference model | 0.007 → 1.5 | Rises quickly after step 60; the policy drifts further from the base model |
| Time per step | 88 s → 61 s | Shorter responses make generation faster |

- After the format was learned, reward stopped rising, but entropy kept falling and KL kept rising. The second half of training mostly **compressed responses** rather than teaching new problem-solving ability. A KL coefficient of 0.001 provides almost no constraint.
- More training would most likely keep increasing KL with no benefit.
- Gradient norm grew from 0.095 to 1.19, another sign of increasingly aggressive updates late in training.

### AIME

| | Before | After |
|---|---|---|
| Correct | 1/30 | 1/30 |
| No `\boxed{}` (mostly hit the 4096 limit) | 26 | 21 |
| Mean response length (chars) | 9371 | 8283 |
| Responses with reflection words | 96.7% | 76.7% |

- **A single greedy AIME run cannot be used for comparison**: on the same GPU type with identical settings, two baselines scored 4/30 and 1/30. vLLM's computation order varies slightly between runs, and the differences compound over 4096-token responses. GSM8K responses are short, so its two baselines differed by only 8 problems (48.4% vs. 49.1%).
- Most responses filled 4096 tokens without reaching an answer: with thinking off, the 2B model cannot solve competition problems.
- After training, likely-looping responses rose from 2 (6.7%) to 9 (30%): on long problems the model repeats the same reasoning until the limit. GSM8K responses are short and almost unaffected (0.2%).

### What this run shows

1. **Models exploit the reward design**: the fastest thing to learn was the grader's format, not better reasoning. This is the same reason the R1 paper designs format and accuracy rewards separately.
2. **Always report both strict and lenient scores**; otherwise a format fix looks like a capability gain.
3. GSM8K is too easy for this model (76% lenient accuracy before training), and a 1024-token limit is short, so R1-style longer responses and reflection do not emerge.

### Next steps (from run 1)

- Fix the evaluation protocol: report strict and lenient accuracy together, or make the prompt explicit enough that the baseline does not lose points on format.
- Sample AIME several times per problem (e.g. 16) and average to reduce variance.
- To observe R1-like behavior: harder training data (e.g. MATH or DAPO-Math), longer response limits, and thinking mode.
- Raise the KL coefficient, or stop early once reward plateaus, to avoid drifting too far from the base model.

## Thinking-mode baseline (eval-*-baseline-think)

Setup: thinking mode on, sampling with the Qwen3.5 model card's recommended parameters (temperature 1.0, top_p 0.95, top_k 20; presence_penalty is not supported by verl's validation config). GSM8K: 1 sample per problem; AIME: 8 samples per problem; response limit 8192 tokens for both (the model card recommends 32768). Logs and responses are in [`think/`](think/); the GSM8K response file (~23 MB) is kept locally only. The first two attempts accidentally used temperature 0.6; they were stopped midway and discarded.

### GSM8K: nearly every problem the model finishes thinking about is right; points are lost to truncation and format

| | Problems | Share |
|---|---|---|
| Finished thinking within 8192 tokens (`</think>` present) | 939 | 71.2% |
| Finished thinking and the answer is right (number only) | 895 | 95.3% of finished |
| ...of which the format was rejected (`####17`, `#### $18`, newline after `####`, ...) | 312 | |
| Still thinking at 8192 tokens | 380 | 28.8% |
| **Strict accuracy (verl grader)** | 589 | **44.7%** |
| **Lenient accuracy** (answer part if finished, otherwise the last number) | 946 | **71.7%** |

Estimated length needed to finish thinking (using ~3.3 characters per token, measured on truncated responses):

| Response limit (tokens) | 2048 | 4096 | 6144 | 8192 |
|---|---|---|---|---|
| Problems that finish thinking within the limit (share of all) | 15% | 45% | 61% | 70% |

- **When thinking finishes, accuracy is about 95%**, far above the 75.8% lenient accuracy with thinking off. GSM8K is not hard for this model; what it lacks is **finishing its reasoning within a limited budget**.
- Unfinished responses contain "wait" 34 times on average (19 for finished ones), mostly re-checking the problem statement after already reaching the answer. They are rarely pure repetition loops (only 8 flagged by the compression check).
- So RL on GSM8K can mainly improve format and reasoning length, not problem-solving ability. With a 4096-token training limit only 45% of problems can finish, so RL would most likely learn to shorten its thinking first.

### AIME: not a single response finished thinking within 8192 tokens

- All 240 responses (30 problems × 8 samples) hit the 8192 limit without `</think>`, so all score 0.
- 8192 tokens is far too short for competition math in thinking mode (the model card recommends 81920 for hard problems). Under the current budget, AIME cannot measure the effect of training.

## Appendix

### L40S runtime reference

- **Smoke-test timing (64 responses per step):** step 1 took 106 s including warm-up. After that, each step took ~25 s: ~12 s generation, ~3 s old-policy log-probs, ~6 s parameter update. Saving a checkpoint took ~21 s.
- **Memory:** training peaked at ~17.5 GB; vLLM reserves another ~30% of GPU memory.
- **GSM8K evaluation:** ~11 minutes end to end, including 3–4 minutes of startup.
- **Sample responses:** with only the console logger, `log_val_generations` prints no samples. `scripts/common.sh` sets `trainer.validation_data_dir`, so every validation response is written to `/workspace/logs/val_generations/<experiment>/`.
- **SIGSEGV at the end of evaluation:** the reward worker may report SIGSEGV when AIME evaluation finishes. It happens after the metrics are printed and does not affect results.

### Troubleshooting log

Problems encountered on other GPUs before settling on the L40S:

| Symptom | Cause | Fix |
|---|---|---|
| RTX 4090: vLLM reports `No available memory for the cache blocks` | With 24 GB, the default `ROLLOUT_GPU_MEM_UTIL=0.3` gives vLLM only ~7 GB | Raising it and enabling offload led to the next row's out-of-memory error |
| RTX 4090: Ray `OutOfMemoryError` (host RAM) | The Pod's RAM limit is ~38 GB; offloading parameters to RAM exceeded it | Switched to a 48 GB GPU |
| Re-running after a failure runs out of RAM or GPU memory | Processes from the failed run did not exit and held ~10 GB of RAM | Run `scripts/cleanup.sh` before re-running |
| 48 GB GPU: `CUDA out of memory` while computing entropy | Qwen3.5's vocabulary is very large; with `use_remove_padding=False` entropy is not chunked and the full logits tensor is materialized | Set `use_remove_padding=True`, enable chunked entropy, and cap tokens per GPU at 6144 (in `scripts/common.sh`) |
| A40 (7 CPU cores): logs stop 5 minutes into the smoke test, GPU stays idle, `ray status` shows CPU 7/7, GPU 0/1 | verl starts 8 TransferQueue storage actors, 8 reward workers, and 8 rollout agents by default; Ray reserves 1 CPU for each, and actors queue forever without an error when CPUs run out | In `scripts/common.sh`, set all three counts to 2 and `ray_kwargs.ray_init.num_cpus=32` |
