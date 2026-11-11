# Project notes (for AI assistants)

LoRA + GRPO training of Qwen3.5-2B with verl on a single L40S, comparing evaluation scores before and after training, and attempting to reproduce what the DeepSeek-R1 paper describes.

## Conventions

- The local machine is Windows and holds only code and small results. No training runs locally.
- Training runs on RunPod. Persistent data lives in `/workspace`; the container disk is wiped when a Pod stops.
- All paths and default parameters live in `scripts/common.sh`. Do not repeat them in other scripts.
- verl is pinned to `VERL_COMMIT` in `scripts/common.sh` and is not committed to this repo. When upgrading, record the reason and re-run the smoke test.
- Models, data, checkpoints, and secrets never go into Git. `results/` holds only logs, scores, and sample outputs.
- Do not claim a script works unless `results/` or the experiment log contains a matching log.
- The repository is public and written in English.

## Experiment order

`smoke_test.sh` → `eval.sh` (baseline) → `train.sh` → `eval.sh` (with adapter)

Get the math tasks working first. LiveCodeBench and similar tasks come after the math pipeline is stable.
