"""Per-step GRPO group statistics, using only the Python standard library.

GRPO samples n responses (a group) per prompt and uses the group mean as the baseline for advantages:
    - If a group is all correct or all wrong, every response in it has zero advantage, so that prompt
      contributes nothing to the update.
    - A rising all-correct share means the prompts are too easy for the model; a high all-wrong share
      means they are too hard.
    - Only mixed groups (some right, some wrong) provide a learning signal.

During training, train.sh writes each step's rollouts to rollout_generations/<experiment>/<step>.jsonl.

Usage (on the Pod while training, or locally after copying the logs back):
    python3 scripts/group_stats.py /workspace/logs/rollout_generations/<experiment>
    python3 scripts/group_stats.py <dir> --every 10     # one row every 10 steps
    python3 scripts/group_stats.py <dir> --last 5       # only the last 5 steps
"""

import argparse
import json
import statistics
import zlib
from collections import defaultdict
from pathlib import Path

THINK_END = "</think>"


def split_thinking(text):
    """Return (thinking, answer, finished). Without a </think> tag the whole text counts as unfinished thinking."""
    if THINK_END in text:
        think, answer = text.rsplit(THINK_END, 1)
        return think, answer, True
    return text, "", False


def looks_looping(text, window=2000, threshold=0.12):
    """A highly repetitive tail (very low compression ratio) usually means the model looped until the length limit."""
    if len(text) < window:
        return False
    tail = text[-window:].encode("utf-8")
    return len(zlib.compress(tail)) / len(tail) < threshold


def step_stats(rows, thinking=None):
    groups = defaultdict(list)
    for r in rows:
        uid = str(r.get("uid", ""))
        parts = uid.rsplit("_", 2)
        groups[parts[0] if len(parts) == 3 else r["input"]].append(r)

    n_groups = len(groups)
    all0 = all1 = 0
    stds = []
    for g in groups.values():
        scores = [1.0 if r["score"] > 0 else 0.0 for r in g]
        if all(s == 0 for s in scores):
            all0 += 1
        elif all(s == 1 for s in scores):
            all1 += 1
        stds.append(statistics.pstdev([r["score"] for r in g]))

    correct = [r for r in rows if r["score"] > 0]
    wrong = [r for r in rows if r["score"] <= 0]
    if thinking is None:
        # In thinking mode the prompt ends with "<think>\n"; otherwise the template inserts an empty "<think></think>".
        thinking = any("<think>" in r["input"][-40:] and THINK_END not in r["input"][-40:] for r in rows)
    split = [split_thinking(r["output"]) for r in rows]

    def mean_len(rs):
        return statistics.mean(len(r["output"]) for r in rs) if rs else float("nan")

    stats = {
        "groups": n_groups,
        "group_size": statistics.mean(len(g) for g in groups.values()),
        "all0": all0 / n_groups,
        "all1": all1 / n_groups,
        "mixed": 1 - (all0 + all1) / n_groups,
        "mean_score": statistics.mean(1.0 if r["score"] > 0 else 0.0 for r in rows),
        "pass_at_n": 1 - all0 / n_groups,
        "group_std": statistics.mean(stds),
        "len_correct": mean_len(correct),
        "len_wrong": mean_len(wrong),
        "looping": sum(looks_looping(r["output"]) for r in rows) / len(rows),
    }
    if thinking:
        stats["think_len"] = statistics.mean(len(t) for t, _, _ in split)
        stats["answer_len"] = statistics.mean(len(a) for _, a, done in split if done) if any(d for *_, d in split) else float("nan")
        stats["think_unfinished"] = sum(not done for *_, done in split) / len(rows)
    return stats


def load_dir(path):
    out = {}
    for p in Path(path).glob("*.jsonl"):
        if p.stem.isdigit():
            with open(p, encoding="utf-8") as f:
                out[int(p.stem)] = [json.loads(line) for line in f if line.strip()]
    return dict(sorted(out.items()))


def all_stats(path):
    return {step: step_stats(rows) for step, rows in load_dir(path).items()}


def pct(x):
    return "—" if x != x else f"{100 * x:.0f}%"


def num(x):
    return "—" if x != x else f"{x:.0f}"


def print_table(stats, every=1, last=None):
    steps = list(stats)
    if last:
        steps = steps[-last:]
    else:
        steps = [s for i, s in enumerate(steps) if i % every == 0 or s == steps[-1]]
    thinking = any("think_len" in stats[s] for s in steps)
    head = ["step", "all wrong", "all correct", "mixed", "mean acc", "pass@n", "len (correct)", "len (wrong)", "looping"]
    if thinking:
        head += ["think len", "answer len", "unfinished thinking"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for s in steps:
        v = stats[s]
        row = [str(s), pct(v["all0"]), pct(v["all1"]), pct(v["mixed"]), pct(v["mean_score"]),
               pct(v["pass_at_n"]), num(v["len_correct"]), num(v["len_wrong"]), pct(v["looping"])]
        if thinking:
            row += [num(v["think_len"]), num(v["answer_len"]), pct(v["think_unfinished"])]
        print("| " + " | ".join(row) + " |")
    print("\nLengths are in characters. All-wrong and all-correct groups have zero advantage and give no learning signal.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="rollout_generations/<experiment> directory")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--last", type=int, default=None)
    args = ap.parse_args()
    stats = all_stats(args.dir)
    if not stats:
        print(f"No <step>.jsonl files in {args.dir} yet")
        return
    print_table(stats, args.every, args.last)


if __name__ == "__main__":
    main()
