"""统计 GRPO 训练中每一步的组内情况，只用 Python 标准库。

GRPO 对每道题生成 n 个回答（一组），用组内平均分做基准算优势：
    - 一组全对或全错时，组内每个回答的优势都是 0，这道题对本次更新没有贡献。
    - 全对比例越来越高，说明题目对模型太简单；全错比例高，说明太难。
    - 只有"有对有错"的组才在提供学习信号。

训练时由 train.sh 把每一步的回答写到 rollout_generations/<实验名>/<步数>.jsonl。

用法（在 Pod 上边训练边看，或者传回本地后看）：
    python3 scripts/group_stats.py /workspace/logs/rollout_generations/<实验名>
    python3 scripts/group_stats.py <目录> --every 10     # 每 10 步显示一行
    python3 scripts/group_stats.py <目录> --last 5       # 只看最近 5 步
"""

import argparse
import json
import statistics
import zlib
from collections import defaultdict
from pathlib import Path

THINK_END = "</think>"


def split_thinking(text):
    """返回 (思考部分, 答案部分, 是否写完了思考)。没打开思考模式时思考部分为空。"""
    if THINK_END in text:
        think, answer = text.rsplit(THINK_END, 1)
        return think, answer, True
    return text, "", False


def looks_looping(text, window=2000, threshold=0.12):
    """回答末尾高度重复（压缩率很低）时，多半是陷入了循环，一直写到长度上限。"""
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
        # 思考模式下提示词以 "<think>\n" 结尾；关闭时模板会放一个空的 "<think></think>"。
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
    head = ["step", "全错组", "全对组", "有对有错", "平均正确率", "pass@n", "对的长度", "错的长度", "疑似循环"]
    if thinking:
        head += ["思考长度", "答案长度", "思考没写完"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for s in steps:
        v = stats[s]
        row = [str(s), pct(v["all0"]), pct(v["all1"]), pct(v["mixed"]), pct(v["mean_score"]),
               pct(v["pass_at_n"]), num(v["len_correct"]), num(v["len_wrong"]), pct(v["looping"])]
        if thinking:
            row += [num(v["think_len"]), num(v["answer_len"]), pct(v["think_unfinished"])]
        print("| " + " | ".join(row) + " |")
    print("\n长度单位是字符。全错组和全对组的优势都是 0，不提供学习信号。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="rollout_generations/<实验名> 目录")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--last", type=int, default=None)
    args = ap.parse_args()
    stats = all_stats(args.dir)
    if not stats:
        print(f"{args.dir} 里还没有 <步数>.jsonl 文件")
        return
    print_table(stats, args.every, args.last)


if __name__ == "__main__":
    main()
