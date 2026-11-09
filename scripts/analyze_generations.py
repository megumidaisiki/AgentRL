"""分析 verl 验证时保存的回答（val_generations/*.jsonl）和训练日志。

在本地运行，只用标准库和 matplotlib：
    python scripts/analyze_generations.py results/run1

输出：
    - 终端打印 Markdown 表格，可以直接贴进 results/README.md
    - <目录>/training_curves.png：训练曲线
"""

import json
import re
import statistics
import sys
from pathlib import Path

# verl 的 GSM8K 严格规则：只看最后 300 个字符，必须是 "#### " 加数字（井号后要有空格）。
STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
# 宽松规则：井号后可以没有空格，也可以有 ** 或 $ 之类的修饰。
LOOSE = re.compile(r"####\s*\**\s*\$?\s*(\-?[0-9][0-9\.\,]*)")
NUMBER = re.compile(r"\-?[0-9][0-9\.\,]*")
REFLECT = re.compile(r"\b(wait|let me (re)?check|double[- ]check|verify|re-?examine|mistake)\b", re.I)


def norm(x):
    x = x.replace(",", "").replace("$", "").rstrip(".")
    try:
        return str(int(float(x))) if float(x) == int(float(x)) else x
    except ValueError:
        return x


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def looks_truncated(text):
    """没有 token 数，用结尾判断：写满上限的回答通常停在半句话中间。"""
    tail = text.rstrip()[-1:] if text.strip() else ""
    return tail not in ".!?)}]*$`" and not tail.isdigit()


def classify_gsm8k(row):
    tail = row["output"][-300:]
    gt = norm(row["gts"])
    if row["score"] > 0:
        return "答对"
    strict = STRICT.findall(tail)
    if strict:
        return "格式对，答案错"
    loose = LOOSE.findall(row["output"])
    if loose:
        return "答案对，格式不合规则" if norm(loose[-1]) == gt else "格式不合规则，答案也错"
    nums = NUMBER.findall(tail)
    if looks_truncated(row["output"]):
        return "没写 ####，疑似被截断"
    if nums and norm(nums[-1]) == gt:
        return "没写 ####，最后的数字是对的"
    return "没写 ####，答案错"


GSM_ORDER = [
    "答对",
    "答案对，格式不合规则",
    "没写 ####，最后的数字是对的",
    "格式对，答案错",
    "格式不合规则，答案也错",
    "没写 ####，答案错",
    "没写 ####，疑似被截断",
]


def gsm8k_summary(rows):
    cats = [classify_gsm8k(r) for r in rows]
    n = len(rows)
    counts = {c: cats.count(c) for c in GSM_ORDER}
    lens = [len(r["output"]) for r in rows]
    reflect = sum(bool(REFLECT.search(r["output"])) for r in rows)
    lenient = counts["答对"] + counts["答案对，格式不合规则"] + counts["没写 ####，最后的数字是对的"]
    return {
        "n": n,
        "counts": counts,
        "strict_acc": counts["答对"] / n,
        "lenient_acc": lenient / n,
        "chars_mean": statistics.mean(lens),
        "chars_median": statistics.median(lens),
        "reflect": reflect / n,
        "cats": cats,
    }


def aime_summary(rows):
    n = len(rows)
    correct = [i for i, r in enumerate(rows) if r["acc"] in (True, 1, 1.0)]
    invalid = sum(r.get("pred") == "[INVALID]" for r in rows)
    lens = [len(r["output"]) for r in rows]
    return {
        "n": n,
        "correct": len(correct),
        "invalid": invalid,
        "chars_mean": statistics.mean(lens),
        "reflect": sum(bool(REFLECT.search(r["output"])) for r in rows) / n,
    }


def by_question(rows):
    return {r["input"]: r for r in rows}


def flips(before, after, ok):
    b, a = by_question(before), by_question(after)
    common = b.keys() & a.keys()
    gained = sum(1 for q in common if not ok(b[q]) and ok(a[q]))
    lost = sum(1 for q in common if ok(b[q]) and not ok(a[q]))
    return len(common), gained, lost


def parse_train_log(path):
    steps = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.search(r"step:(\d+) - (.*)", line)
        if not m:
            continue
        rec = steps.setdefault(int(m.group(1)), {})
        for kv in m.group(2).split(" - "):
            if ":" not in kv:
                continue
            k, v = kv.split(":", 1)
            v = re.sub(r"np\.\w+\((.*)\)", r"\1", v.strip())
            try:
                rec[k.strip()] = float(v)
            except ValueError:
                pass
    return steps


def plot(steps, val_stats, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = sorted(s for s in steps if "critic/score/mean" in steps[s])
    panels = [
        ("训练奖励（采样回答的平均分）", "critic/score/mean"),
        ("训练回答平均长度（token）", "response_length/mean"),
        ("训练回答截断比例", "response_length/clip_ratio"),
        ("策略熵", "actor/entropy"),
        ("和参考模型的 KL", "actor/kl_loss"),
        ("每步耗时（秒）", "timing_s/step"),
    ]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (title, key) in zip(axes.flat, panels):
        ys = [(s, steps[s][key]) for s in xs if key in steps[s]]
        if ys:
            ax.plot(*zip(*ys), lw=1, color="#3b6ea5")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    vs = sorted(val_stats)
    ax = axes.flat[6]
    ax.plot(vs, [val_stats[s]["strict_acc"] for s in vs], "o-", label="严格评分（verl）")
    ax.plot(vs, [val_stats[s]["lenient_acc"] for s in vs], "s--", label="宽松评分（只看答案）")
    ax.set_title("GSM8K 测试集准确率")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes.flat[7]
    ax.plot(vs, [val_stats[s]["chars_mean"] for s in vs], "o-")
    ax.set_title("测试集回答平均长度（字符）")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    ax = axes.flat[8]
    ax.plot(vs, [val_stats[s]["reflect"] for s in vs], "o-")
    ax.set_title("含反思用语的回答比例（wait/verify 等）")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def pct(x):
    return f"{100 * x:.1f}%"


def main(root):
    root = Path(root)
    vg = root / "val_generations"

    print("## GSM8K 训练前后（1319 题）\n")
    base = load(vg / "eval-gsm8k-baseline" / "0.jsonl")
    trained = load(next((vg / "eval-gsm8k-trained").glob("*.jsonl")))
    sb, st = gsm8k_summary(base), gsm8k_summary(trained)
    print("| 类别 | 训练前 | 训练后 |\n|---|---|---|")
    for c in GSM_ORDER:
        print(f"| {c} | {sb['counts'][c]} | {st['counts'][c]} |")
    print(f"| **严格准确率（verl 评分）** | **{pct(sb['strict_acc'])}** | **{pct(st['strict_acc'])}** |")
    print(f"| **宽松准确率（只看答案）** | **{pct(sb['lenient_acc'])}** | **{pct(st['lenient_acc'])}** |")
    print(f"| 回答长度（字符，平均 / 中位数） | {sb['chars_mean']:.0f} / {sb['chars_median']:.0f} | {st['chars_mean']:.0f} / {st['chars_median']:.0f} |")
    print(f"| 含反思用语的回答 | {pct(sb['reflect'])} | {pct(st['reflect'])} |")
    n, g, l = flips(base, trained, lambda r: r["score"] > 0)
    print(f"\n逐题对比（严格评分）：{n} 题中 {g} 题从错变对，{l} 题从对变错。")
    lenient_ok = lambda r: classify_gsm8k(r) in GSM_ORDER[:3]
    n, g, l = flips(base, trained, lenient_ok)
    print(f"逐题对比（宽松评分）：{g} 题从错变对，{l} 题从对变错。\n")

    print("## AIME 2026 训练前后（30 题）\n")
    ab = load(vg / "eval-aime26-baseline" / "0.jsonl")
    at = load(next((vg / "eval-aime26-trained").glob("*.jsonl")))
    xb, xt = aime_summary(ab), aime_summary(at)
    print("| | 训练前 | 训练后 |\n|---|---|---|")
    print(f"| 答对 | {xb['correct']}/30 | {xt['correct']}/30 |")
    print(f"| 没写出 \\boxed{{}}（多半写满 4096 被截断） | {xb['invalid']} | {xt['invalid']} |")
    print(f"| 回答平均长度（字符） | {xb['chars_mean']:.0f} | {xt['chars_mean']:.0f} |")
    print(f"| 含反思用语的回答 | {pct(xb['reflect'])} | {pct(xt['reflect'])} |")
    ok = lambda r: r["acc"] in (True, 1, 1.0)
    n, g, l = flips(ab, at, ok)
    print(f"\n逐题对比：{g} 题从错变对，{l} 题从对变错。\n")

    train_dir = next((d for d in vg.iterdir() if d.name.startswith("gsm8k-")), None)
    logs = sorted(root.glob("gsm8k-*.log"))
    if train_dir and logs:
        val_stats = {int(p.stem): gsm8k_summary(load(p)) for p in train_dir.glob("*.jsonl")}
        print("## 训练中的验证（GSM8K 测试集）\n")
        print("| step | 严格准确率 | 宽松准确率 | 格式不合规则但答案对 | 疑似截断 | 平均长度（字符） | 反思用语 |")
        print("|---|---|---|---|---|---|---|")
        for s in sorted(val_stats):
            v = val_stats[s]
            slip = v["counts"]["答案对，格式不合规则"] + v["counts"]["没写 ####，最后的数字是对的"]
            print(f"| {s} | {pct(v['strict_acc'])} | {pct(v['lenient_acc'])} | {slip} | "
                  f"{v['counts']['没写 ####，疑似被截断']} | {v['chars_mean']:.0f} | {pct(v['reflect'])} |")
        steps = parse_train_log(logs[-1])
        out = root / "training_curves.png"
        plot(steps, val_stats, out)
        print(f"\n训练曲线已保存到 {out}")
        keys = ["critic/score/mean", "response_length/mean", "response_length/clip_ratio",
                "actor/entropy", "actor/kl_loss", "actor/pg_clipfrac", "timing_s/step"]
        print("\n训练指标（前 5 步平均 → 最后 5 步平均）：")
        xs = sorted(s for s in steps if "critic/score/mean" in steps[s])
        for k in keys:
            vals = [steps[s][k] for s in xs if k in steps[s]]
            if len(vals) >= 10:
                print(f"- {k}: {statistics.mean(vals[:5]):.4g} → {statistics.mean(vals[-5:]):.4g}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/run1")
