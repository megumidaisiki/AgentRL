"""分析一次实验的结果：评测回答、训练中的验证回答、训练回答和训练日志。

在本地运行，只用标准库和 matplotlib：
    python scripts/analyze_generations.py results/run1            # 关闭思考模式的实验
    python scripts/analyze_generations.py results/run2 --think    # 打开思考模式的实验

目录结构就是 Pod 上 /workspace/logs 的结构（val_generations/、rollout_generations/、*.log）。

输出：
    - 终端打印 Markdown 表格，可以直接贴进 results/README.md
    - <目录>/training_curves.png：训练曲线
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from group_stats import all_stats as rollout_group_stats  # noqa: E402
from group_stats import looks_looping, print_table, split_thinking  # noqa: E402

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
        return [json.loads(line) for line in f if line.strip()]


def is_thinking(row):
    tail = row["input"][-40:]
    return "<think>" in tail and "</think>" not in tail


def looks_truncated(text):
    """没有 token 数，用结尾判断：写满上限的回答通常停在半句话中间。"""
    tail = text.rstrip()[-1:] if text.strip() else ""
    return tail not in ".!?)}]*$`" and not tail.isdigit()


def classify_gsm8k(row):
    output = row["output"]
    if is_thinking(row):
        _, answer, done = split_thinking(output)
        if not done:
            return "思考没写完（被截断）" if row["score"] <= 0 else "答对"
    else:
        answer = output
    gt = norm(row["gts"])
    if row["score"] > 0:
        return "答对"
    if STRICT.findall(output[-300:]):
        return "格式对，答案错"
    loose = LOOSE.findall(answer)
    if loose:
        return "答案对，格式不合规则" if norm(loose[-1]) == gt else "格式不合规则，答案也错"
    nums = NUMBER.findall(answer[-300:])
    if looks_truncated(answer):
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
    "思考没写完（被截断）",
]
LENIENT_OK = set(GSM_ORDER[:3])


def gsm8k_summary(rows):
    cats = [classify_gsm8k(r) for r in rows]
    n = len(rows)
    counts = {c: cats.count(c) for c in GSM_ORDER}
    lens = [len(r["output"]) for r in rows]
    s = {
        "n": n,
        "counts": counts,
        "strict_acc": counts["答对"] / n,
        "lenient_acc": sum(counts[c] for c in LENIENT_OK) / n,
        "chars_mean": statistics.mean(lens),
        "chars_median": statistics.median(lens),
        "reflect": sum(bool(REFLECT.search(r["output"])) for r in rows) / n,
        "looping": sum(looks_looping(r["output"]) for r in rows) / n,
        "thinking": any(is_thinking(r) for r in rows),
    }
    if s["thinking"]:
        parts = [split_thinking(r["output"]) for r in rows]
        s["think_len"] = statistics.mean(len(t) for t, _, _ in parts)
        done = [a for _, a, d in parts if d]
        s["answer_len"] = statistics.mean(len(a) for a in done) if done else float("nan")
    return s


def by_question(rows):
    """同一道题可能有多次采样，按题目分组。"""
    out = {}
    for r in rows:
        out.setdefault(r["input"], []).append(r)
    return out


def aime_ok(r):
    return r["acc"] in (True, 1, 1.0)


def aime_summary(rows):
    qs = by_question(rows)
    k = round(len(rows) / len(qs))
    return {
        "questions": len(qs),
        "k": k,
        "mean_at_k": statistics.mean(aime_ok(r) for r in rows),
        "pass_at_k": statistics.mean(any(aime_ok(r) for r in g) for g in qs.values()),
        "invalid": statistics.mean(r.get("pred") == "[INVALID]" for r in rows),
        "chars_mean": statistics.mean(len(r["output"]) for r in rows),
        "reflect": statistics.mean(bool(REFLECT.search(r["output"])) for r in rows),
        "looping": statistics.mean(looks_looping(r["output"]) for r in rows),
    }


def flips(before, after, ok):
    b, a = by_question(before), by_question(after)
    common = b.keys() & a.keys()
    score = lambda g: statistics.mean(ok(r) for r in g)  # noqa: E731
    gained = sum(1 for q in common if score(b[q]) < 0.5 <= score(a[q]))
    lost = sum(1 for q in common if score(b[q]) >= 0.5 > score(a[q]))
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


def plot(steps, val_stats, groups, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    xs = sorted(s for s in steps if "critic/score/mean" in steps[s])
    log_panels = [
        ("训练奖励（采样回答的平均分）", "critic/score/mean"),
        ("训练回答平均长度（token）", "response_length/mean"),
        ("训练回答截断比例", "response_length/clip_ratio"),
        ("策略熵", "actor/entropy"),
        ("和参考模型的 KL", "actor/kl_loss"),
        ("PPO 截断比例（pg_clipfrac）", "actor/pg_clipfrac"),
        ("梯度范数", "actor/grad_norm"),
        ("每步耗时（秒）", "timing_s/step"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    flat = list(axes.flat)
    for ax, (title, key) in zip(flat, log_panels):
        ys = [(s, steps[s][key]) for s in xs if key in steps[s]]
        if ys:
            ax.plot(*zip(*ys), lw=1, color="#3b6ea5")
        ax.set_title(title)

    ax = flat[8]
    if groups:
        gs = sorted(groups)
        ax.stackplot(gs, [groups[s]["all0"] for s in gs], [groups[s]["mixed"] for s in gs],
                     [groups[s]["all1"] for s in gs], labels=["全错组", "有对有错", "全对组"],
                     colors=["#c0504d", "#9bbb59", "#4f81bd"], alpha=0.85)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "没有训练回答记录", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("GRPO 组内情况（每组 n 个回答）")

    vs = sorted(val_stats)
    ax = flat[9]
    ax.plot(vs, [val_stats[s]["strict_acc"] for s in vs], "o-", label="严格评分（verl）")
    ax.plot(vs, [val_stats[s]["lenient_acc"] for s in vs], "s--", label="宽松评分（只看答案）")
    ax.set_title("GSM8K 测试集准确率")
    ax.legend()

    ax = flat[10]
    if groups and "think_len" in next(iter(groups.values())):
        gs = sorted(groups)
        ax.plot(gs, [groups[s]["think_len"] for s in gs], label="思考部分")
        ax.plot(gs, [groups[s]["answer_len"] for s in gs], label="答案部分")
        ax.set_title("训练回答长度（字符）")
        ax.legend()
    else:
        ax.plot(vs, [val_stats[s]["chars_mean"] for s in vs], "o-")
        ax.set_title("测试集回答平均长度（字符）")

    ax = flat[11]
    ax.plot(vs, [val_stats[s]["reflect"] for s in vs], "o-", label="含反思用语")
    ax.plot(vs, [val_stats[s]["looping"] for s in vs], "s--", label="疑似重复循环")
    ax.set_title("测试集回答的行为")
    ax.legend()

    for ax in flat:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=100)


def pct(x):
    return "—" if x != x else f"{100 * x:.1f}%"


def find_one(base, pattern):
    hits = sorted(base.glob(pattern))
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/run1")
    ap.add_argument("--think", action="store_true", help="分析打开思考模式的实验（实验名带 -think）")
    args = ap.parse_args()
    root = Path(args.root)
    vg = root / "val_generations"
    sfx = "-think" if args.think else ""

    base_f = find_one(vg, f"eval-gsm8k-baseline{sfx}/*.jsonl")
    trained_f = find_one(vg, f"eval-gsm8k-trained{sfx}/*.jsonl")
    if base_f and trained_f:
        print("## GSM8K 训练前后（1319 题）\n")
        base, trained = load(base_f), load(trained_f)
        sb, st = gsm8k_summary(base), gsm8k_summary(trained)
        print("| 类别 | 训练前 | 训练后 |\n|---|---|---|")
        for c in GSM_ORDER:
            if sb["counts"][c] or st["counts"][c]:
                print(f"| {c} | {sb['counts'][c]} | {st['counts'][c]} |")
        print(f"| **严格准确率（verl 评分）** | **{pct(sb['strict_acc'])}** | **{pct(st['strict_acc'])}** |")
        print(f"| **宽松准确率（只看答案）** | **{pct(sb['lenient_acc'])}** | **{pct(st['lenient_acc'])}** |")
        print(f"| 回答长度（字符，平均 / 中位数） | {sb['chars_mean']:.0f} / {sb['chars_median']:.0f} | "
              f"{st['chars_mean']:.0f} / {st['chars_median']:.0f} |")
        if sb["thinking"]:
            print(f"| 思考部分 / 答案部分长度（字符） | {sb['think_len']:.0f} / {sb['answer_len']:.0f} | "
                  f"{st['think_len']:.0f} / {st['answer_len']:.0f} |")
        print(f"| 含反思用语的回答 | {pct(sb['reflect'])} | {pct(st['reflect'])} |")
        print(f"| 疑似重复循环 | {pct(sb['looping'])} | {pct(st['looping'])} |")
        n, g, lost = flips(base, trained, lambda r: r["score"] > 0)
        print(f"\n逐题对比（严格评分）：{n} 题中 {g} 题从错变对，{lost} 题从对变错。")
        n, g, lost = flips(base, trained, lambda r: classify_gsm8k(r) in LENIENT_OK)
        print(f"逐题对比（宽松评分）：{g} 题从错变对，{lost} 题从对变错。\n")

    ab_f = find_one(vg, f"eval-aime26-baseline{sfx}/*.jsonl")
    at_f = find_one(vg, f"eval-aime26-trained{sfx}/*.jsonl")
    if ab_f and at_f:
        ab, at = load(ab_f), load(at_f)
        xb, xt = aime_summary(ab), aime_summary(at)
        print(f"## AIME 2026 训练前后（30 题，每题 {xb['k']} 次）\n")
        print("| | 训练前 | 训练后 |\n|---|---|---|")
        print(f"| 平均正确率（mean@{xb['k']}） | {pct(xb['mean_at_k'])} | {pct(xt['mean_at_k'])} |")
        if xb["k"] > 1:
            print(f"| 至少答对一次的题（pass@{xb['k']}） | {pct(xb['pass_at_k'])} | {pct(xt['pass_at_k'])} |")
        print(f"| 没写出 \\boxed{{}}（多半写满上限被截断） | {pct(xb['invalid'])} | {pct(xt['invalid'])} |")
        print(f"| 回答平均长度（字符） | {xb['chars_mean']:.0f} | {xt['chars_mean']:.0f} |")
        print(f"| 含反思用语的回答 | {pct(xb['reflect'])} | {pct(xt['reflect'])} |")
        print(f"| 疑似重复循环 | {pct(xb['looping'])} | {pct(xt['looping'])} |")
        n, g, lost = flips(ab, at, aime_ok)
        print(f"\n逐题对比（多数次答对算对）：{g} 题从错变对，{lost} 题从对变错。\n")

    train_dirs = [d for d in sorted(vg.iterdir()) if d.name.startswith("gsm8k-") and (("-think" in d.name) == args.think)]
    if not train_dirs:
        return
    name = train_dirs[-1].name
    log = find_one(root, f"{name}-*.log")
    val_stats = {int(p.stem): gsm8k_summary(load(p)) for p in train_dirs[-1].glob("*.jsonl")}
    rollout_dir = root / "rollout_generations" / name
    groups = rollout_group_stats(rollout_dir) if rollout_dir.exists() else {}

    print(f"## 训练中的验证（{name}，GSM8K 测试集）\n")
    head = "| step | 严格准确率 | 宽松准确率 | 格式不合规则但答案对 | 疑似截断 | 平均长度（字符） | 反思用语 |"
    print(head + "\n|---|---|---|---|---|---|---|")
    for s in sorted(val_stats):
        v = val_stats[s]
        slip = v["counts"]["答案对，格式不合规则"] + v["counts"]["没写 ####，最后的数字是对的"]
        cut = v["counts"]["没写 ####，疑似被截断"] + v["counts"]["思考没写完（被截断）"]
        print(f"| {s} | {pct(v['strict_acc'])} | {pct(v['lenient_acc'])} | {slip} | {cut} | "
              f"{v['chars_mean']:.0f} | {pct(v['reflect'])} |")

    if groups:
        print("\n## GRPO 组内情况（训练回答）\n")
        print_table(groups, every=max(1, len(groups) // 12))

    if log:
        steps = parse_train_log(log)
        out = root / "training_curves.png"
        plot(steps, val_stats, groups, out)
        print(f"\n训练曲线已保存到 {out}")
        keys = ["critic/score/mean", "response_length/mean", "response_length/clip_ratio", "actor/entropy",
                "actor/kl_loss", "actor/pg_clipfrac", "actor/grad_norm", "timing_s/step"]
        print("\n训练指标（前 5 步平均 → 最后 5 步平均）：")
        xs = sorted(s for s in steps if "critic/score/mean" in steps[s])
        for k in keys:
            vals = [steps[s][k] for s in xs if k in steps[s]]
            if len(vals) >= 10:
                print(f"- {k}: {statistics.mean(vals[:5]):.4g} → {statistics.mean(vals[-5:]):.4g}")


if __name__ == "__main__":
    main()
