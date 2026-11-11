"""Analyze one experiment: evaluation responses, in-training validation responses, rollouts, and the training log.

Runs locally with only the standard library and matplotlib:
    python scripts/analyze_generations.py results/run1            # non-thinking experiment
    python scripts/analyze_generations.py results/run2 --think    # thinking-mode experiment

The directory mirrors /workspace/logs on the Pod (val_generations/, rollout_generations/, *.log).

Output:
    - Markdown tables printed to the terminal, ready to paste into results/README.md
    - <dir>/training_curves.png: training curves
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

# verl's strict GSM8K rule: only the last 300 characters count, and the answer must be "#### " plus a number
# (with a space after the hashes).
STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
# Lenient rule: allows no space after the hashes and decorations such as ** or $.
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
    """No token counts are logged, so judge by the ending: responses that hit the limit usually stop mid-sentence."""
    tail = text.rstrip()[-1:] if text.strip() else ""
    return tail not in ".!?)}]*$`" and not tail.isdigit()


def classify_gsm8k(row):
    output = row["output"]
    if is_thinking(row):
        _, answer, done = split_thinking(output)
        if not done:
            return UNFINISHED if row["score"] <= 0 else CORRECT
    else:
        answer = output
    gt = norm(row["gts"])
    if row["score"] > 0:
        return CORRECT
    if STRICT.findall(output[-300:]):
        return FORMAT_OK_WRONG
    loose = LOOSE.findall(answer)
    if loose:
        return RIGHT_BAD_FORMAT if norm(loose[-1]) == gt else BAD_FORMAT_WRONG
    nums = NUMBER.findall(answer[-300:])
    if looks_truncated(answer):
        return NO_HASH_TRUNCATED
    if nums and norm(nums[-1]) == gt:
        return NO_HASH_RIGHT
    return NO_HASH_WRONG


CORRECT = "Correct (strict)"
RIGHT_BAD_FORMAT = "Right answer, format rejected by grader"
NO_HASH_RIGHT = "No ####, last number is right"
FORMAT_OK_WRONG = "Format OK, wrong answer"
BAD_FORMAT_WRONG = "Bad format, wrong answer"
NO_HASH_WRONG = "No ####, wrong answer"
NO_HASH_TRUNCATED = "No ####, likely truncated"
UNFINISHED = "Thinking unfinished (truncated)"
GSM_ORDER = [
    CORRECT,
    RIGHT_BAD_FORMAT,
    NO_HASH_RIGHT,
    FORMAT_OK_WRONG,
    BAD_FORMAT_WRONG,
    NO_HASH_WRONG,
    NO_HASH_TRUNCATED,
    UNFINISHED,
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
        "strict_acc": counts[CORRECT] / n,
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
    """Group samples by prompt (a prompt may be sampled several times)."""
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

    xs = sorted(s for s in steps if "critic/score/mean" in steps[s])
    log_panels = [
        ("Training reward (mean rollout score)", "critic/score/mean"),
        ("Mean rollout length (tokens)", "response_length/mean"),
        ("Rollouts truncated at the length limit", "response_length/clip_ratio"),
        ("Policy entropy", "actor/entropy"),
        ("KL to reference model", "actor/kl_loss"),
        ("PPO clip fraction (pg_clipfrac)", "actor/pg_clipfrac"),
        ("Gradient norm", "actor/grad_norm"),
        ("Time per step (s)", "timing_s/step"),
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
                     [groups[s]["all1"] for s in gs], labels=["all wrong", "mixed", "all correct"],
                     colors=["#c0504d", "#9bbb59", "#4f81bd"], alpha=0.85)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no rollout logs", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("GRPO groups (n responses per prompt)")

    vs = sorted(val_stats)
    ax = flat[9]
    ax.plot(vs, [val_stats[s]["strict_acc"] for s in vs], "o-", label="strict (verl grader)")
    ax.plot(vs, [val_stats[s]["lenient_acc"] for s in vs], "s--", label="lenient (answer only)")
    ax.set_title("GSM8K test accuracy")
    ax.legend()

    ax = flat[10]
    if groups and "think_len" in next(iter(groups.values())):
        gs = sorted(groups)
        ax.plot(gs, [groups[s]["think_len"] for s in gs], label="thinking")
        ax.plot(gs, [groups[s]["answer_len"] for s in gs], label="answer")
        ax.set_title("Rollout length (characters)")
        ax.legend()
    else:
        ax.plot(vs, [val_stats[s]["chars_mean"] for s in vs], "o-")
        ax.set_title("Mean test response length (characters)")

    ax = flat[11]
    ax.plot(vs, [val_stats[s]["reflect"] for s in vs], "o-", label="contains reflection words")
    ax.plot(vs, [val_stats[s]["looping"] for s in vs], "s--", label="likely looping")
    ax.set_title("Test response behavior")
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
    ap.add_argument("--think", action="store_true", help="analyze a thinking-mode experiment (run names end in -think)")
    args = ap.parse_args()
    root = Path(args.root)
    vg = root / "val_generations"
    sfx = "-think" if args.think else ""

    base_f = find_one(vg, f"eval-gsm8k-baseline{sfx}/*.jsonl")
    trained_f = find_one(vg, f"eval-gsm8k-trained{sfx}/*.jsonl")
    if base_f and trained_f:
        print("## GSM8K before vs. after training (1319 problems)\n")
        base, trained = load(base_f), load(trained_f)
        sb, st = gsm8k_summary(base), gsm8k_summary(trained)
        print("| Category | Before | After |\n|---|---|---|")
        for c in GSM_ORDER:
            if sb["counts"][c] or st["counts"][c]:
                print(f"| {c} | {sb['counts'][c]} | {st['counts'][c]} |")
        print(f"| **Strict accuracy (verl grader)** | **{pct(sb['strict_acc'])}** | **{pct(st['strict_acc'])}** |")
        print(f"| **Lenient accuracy (answer only)** | **{pct(sb['lenient_acc'])}** | **{pct(st['lenient_acc'])}** |")
        print(f"| Response length (chars, mean / median) | {sb['chars_mean']:.0f} / {sb['chars_median']:.0f} | "
              f"{st['chars_mean']:.0f} / {st['chars_median']:.0f} |")
        if sb["thinking"]:
            print(f"| Thinking / answer length (chars) | {sb['think_len']:.0f} / {sb['answer_len']:.0f} | "
                  f"{st['think_len']:.0f} / {st['answer_len']:.0f} |")
        print(f"| Responses with reflection words | {pct(sb['reflect'])} | {pct(st['reflect'])} |")
        print(f"| Likely looping | {pct(sb['looping'])} | {pct(st['looping'])} |")
        n, g, lost = flips(base, trained, lambda r: r["score"] > 0)
        print(f"\nPer-problem (strict): of {n} problems, {g} went wrong->right and {lost} right->wrong.")
        n, g, lost = flips(base, trained, lambda r: classify_gsm8k(r) in LENIENT_OK)
        print(f"Per-problem (lenient): {g} went wrong->right and {lost} right->wrong.\n")

    ab_f = find_one(vg, f"eval-aime26-baseline{sfx}/*.jsonl")
    at_f = find_one(vg, f"eval-aime26-trained{sfx}/*.jsonl")
    if ab_f and at_f:
        ab, at = load(ab_f), load(at_f)
        xb, xt = aime_summary(ab), aime_summary(at)
        print(f"## AIME 2026 before vs. after training (30 problems, {xb['k']} samples each)\n")
        print("| | Before | After |\n|---|---|---|")
        print(f"| Mean accuracy (mean@{xb['k']}) | {pct(xb['mean_at_k'])} | {pct(xt['mean_at_k'])} |")
        if xb["k"] > 1:
            print(f"| Solved at least once (pass@{xb['k']}) | {pct(xb['pass_at_k'])} | {pct(xt['pass_at_k'])} |")
        print(f"| No \\boxed{{}} (mostly hit the length limit) | {pct(xb['invalid'])} | {pct(xt['invalid'])} |")
        print(f"| Mean response length (chars) | {xb['chars_mean']:.0f} | {xt['chars_mean']:.0f} |")
        print(f"| Responses with reflection words | {pct(xb['reflect'])} | {pct(xt['reflect'])} |")
        print(f"| Likely looping | {pct(xb['looping'])} | {pct(xt['looping'])} |")
        n, g, lost = flips(ab, at, aime_ok)
        print(f"\nPer-problem (majority of samples correct): {g} went wrong->right and {lost} right->wrong.\n")

    train_dirs = [d for d in sorted(vg.iterdir()) if d.name.startswith("gsm8k-") and (("-think" in d.name) == args.think)]
    if not train_dirs:
        return
    name = train_dirs[-1].name
    log = find_one(root, f"{name}-*.log")
    val_stats = {int(p.stem): gsm8k_summary(load(p)) for p in train_dirs[-1].glob("*.jsonl")}
    rollout_dir = root / "rollout_generations" / name
    groups = rollout_group_stats(rollout_dir) if rollout_dir.exists() else {}

    print(f"## Validation during training ({name}, GSM8K test set)\n")
    head = "| step | strict acc | lenient acc | right answer, bad format | likely truncated | mean length (chars) | reflection words |"
    print(head + "\n|---|---|---|---|---|---|---|")
    for s in sorted(val_stats):
        v = val_stats[s]
        slip = v["counts"][RIGHT_BAD_FORMAT] + v["counts"][NO_HASH_RIGHT]
        cut = v["counts"][NO_HASH_TRUNCATED] + v["counts"][UNFINISHED]
        print(f"| {s} | {pct(v['strict_acc'])} | {pct(v['lenient_acc'])} | {slip} | {cut} | "
              f"{v['chars_mean']:.0f} | {pct(v['reflect'])} |")

    if groups:
        print("\n## GRPO groups (training rollouts)\n")
        print_table(groups, every=max(1, len(groups) // 12))

    if log:
        steps = parse_train_log(log)
        out = root / "training_curves.png"
        plot(steps, val_stats, groups, out)
        print(f"\nTraining curves saved to {out}")
        keys = ["critic/score/mean", "response_length/mean", "response_length/clip_ratio", "actor/entropy",
                "actor/kl_loss", "actor/pg_clipfrac", "actor/grad_norm", "timing_s/step"]
        print("\nTraining metrics (mean of first 5 steps -> mean of last 5 steps):")
        xs = sorted(s for s in steps if "critic/score/mean" in steps[s])
        for k in keys:
            vals = [steps[s][k] for s in xs if k in steps[s]]
            if len(vals) >= 10:
                print(f"- {k}: {statistics.mean(vals[:5]):.4g} -> {statistics.mean(vals[-5:]):.4g}")


if __name__ == "__main__":
    main()
