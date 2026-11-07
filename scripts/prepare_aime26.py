"""把 math-ai/aime26 转成 verl 使用的 parquet 格式。

data_source 以 "aime" 开头时，verl 会用 math_dapo 评分。
它读取 "Answer: ..." 或 \\boxed{...}，这里要求模型使用 \\boxed{}。
"""

import argparse
import os

INSTRUCTION = (
    "Let's think step by step and put the final answer, "
    "an integer from 0 to 999, in \\boxed{}."
)


def to_verl_record(example, index):
    problem = example["problem"]
    return {
        "data_source": "aime26",
        "prompt": [{"role": "user", "content": f"{problem} {INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(example["answer"]).strip()},
        "extra_info": {"split": "test", "index": index, "id": example["id"], "question": problem},
    }


def main():
    import datasets

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dataset = datasets.load_dataset("math-ai/aime26", split="test")
    dataset = dataset.map(to_verl_record, with_indices=True, remove_columns=dataset.column_names)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "test.parquet")
    dataset.to_parquet(output_path)
    print(f"已写入 {len(dataset)} 道题：{output_path}")


if __name__ == "__main__":
    main()
