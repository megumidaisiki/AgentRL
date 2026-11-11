"""Convert math-ai/aime26 into the parquet format verl expects.

When data_source starts with "aime", verl scores answers with math_dapo, which reads
"Answer: ..." or \\boxed{...}. The prompt here asks the model to use \\boxed{}.
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
    print(f"Wrote {len(dataset)} problems to {output_path}")


if __name__ == "__main__":
    main()
