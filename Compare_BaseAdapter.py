import argparse
import json
import os

def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def compare_results(base_path, adapter_path, output_filename, num_examples):
    base_results = load_results(base_path)
    adapter_results = load_results(adapter_path)

    assert len(base_results) == len(adapter_results), "Mismatched number of examples!"

    total = len(base_results)
    base_correct = 0
    adapter_correct = 0
    both_correct = 0
    both_wrong = 0
    adapter_only = 0
    base_only = 0
    flipped_examples = []

    for base, adapter in zip(base_results, adapter_results):
        b = base["is_correct"]
        a = adapter["is_correct"]

        if b and a:
            both_correct += 1
        elif not b and not a:
            both_wrong += 1
        elif not b and a:
            adapter_only += 1
            flipped_examples.append({
                "prompt": adapter["prompt"],
                "base_predicted": base["predicted"],
                "adapter_predicted": adapter["predicted"],
                "correct": adapter["correct"],
                "change": "adapter_fixed"
            })
        elif b and not a:
            base_only += 1
            flipped_examples.append({
                "prompt": adapter["prompt"],
                "base_predicted": base["predicted"],
                "adapter_predicted": adapter["predicted"],
                "correct": adapter["correct"],
                "change": "adapter_wrong"
            })

    base_correct = sum(r['is_correct'] for r in base_results)
    adapter_correct = sum(r['is_correct'] for r in adapter_results)

    summary = {
        "total": total,
        "base_correct": base_correct,
        "adapter_correct": adapter_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "adapter_only": adapter_only,
        "base_only": base_only,
        "net_gain": adapter_only - base_only
    }

    print("=== Comparison Summary ===")
    for k, v in summary.items():
        print(f"{k.replace('_', ' ').capitalize()}: {v}")

    print(f"\n=== Changed Predictions (first {num_examples} examples) ===")
    for ex in flipped_examples[:num_examples]:
        print(f"\nPrompt: {ex['prompt']}")
        print(f"Correct:  {ex['correct']}")
        print(f"Base:     {ex['base_predicted']}")
        print(f"Adapter:  {ex['adapter_predicted']}")
        print(f"Change:   {ex['change']}")

    # Construct full path under "comparison/"
    output_dir = "comparison"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Save summary and examples
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "examples": flipped_examples[:num_examples]},
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\n📁 Comparison saved to: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare base and adapter model results on a cloze task.")

    parser.add_argument(
        "--base_results",
        type=str,
        required=True,
        help="Path to base model's results.jsonl"
    )
    parser.add_argument(
        "--adapter_results",
        type=str,
        required=True,
        help="Path to adapter model's results.jsonl"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_summary.json",
        help="Filename to save comparison summary (will be placed in ./comparison/)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of changed prediction examples to print and save"
    )

    args = parser.parse_args()

    compare_results(args.base_results, args.adapter_results, args.output, args.num_examples)
