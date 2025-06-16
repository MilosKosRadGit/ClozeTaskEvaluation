import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score_candidate(model, tokenizer, prompt, candidate):
    input_ids = tokenizer(prompt + candidate, return_tensors="pt")["input_ids"].to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    candidate_ids = input_ids[0, len(prompt_ids[0]):]
    candidate_logits = logits[0, len(prompt_ids[0]) - 1 : -1]
    log_probs = F.log_softmax(candidate_logits, dim=-1)
    scores = log_probs.gather(1, candidate_ids.unsqueeze(1)).squeeze(1)

    return scores.sum().item()

def evaluate_model(model, tokenizer, dataset_path, verbose=True):
    correct = 0
    total = 0
    results = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    for item in tqdm(data, desc="Evaluating"):
        prompt = item["prompt"]
        candidates = item["candidates"]
        correct_answer = item["correct"]

        scores = {cand: score_candidate(model, tokenizer, prompt, cand) for cand in candidates}
        predicted = max(scores, key=scores.get)

        is_correct = predicted.strip() == correct_answer.strip()
        correct += int(is_correct)
        total += 1

        results.append({
            "prompt": prompt,
            "predicted": predicted,
            "correct": correct_answer,
            "is_correct": is_correct
        })

        if verbose:
            print("\n---")
            print(f"Prompt: {prompt}")
            print(f"Predicted: {predicted}")
            print(f"Correct:   {correct_answer}")
            print(f"Correct" if is_correct else "Incorrect")

    accuracy = correct / total if total > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a causal language model on a cloze dataset.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Name or path of the model."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/serbian_cloze_eval.jsonl",
        help="Path to the dataset JSONL file."
    )
    parser.add_argument(
        "--no_verbose",
        action="store_true",
        help="Suppress prompt-by-prompt output."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results.jsonl",
        help="Optional path to save detailed predictions as JSONL."
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    results = evaluate_model(model, tokenizer, args.dataset_path, verbose=not args.no_verbose)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f_out:
            for r in results:
                json.dump(r, f_out, ensure_ascii=False)
                f_out.write("\n")
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
