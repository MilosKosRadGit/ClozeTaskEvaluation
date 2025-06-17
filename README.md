# Serbian Cloze Evaluation Task

This repository evaluates causal language models on a grammatical cloze task in Serbian using multiple-choice completions. It uses masked-prompt scoring and works with Hugging Face transformer models.

## Introduction

This project evaluates the grammatical and linguistic competence of causal language models on a **Serbian cloze task** using multiple-choice masked prompts. Designed for morphologically rich languages, the dataset probes fine-grained linguistic phenomena including grammar, semantics, idioms, and lexical knowledge. It supports the evaluation of both base models and adapter-enhanced models (e.g., LoRA fine-tuning), providing insights into how well these models capture native-like linguistic fluency in Serbian.

---

## Project Structure

```bash
CLOZETASKEVAL/
├── comparison/
│   └── comparison_summary.json        # JSON output comparing base and adapter results
├── data/
│   └── serbian_cloze_eval.jsonl       # Main evaluation dataset (Serbian cloze task)
├── results_adapter.jsonl              # Predictions from adapter-enhanced model
├── results_base.jsonl                 # Predictions from base model
├── Cloze_task_onBase.py               # Evaluates base model on cloze task
├── Cloze_task_onAdapter.py            # Evaluates PEFT adapter model on cloze task
├── Compare_BaseAdapter.py             # Compares results from base and adapter models
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation

```

## Serbian Cloze Dataset

The dataset is a list of multiple-choice cloze task instances formatted in JSONL (.jsonl) where each entry is a single evaluation sample. An example format:

```json
{
  "prompt": "Ana je otišla u __.",
  "candidates": ["školu", "školom", "školi", "škola"],
  "correct": "školu."
}
```

### Felds

- `prompt`: A natural language sentence in Serbian with a missing word marked as `__`.

- `candidates`: A list of plausible word completions (including distractors).

- `correct`: The correct choice based on grammar, meaning, or usage.

### Purpose

The evaluation dataset is designed to measure a model’s ability to handle a wide range of linguistic phenomena in Serbian, including but not limited to:

**Grammatical competence:**
- **Case, gender, and number agreement** between nouns, adjectives, verbs, and pronouns.

- **Verb-argument structure**, including selection of appropriate arguments and inflections.

- **Prepositional-case constructions**, ensuring correct preposition + case pairing.

- **Definiteness and specificity** expressed via short vs. long forms of adjectives.

- **Syntax and morphological** disambiguation in ambiguous or inflection-heavy contexts.

**Lexical and semantic understanding:**
- **Lexical choice**: selecting the most appropriate word among semantically similar options.

- **Semantic plausibility**: ensuring the completion makes factual and contextual sense.

- **Idiomatic usage**: resolving fixed expressions and collocations properly.

- **World knowledge and factual grounding**: picking options based on real-world plausibility.

This diversity of task types makes the dataset a **comprehensive diagnostic tool** for evaluating how well causal LMs understand and generate morphosyntactically and semantically correct Serbian text.

## Scripts Overview

| Script Name      | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `Cloze_task_onBase.py`  | Main script for evaluating a model on a Serbian cloze dataset.              |
| `Cloze_task_onAdapter.py`  | Main script for evaluating a model on a Serbian cloze dataset.              |
| `Compare_BaseAdapter.py`  | Main script for evaluating a model on a Serbian cloze dataset.              |

---

## 1. `Cloze_task_onBase.py`

Evaluates a causal LM (e.g. LLaMA 3.2) by scoring candidate completions after a prompt and comparing predictions to the ground truth.

### Usage

```bash
python Cloze_task_onBase.py \
  --model_name meta-llama/Llama-3.2-3B \
  --dataset_path ./data/serbian_cloze_eval.jsonl \
  --output ./results.jsonl
```
### Arguments

| Argument         | Type | Default                           | Description                                           |
| ---------------- | ---- | --------------------------------- | ----------------------------------------------------- |
| `--model_name`   | str  | `meta-llama/Llama-3.2-3B`         | Name or path of Hugging Face causal LM.               |
| `--dataset_path` | str  | `./data/serbian_cloze_eval.jsonl` | Path to dataset in JSONL format.                      |
| `--no_verbose`   | flag | `False`                           | If set, disables detailed output for each prediction. |
| `--output`       | str  | `./results.jsonl`                 | File to save detailed predictions and evaluation.     |

### Output Format

After running the script, results are saved as results.jsonl with per-example evaluations:

```json
{
  "prompt": "Ana je otišla u __.",
  "predicted": "školu",
  "correct": "školu",
  "is_correct": true
}
```
Additionally, overall accuracy is printed to the console.

## 2. Cloze_task_onAdapter.py

Evaluates a **LoRA-adapted** causal language model (e.g. LLaMA with PEFT adapter) on the Serbian grammatical cloze task using masked prompt scoring.

This script loads a base model and applies a LoRA adapter using the PEFT library to assess fine-tuned performance.

### Usage

```bash
python Cloze_task_onAdapter.py \
  --model_name meta-llama/Llama-3.2-3B \
  --adapter_path ./adapters/my_adapter \
  --dataset_path ./data/serbian_cloze_eval.jsonl \
  --output ./results_adapter.jsonl
```

### Arguments

| Argument         | Type | Default                           | Description                                                     |
| ---------------- | ---- | --------------------------------- | --------------------------------------------------------------- |
| `--model_name`   | str  | `meta-llama/Llama-3.2-3B`         | Name or path of base Hugging Face model.                        |
| `--adapter_path` | str  | *required*                        | Path to the LoRA adapter directory (trained with PEFT).         |
| `--dataset_path` | str  | `./data/serbian_cloze_eval.jsonl` | Path to evaluation dataset in JSONL format.                     |
| `--no_verbose`   | flag | `False`                           | If set, disables output for each individual prediction.         |
| `--output`       | str  | `./results_adapter.jsonl`         | File path to save detailed results with prediction correctness. |

### Functionality

- Loads a base model and applies a LoRA adapter using `peft.PeftModel`.

- Evaluates cloze accuracy by scoring each candidate word with log-likelihood.

- Computes and prints overall accuracy.

- Optionally writes full evaluation results to a `.jsonl` file.

### Output Format

Each line in the output file represents one example:

```json
{
  "prompt": "Ana je otišla u __.",
  "predicted": "školu",
  "correct": "školu",
  "is_correct": true
}
```

The script also prints accuracy statistics to the console:

```sql
Final Accuracy: 87.00% (87/100)
```

## 3. Compare_BaseAdapter.py

This script compares prediction results from a base model and a fine-tuned adapter model on the cloze evaluation task. It highlights where the adapter improved or regressed compared to the base model.

### Usage

```bash
python compare_BaseAdapter.py \
  --base_results ./results/base.jsonl \
  --adapter_results ./results/adapter.jsonl \
  --output adapter_vs_base.json \
  --num_examples 10
```

### Arguments

| Argument            | Type | Default                   | Description                                                            |
| ------------------- | ---- | ------------------------- | ---------------------------------------------------------------------- |
| `--base_results`    | str  | *required*                | Path to JSONL file with base model predictions.                        |
| `--adapter_results` | str  | *required*                | Path to JSONL file with adapter model predictions.                     |
| `--output`          | str  | `comparison_summary.json` | Name of the output file (saved in `./comparison/` folder).             |
| `--num_examples`    | int  | `10`                      | Number of changed examples to print and save (adapter win/loss cases). |

### Output

- Prints summary statistics to console, including:

    - Total number of examples

    - Correct predictions for base and adapter

    - Number of examples improved or worsened by adapter

    - Net gain

- Displays and saves up to `--num_examples` changed predictions (e.g., where the adapter fixed an incorrect base prediction).

- Output is saved in the `comparison/` folder under the name provided via `--output` argument.

### Example JSON Output

```json
{
  "summary": {
    "total": 2,
    "base_correct": 1,
    "adapter_correct": 2,
    "both_correct": 1,
    "both_wrong": 0,
    "adapter_only": 1,
    "base_only": 0,
    "net_gain": 1
  },
  "examples": [
    {
      "prompt": "Ana je kupila __ knjigu.",
      "base_predicted": "novo",
      "adapter_predicted": "novu",
      "correct": "novu",
      "change": "adapter_fixed"
    }
  ]
}
```

## Environment Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
2. Install dependences:
```bash
pip install -r requirements.txt
```

### **requirements.txt** Example
```txt
transformers>=4.38.0
torch>=2.1.0
tqdm>=4.65.0
sentencepiece
```

### **.gitignore** Example
```gitignore
__pycache__/
*.pyc
*.log
*.jsonl
results.jsonl
.venv/
.cache/
```

## Notes

- Works best with causal models (e.g. LLaMA, GPT-2, Mistral).

- Evaluation is strict string match (strip()-based).

- Dataset focuses on grammatical agreement and syntax in Serbian.

