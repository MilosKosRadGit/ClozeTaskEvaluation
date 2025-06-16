# Serbian Cloze Evaluation Task

This repository evaluates causal language models on a grammatical cloze task in Serbian using multiple-choice completions. It uses masked-prompt scoring and works with Hugging Face transformer models.

---

## Scripts Overview

| Script Name      | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `Cloze_task.py`  | Main script for evaluating a model on a Serbian cloze dataset.              |
| `data/serbian_cloze_eval.jsonl` | Dataset containing prompts, candidates, and correct answers. |
| `results.jsonl`  | Output of model predictions with accuracy and per-example evaluation.       |

---

## 1. `Cloze_task.py`

Evaluates a causal LM (e.g. LLaMA 3.2) by scoring candidate completions after a prompt and comparing predictions to the ground truth.

### Usage

```bash
python Cloze_task.py \
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

### Dataset Format

The cloze dataset is stored as a JSONL file where each line is a dictionary with the following keys:

```json
{
  "prompt": "Ana je otišla u __.",
  "candidates": ["školu", "školom", "školi", "škola"],
  "correct": "školu."
}
```

- prompt: Prefix sentence with omitted word.

- candidates: List of valid completions (grammatical distractors).

- correct: The correct grammatical form.

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

### Environment Setup

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

### Notes

- Works best with causal models (e.g. LLaMA, GPT-2, Mistral).

- Evaluation is strict string match (strip()-based).

- Dataset focuses on grammatical agreement and syntax in Serbian.
