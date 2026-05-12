# 🔍 SMILES-2026 Hallucination Detection

Detect whether a small language
model's answer is *hallucinated* (fabricated) or *truthful* using the model's
own internal representations (hidden states).

## Overview

Large (and small) language models sometimes *hallucinate* — they generate
plausible-sounding text that is factually incorrect.  This competition asks you
to build a **lightweight binary classifier** (called a *probe*) that reads the
model's internal hidden states and predicts whether a given response is
truthful (`label = 0`) or hallucinated (`label = 1`).

The language model used throughout is **[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)** — a
decoder-only causal transformer with 24 layers and a hidden dimension of 896.
It fits comfortably on a free Google Colab T4 GPU.

**Primary ranking metric:** Accuracy on the held-out `test.csv`.

## Repository Structure

```
SMILES-HALLUCINATION-DETECTION/
├── data/
│   ├── dataset.csv        # Labelled training data (prompt, response, label)
│   └── test.csv           # Unlabelled competition test set
│
├── solution.py            # Main script — produces results.json + predictions.csv
│
│   ── Final model (clean, only the winning configuration) ─────────────
├── aggregation.py         # Pool 3 mid-to-late layers with last + tail_mean
├── probe.py               # HallucinationProbe — 5-seed MLP ensemble
├── splitting.py           # 10-fold stratified train/val/test split
│
│   ── Fixed infrastructure (do not edit) ──────────────────────────────
├── model.py               # Loads Qwen2.5-0.5B
├── evaluate.py            # Evaluation loop, metrics, summary, JSON output
│
├── ablation_summary.csv   # Aggregated metrics for the ~150 ablation runs
├── SOLUTION.md            # Final report (approach, ablations, reproducibility)
├── requirements.txt       # Python dependencies
├── results.json
└── LICENSE
```

## Final model

The shipped configuration pools transformer layers `-14, -12, -10` using two
strategies — the last real-token vector and the mean over the last 16 tokens
— giving a 5376-dim feature. A 2-layer MLP probe (hidden 256, dropout 0.3) is
trained with `BCEWithLogitsLoss(pos_weight)` and 5-seed ensembling, evaluated
under stratified 10-fold cross-validation. Test AUROC ≈ 0.762, test accuracy
≈ 0.753. Per-run metrics from the full sweep (~150 ablations) are in
`ablation_summary.csv`; the selection rationale is written up in
`SOLUTION.md`.


## Quick Start

### Google Colab

Open the terminal in Colab and run:

```python
git clone https://github.com/MichaelToschenko/Hallucination-detection-task.git
cd Hallucination-detection-task
pip install -r requirements.txt
python solution.py
```

### Local Setup

```bash
git clone https://github.com/MichaelToschenko/Hallucination-detection-task.git
cd Hallucination-detection-task

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
python solution.py
```

## Dataset

`data/dataset.csv` contains 689 labelled samples with three columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Full ChatML-formatted conversation context fed to Qwen |
| `response` | str | The model's generated response |
| `label` | float | `1.0` = hallucinated · `0.0` = truthful |

The `prompt` uses the **ChatML** template built into Qwen models:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Given the context, answer the question …<|im_end|>
<|im_start|>assistant
```


`data/test.csv` is structured identically but the `label` column is null - these are the samples you submit predictions for via a `predictions.csv` generated file.


## What You Implement

You are expected to edit **three files**:  
- `aggregation.py`
- `probe.py`
- `splitting.py`

The rest of the codebase shall remain untouched.

**Feature Engineering & Dimensionality Reduction**: Applicants are encouraged to experiment with adding hand-crafted features during the aggregation step, drawing on geometrical or topological methods to enrich the representation of probe outputs. Additionally, you may apply dimensionality reduction techniques within probe.py to compress or refine the feature space.

## Evaluation

For each fold `evaluate.py` reports four numbers:

| # | Checkpoint | Metrics |
|---|-----------|---------|
| 1 | Majority-class baseline | Accuracy, F1 |
| 2 | `HallucinationProbe` on **training** split | Accuracy, F1, AUROC |
| 3 | `HallucinationProbe` on **validation** split | Accuracy, F1, AUROC |
| 4 | `HallucinationProbe` on **test** split | Accuracy, F1, AUROC |

**Accuracy on the `test.csv` is the primary competition metric.**

Results are averaged across folds (if using k-fold) and saved to
`results.json`.


# What is expected from the applicant of SMILES-2026 ?

**Q1:** What must the applicant submit in the application form ?<br>
**A1:** Submit: 
1. A link to your Github repository
2. A link to your `predictions.csv` publicly available file on some cloud storage

**Q2:** What the applicants must include in the repository ?<br>
**A2:** Your repository must contain: 
1. `results.json` - produced by the official `solution.py`
2. Report file in Markdown format `SOLUTION.md`. 

**Q3:** Report requirements (`SOLUTION.md`)<br>
**A3:** Your report must include:<br>
- Reproducibility instructions: exact commands to run your solution and acquire the same `predictions.csv`, required environment (if any), any important implementation details needed to reproduce your result.
- Final solution description: What components you modified ? What your final approach is ? Why you made these choices ? What contributed most to improving the metric ?
- Experiments and failed attempts: What ideas you tried but did not include in the final solution ? Why they did not work or were discarded ?

**Q4:** Reproducibility<br>
**A4:** The repository must be self-contained and runnable with the provided `solution.py` file. Your solution must not require changes to the fixed infrastructure files. Running `solution.py` must generate your submitted `predictions.csv`.
