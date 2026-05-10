# MaskTable: Masked Diffusion LMs for Tabular-to-Text Generation

> *Can non-autoregressive generation better capture the unordered nature of tabular data?*
---
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Model](https://img.shields.io/badge/Model-MDLM-8a2be2)
![Baseline](https://img.shields.io/badge/Baseline-GPT--2-lightgrey)
![Dataset](https://img.shields.io/badge/Dataset-ToTTo-4caf50)
![Task](https://img.shields.io/badge/Task-Table--to--Text-ff69b4)
![Finetuning](https://img.shields.io/badge/Finetuning-None-red)
![Setting](https://img.shields.io/badge/Setting-Zero%2FFew--Shot-orange)
---

## Overview

This project investigates whether **Masked Diffusion Language Models (DLMs)** are better suited than autoregressive models for **structured-to-text generation** — specifically, generating fluent and factually faithful text descriptions from tabular data.

Autoregressive models (like GPT-2) generate text left-to-right, imposing a linear order that doesn't naturally align with the relational, non-sequential structure of tables. Masked diffusion models, by contrast, generate text through iterative demasking with full bidirectional context — making them theoretically well-suited for this task.

We test this hypothesis empirically using the **ToTTo** benchmark dataset, comparing **MDLM** against **GPT-2** in zero-shot and few-shot settings, with no fine-tuning.

---

## Research Question

> Are masked diffusion language models better than autoregressive models at generating faithful text descriptions from tabular data?

---

## Key Contributions

- **Novel task formulation**: First application of masked diffusion LMs to conditional structured-to-text generation
- **Testable hypothesis**: Non-sequential generation should benefit tasks where the source (a table) has no natural linear order
- **Controlled comparison**: MDLM vs. GPT-2 under identical zero-shot / few-shot conditions, no fine-tuning

---

## Models

| Model | Type | Parameters | Fine-tuning |
|-------|------|-----------|-------------|
| [MDLM](https://huggingface.co/) | Masked Diffusion LM | ~110M | Zero-shot |
| GPT-2 small | Autoregressive | ~117M | Zero-shot |

**Conditioning strategy:**
- **MDLM**: Serialized table provided as unmasked prefix context; only the description tokens are progressively demasked
- **GPT-2**: Serialized table provided as prompt; description generated autoregressively

---

## Dataset

**[ToTTo](https://github.com/google-research-datasets/totto)** (Parikh et al., 2020) — Wikipedia tables paired with controlled natural language descriptions.

- Training subset: 5,000 examples
- Evaluation subset: 500 examples
- Table serialization format: `column | value | column | value | ...`
- Max description length: 64 tokens

---

## Project Structure
### Notebooks

All project notebooks are stored in the `notebooks/` folder.

STEPS:
- Environment setup, ToTTo dataset exploration, table serialization, and subset construction. 
- Conditional text generation using GPT-2 (zero-shot and few-shot prompting). 
- Masked Diffusion Language Model (MDLM) setup and conditional generation. 
- Generation of multiple outputs per table on the evaluation subset. 
- Automatic evaluation using BLEU, ROUGE, and BERTScore. 
- Human evaluation of factual consistency. 
- Comparative analysis and final conclusions. 
---

## Evaluation

Each model generates **3 descriptions per table** (for diversity analysis).

| Metric | Purpose | Tool |
|--------|---------|------|
| BLEU / ROUGE | Surface-level overlap with reference | `evaluate` |
| BERTScore | Semantic faithfulness | `evaluate` |
| Self-BLEU | Generation diversity | `nltk` |
| Manual faithfulness | Are numbers/facts correctly transcribed? | Human (50 examples) |

---
## Quickstart
```bash
# Clone the repo
git clone https://github.com/baynaz/Masked-Diffusion-Language-Models-for-Structured-to-Text-Generation-Tabular-Data.git
cd Masked-Diffusion-Language-Models-for-Structured-to-Text-Generation-Tabular-Data

```

Or open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

##  Dataset Preparation

### The ToTTo Dataset
- **Source**: `totto_train_data.jsonl` downloaded from the official Google repository
- **Format**: JSONL (JSON Lines) — one example per line, allowing the file to be read
  line by line without loading all ~120,000 examples into memory at once
- **Each example contains**:
  - `table`: the Wikipedia table as a list of rows, each cell having a `value`,
    an `is_header` flag, and merge info (`column_span`, `row_span`)
  - `highlighted_cells`: list of `[row_idx, col_idx]` coordinates pointing to the cells
    **relevant** to the description — human annotations provided by Google
  - `sentence_annotations`: several versions of the reference description;
    we use `final_sentence` as the ground truth for evaluation
  - `table_page_title`, `table_section_title`: contextual metadata

#### Why highlighted_cells?
- A Wikipedia table can contain dozens of rows and columns
- The reference description only concerns **a few specific cells**
- Example: a table with 105 cells whose description only mentions one,
  `"A Favorita"` → `highlighted_cells: [[13, 2]]`
- By only serializing these cells, we give the model **the relevant information**
  without drowning it in unnecessary noise

#### Table Serialization
- Language models (GPT-2, MDLM) only understand **plain text**
- Serialization converts the highlighted cells into a readable structured sentence: "Title : A Favorita | Director : Ricardo Waddington | Ibope Rating : 39.5"
- Chosen format: `"header : value"` separated by `|` — simple, unambiguous,
  and compatible with the limited context window of the models

#### Building the Subsets
- The full file (~120,000 examples) is too large for free Colab T4
- We extract two subsets by filtering out invalid examples
  (missing `highlighted_cells` or empty `final_sentence`):
  - **train_subset**: 5,000 examples — used to build few-shot prompts
  - **eval_subset**: 500 examples — used for BLEU / ROUGE / BERTScore evaluation
- Subsets are saved as local JSON files to avoid re-reading the JSONL
  at every Colab session
---

#### Expected Results

- Quantitative comparison of MDLM vs. GPT-2 across zero-shot, 3-shot, 5-shot, and 10-shot settings
- Qualitative examples of generated descriptions from real Wikipedia tables
- Analysis of factual faithfulness (number/entity transcription accuracy)
- Conclusion on whether DLMs offer a structural advantage for conditional generation from unordered data

---

## References

- Parikh et al. (2020) — [ToTTo: A Controlled Table-To-Text Generation Dataset](https://aclanthology.org/2020.emnlp-main.89/) — EMNLP 2020
- Lou et al. (2023) — Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (MDLM)
- Ni et al. (2025) — [Diffusion Language Models are Super Data Learners](https://arxiv.org/abs/2511.03276)
- Austin et al. (2021) — Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM) — NeurIPS 2021
- Radford et al. (2019) — Language Models are Unsupervised Multitask Learners (GPT-2)
- Zhang et al. (2020) — BERTScore: Evaluating Text Generation with BERT — ICLR 2020

---

## License


