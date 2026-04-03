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

## Requirements
```bash
pip install torch

```

---


## Quickstart
```bash
# Clone the repo
git clone https://github.com/your-username/masktable.git
cd masktable

# Install dependencies
pip install -r requirements.txt
```

Or open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## Expected Results

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


