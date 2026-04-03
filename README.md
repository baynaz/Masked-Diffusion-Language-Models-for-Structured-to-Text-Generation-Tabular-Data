# MaskTable: Masked Diffusion LMs for Tabular-to-Text Generation

> *Can non-autoregressive generation better capture the unordered nature of tabular data?*

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
