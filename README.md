# LLM RAG Correction System

A Python research script that reads answers collected by the [batch runner](https://github.com/brianfeng31/LLM_batch_runner), uses selected peer answers as context, and asks a chosen model to answer the question again.

## How it works

1. Load the question and model answers from `results.csv`.
2. The caller chooses the model to revise and the reference models.
3. Embed the **question** and reference answers using `paraphrase-MiniLM-L6-v2`, then order the references by cosine similarity. The correction workflow includes all selected references; the retrieval helper also supports returning only the top k.
4. Put the reference **text** and question into a new prompt and call the selected model through Ollama.
5. Save the original and revised answers to `rag_corrections.csv`.

This script implements the retrieval and re-prompting portion of the research. Reference selection is manual: it does not implement the paper's evaluator, automatically choose a weaker model, or apply a 0.85 cutoff. Peer agreement and cosine similarity do not establish factual correctness. The arguments `wrong_model` and `correct_models` describe the caller's selections.

## Setup

Use Python 3.12 or later. From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OLLAMA_API_KEY="your-api-key"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` and set the key with `$env:OLLAMA_API_KEY="your-api-key"`. See [Ollama's cloud setup](https://docs.ollama.com/cloud) for obtaining a key.

The first retrieval downloads the embedding model; later runs reuse its cache. Importing `RAG.py` does not download weights or require an API key.

## Usage

Run the batch runner and copy its `results.csv` into this repository. Review the answers and select suitable references. Edit the example at the bottom of `RAG.py`, then run `python RAG.py`. Or call the function directly:

```python
from RAG import run_rag_correction

run_rag_correction(
    prompt="What is the capital of France?",
    wrong_model="gemma",
    correct_models=["gpt_oss_120b", "deepseek", "qwen3"],
    results_csv="results.csv",
    output_csv="rag_corrections.csv",
)
```

CSV paths are relative to `RAG.py`; absolute paths also work. Each question/model pair must have one response. Empty answers and `OLLAMA_CLOUD_ERROR:` entries are excluded. Missing references or a target used as its own reference cause a clear error. Failed generations are not saved as corrected answers. Each successful run appends one row.

Output columns are `prompt`, `wrong_model`, `original_answer`, `corrected_answer`, `correct_models_used`, and `num_correct_references`. “Corrected” means revised, not independently verified. The example model selections illustrate usage and are not a quality ranking; the prompt must match the question in the input CSV.

## Models

`MODEL_MAP` maps CSV aliases to API model IDs from the [public Ollama model list](https://ollama.com/api/tags), checked September 22, 2026:

| Alias | Current API model |
| --- | --- |
| `gpt_oss_120b` | `gpt-oss:120b` |
| `deepseek` | `deepseek-v4.1-flash` |
| `qwen3` | `qwen3.5:397b` |
| `gemma` | `gemma4:31b` |

These defaults are **not the exact historical experiment lineup**. Keep `MODEL_MAP` consistent with the batch runner's `ALL_MODELS` so new answers and revisions use the same models. For historical CSVs, configure the original model if still available; using a newer model is a separate experiment.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests use controlled vectors and mocked cloud responses to check retrieval, reference validation, API errors, and CSV output, without a key or model download. A separate local check exercised the actual pretrained encoder. Authenticated live cloud generation has not been verified.
