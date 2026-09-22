# LLM response comparison and RAG revision

A Python research demo that asks several models the same questions, measures agreement between their answers, and gives an outlying model selected peer answers as context for a second response.

The original project collected LLM outputs and used peer answers for RAG. This version makes collection, scoring, retrieval, and revision explicit and runnable. It **does not reproduce the paper's reported measurements**: the paper did not fully specify its evaluator or scoring formula, and this version documents its own numerical method. Peer agreement is not factual accuracy.

## One-command demo

Requires Python 3.12 or newer. From this repository:

```bash
python3 pipeline.py --demo
```

Open `runs/demo/report.md`. No API key, third-party packages, model downloads, or paid requests are required. **The demo uses synthetic answers and manually assigned vectors**, including an intentionally unrelated answer for each question. It exercises the software, not model quality. See the [generated example report](examples/demo-report.md).

```bash
# Resume without repeating completed work.
python3 pipeline.py --demo --resume

# Exercise the optional reviewer interface with a synthetic JSON response.
python3 pipeline.py --demo --reviewer fixture-reviewer --output-dir runs/demo-with-reviewer
```

## Real sentence embeddings

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python pipeline.py --demo --real-embeddings --output-dir runs/real-embeddings
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. The dependency lock was resolved and tested with Python 3.12. The first semantic run downloads `sentence-transformers/paraphrase-MiniLM-L6-v2`. This mode still uses synthetic generation responses but computes real embeddings and cosine similarities; its revision count can differ from the synthetic-vector demo.

## Live Ollama run

Set `OLLAMA_API_KEY` in your environment. `.env.example` lists the variables. Scripts do not load `.env` automatically; on Bash/Zsh, after copying and editing it, use `set -a; source .env; set +a`. Never commit a populated `.env`.

```bash
# Check model IDs and update examples/models.json if necessary.
python batch_llm_runner.py --list-models

# Three questions, three models: nine initial API calls, plus eligible revisions.
python pipeline.py --prompts examples/prompts.csv --models-file examples/models.json --output-dir runs/live

# Resume the same inputs and settings; explicitly retry failed requests.
python pipeline.py --prompts examples/prompts.csv --models-file examples/models.json --output-dir runs/live --resume --retry-failed

# Optional LLM assessment, separate from numerical similarity.
python pipeline.py --output-dir runs/with-reviewer --reviewer gpt-oss:120b

# Analyze an existing CSV without correction/reviewer API calls.
python RAG.py --results results.csv --analyze-only --output-dir runs/analysis

# Collect answers only, independently of the RAG code.
python batch_llm_runner.py --output runs/batch/results.csv
```

Cloud usage is governed by your Ollama account. A reviewer adds one request per successful revision. Example model IDs were advertised by the provider on 2026-09-22; these are not the original paper's model lineup. Availability can change, so preflight rejects unavailable IDs. [Ollama Cloud documentation](https://docs.ollama.com/cloud).

For local Ollama, set `OLLAMA_HOST=http://localhost:11434`, leave the key unset, and supply a JSON alias-to-ID mapping with three or more different installed models. The scripts do not install generation models.

Paths are relative to the current working directory unless absolute, except the bundled demo resolves its prompt file relative to the script. See `--help` for threshold, top-k, timeout, retries, token limit, temperature, and embedding-model options.

## Implemented method

1. **Collect:** send each unique question to each distinct API model. Store answer text, alias, model ID, status, latency, attempts, and timestamp. Failures have a separate error field and an empty answer.
2. **Embed:** convert successful answers to the same question into sentence vectors.
3. **Compare:** an answer's score is its **mean cosine similarity to all other successful answers**, excluding itself. Below the configurable threshold (default `0.85`) means a candidate for revision, not a proven factual error.
4. **Choose references:** find the unique largest group in which every pair meets the threshold. Require at least two references and three successful models overall. Abstain when equally large groups compete or no agreeing pair exists. Members of the reference group are not revised just because an outlier lowered their overall mean.
5. **Retrieve:** embed the **original question**, rank the reference answers by similarity to it, and keep up to `--top-k`. The target cannot be its own reference.
6. **Revise:** send the original question and actual reference text to the target model. References are explicitly described as unverified peer answers, not ground truth.
7. **Measure:** compare the revised answer to the **same original peer answers** used for the before score. Store any optional evaluator judgment separately.

The exact reference-group search is bounded to 12 successful models per question. The largest agreeing group need not be a majority, and agreement is not independent factual verification. Long answers can exceed the embedding model's token limit; the examples intentionally request short answers. Thresholds require validation against suitable labeled data before accuracy claims. This uses references stored in CSV, not a vector database, and does not train or fine-tune models.

## Reliability and outputs

- Configurable request timeouts and bounded retries for connection failures, rate limits, and selected server errors; permanent client errors are not repeatedly retried.
- Empty/truncated responses are failures and are not embedded as answers.
- CSV/JSON checkpoints are flushed and atomically replaced after completed calls. A process stopping between a remote response and its local checkpoint can still repeat that one request.
- Resume checks input/configuration fingerprints. Changed settings require a new output location. `--retry-failed` can retry a failed reviewer without regenerating its successful revision.
- Run only one process per output location; checkpoints are not a multi-writer database.

| File | Contents |
|---|---|
| `results.csv` | Answers, model IDs, request statuses, timing, attempts |
| `results.csv.meta.json` | Input/configuration fingerprint and settings, without credentials |
| `corrections.checkpoint.json` | Completed revisions and reviewer results for resume |
| `report.json` | Structured scores, selected references, revision statuses |
| `report.md` | Human-readable before/after report |

Exit codes: `0` completed without request errors; `1` report produced with collection/revision/reviewer errors; `2` configuration/input error; `130` interrupted. Skipped revisions have explicit reasons and are not API failures.

## Existing CSVs and manual correction

Original CSVs with `prompt,model,response,latency_sec` are accepted for analysis. Legacy `OLLAMA_CLOUD_ERROR:` text is excluded. Live correction also needs actual API IDs in a `model_id` column; old aliases are not silently mapped to replacement models.

The original manual function remains:

```python
from RAG import run_rag_correction

run_rag_correction(
    prompt="What is the capital of France? Answer in one sentence.",
    wrong_model="gemma",
    correct_models=["gpt_oss_120b", "qwen"],
    results_csv="runs/live/results.csv",
    output_csv="runs/manual.csv",
    top_k=2,
)
```

`wrong_model` and `correct_models` are legacy names for user-selected targets/references, not guarantees of correctness. This path bypasses automatic selection. For legacy CSVs without API IDs, pass `model_id` explicitly. `RAG.py` also supports the same CLI as `pipeline.py`.

## Code map and checks

- `batch_llm_runner.py`: HTTP client, validation, retries, atomic collection checkpoints.
- `rag_core.py`: embeddings, scoring, reference selection, revision/reviewer prompts, reports.
- `pipeline.py`: end-to-end command line.
- `RAG.py`: original manual entry point.
- `demo_fixtures.py`: labeled synthetic test data and responses.
- `tests/`: failure, resume, reference-selection, and CLI integration tests.

```bash
python3 -m unittest discover -s tests -v
```

Tests require no credentials, embedding dependencies, or internet access. GitHub Actions runs them and the offline demo. See [VALIDATION.md](VALIDATION.md) for checks actually performed. The standalone collector is also maintained in [LLM_batch_runner](https://github.com/brianfeng31/LLM_batch_runner).
