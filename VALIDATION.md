# Validation

Checked locally on 2026-09-22 with Python 3.12.14 on macOS.

| Check | Result |
|---|---|
| Automated unit/integration tests | 27 passed; no external network, credentials, or embedding dependencies required |
| Actual HTTP serialization/retry | Local loopback stub served model-list and chat endpoints; HTTP 503 retried successfully |
| Offline end-to-end demo | 9 synthetic initial responses; 3 synthetic revisions; no errors |
| Optional reviewer interface | Synthetic JSON review validated; malformed review and retry behavior covered in tests |
| Resume | Collection and revision checkpoints tested, including reading numeric metadata back from CSV and resuming after interruption |
| Real sentence embeddings | Loaded `paraphrase-MiniLM-L6-v2` at revision `c9a2bfebc254878aee8c3aca9e6844d5bbb102d1`; 9 synthetic responses evaluated; 1 revision and 2 abstentions at threshold 0.85; no errors |
| Public Ollama API | Actual HTTP request to `/api/tags` succeeded; configured example model IDs were advertised |
| Authenticated LLM generation/reviewer | **Not run:** no `OLLAMA_API_KEY` was configured in this environment |

The real-embedding check still uses synthetic generation responses. Neither demo is an experiment measuring a live model's accuracy. The example report is marked with its provenance. The scoring method in this version is explicitly defined and should not be presented as a reproduction of the original paper's underspecified evaluator or numerical results.

Reproduce the automated checks and demo:

```bash
python3 -m unittest discover -s tests -v
python3 pipeline.py --demo --reviewer fixture-reviewer --output-dir runs/check
python3 pipeline.py --demo --reviewer fixture-reviewer --output-dir runs/check --resume
```

After installing `requirements.txt`, run the real-embedding check:

```bash
python3 pipeline.py --demo --real-embeddings --output-dir runs/embedding-check
```

To complete authenticated verification, configure your key and run the small live example in the README. That run sends the example questions to the configured provider. Report its actual outputs and statuses; do not relabel synthetic examples as live results.
