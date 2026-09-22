"""Original manual-correction entry point; pipeline.py provides automatic evaluation.

The legacy wrong_model/correct_models argument names are retained for callers.
They express a user's selections, not a guarantee of factual correctness.
"""
import csv
import io
import json
from pathlib import Path
import sys

from batch_llm_runner import DEFAULT_MODELS, OllamaClient, atomic_text, read_results
from rag_core import SentenceEncoder, create_rag_prompt, retrieve_references


def load_results(results_csv="results.csv"):
    return read_results(results_csv)


def get_answers_for_prompt(rows, prompt):
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    return {row["model"]: row["response"] for row in rows
            if row["prompt"] == prompt and row.get("status", "success") == "success"
            and row["response"].strip() and not row["response"].startswith("OLLAMA_CLOUD_ERROR:")}


def retrieve_top_k_similar(query_answer, correct_answers, top_k=3, *, encoder=None):
    references = [{"model": str(i), "response": answer} for i, answer in enumerate(correct_answers)]
    selected = retrieve_references(query_answer, references, encoder or SentenceEncoder(), top_k)
    return "\n\n---\n\n".join(item["response"] for item in selected)


def correct_model_answer(wrong_model_name, original_prompt, correct_answers, *, client=None, encoder=None, top_k=3):
    if not correct_answers:
        raise ValueError("Select at least one reference answer.")
    client = client or OllamaClient()
    model_id = DEFAULT_MODELS.get(wrong_model_name, wrong_model_name)
    references = [{"model": f"reference_{i + 1}", "response": answer} for i, answer in enumerate(correct_answers)]
    selected = retrieve_references(original_prompt, references, encoder or SentenceEncoder(), top_k)
    result = client.generate(model_id, create_rag_prompt(original_prompt, selected))
    if result.status != "success":
        raise RuntimeError(result.error)
    return result.response


def run_rag_correction(prompt, wrong_model, correct_models, results_csv="results.csv",
                       output_csv="rag_corrections.csv", *, client=None, encoder=None, top_k=3,
                       model_id=None):
    """Manual reference selection, retained for compatibility with the original script."""
    if Path(results_csv).resolve() == Path(output_csv).resolve():
        raise ValueError("Correction output cannot overwrite the input results.")
    if wrong_model in correct_models or len(set(correct_models)) != len(correct_models):
        raise ValueError("Reference models must be distinct and exclude the target.")
    rows = load_results(results_csv)
    answers = get_answers_for_prompt(rows, prompt)
    if wrong_model not in answers:
        raise ValueError("The target has no successful answer for this exact prompt.")
    missing = set(correct_models) - answers.keys()
    if missing or not correct_models:
        raise ValueError(f"Select available successful reference models. Missing: {sorted(missing)}")
    target = next(row for row in rows if row["prompt"] == prompt and row["model"] == wrong_model)
    api_model = model_id or target["model_id"]
    client = client or OllamaClient()
    if hasattr(client, "preflight"):
        client.preflight([api_model])
    references = [{"model": alias, "response": answers[alias]} for alias in correct_models]
    selected = retrieve_references(prompt, references, encoder or SentenceEncoder(), top_k)
    result = client.generate(api_model, create_rag_prompt(prompt, selected))
    if result.status != "success":
        raise RuntimeError(result.error)
    record = {"prompt": prompt, "wrong_model": wrong_model, "model_id": api_model,
              "original_answer": answers[wrong_model], "corrected_answer": result.response,
              "reference_models": json.dumps([item["model"] for item in selected]),
              "reference_selection": "manual; not independently verified"}
    existing = []
    if Path(output_csv).exists():
        with Path(output_csv).open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    existing = [row for row in existing if (row.get("prompt"), row.get("wrong_model")) != (prompt, wrong_model)]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(record), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(existing + [record])
    atomic_text(output_csv, buffer.getvalue())
    return result.response


if __name__ == "__main__":
    from pipeline import main
    sys.exit(main())
