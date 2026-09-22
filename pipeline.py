"""Collect peer answers, measure agreement, and generate reference-assisted revisions."""
import argparse
from pathlib import Path
import sys

from batch_llm_runner import add_client_arguments, client_from_args, load_models, read_results, run_batch
from rag_core import DEFAULT_ENCODER, SentenceEncoder, evaluate_and_correct


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use synthetic responses and vectors, with no API or model downloads")
    parser.add_argument("--real-embeddings", action="store_true", help="Use real sentence embeddings with the synthetic demo responses")
    parser.add_argument("--prompts", default="examples/prompts.csv")
    parser.add_argument("--models-file", help="JSON mapping aliases to current Ollama model IDs")
    parser.add_argument("--results", help="Evaluate an existing results CSV instead of collecting answers")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--reviewer", help="Optional evaluator API model ID (fixture-reviewer in demo mode)")
    parser.add_argument("--analyze-only", action="store_true", help="Measure agreement without correction or review API calls")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    add_client_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if not 0 < args.threshold <= 1 or not 2 <= args.top_k <= 12:
            raise ValueError("Use a threshold in (0, 1] and top-k between 2 and 12.")
        if args.retry_failed and not args.resume:
            raise ValueError("--retry-failed requires --resume.")
        if args.real_embeddings and not args.demo:
            raise ValueError("Live/imported runs already use real embeddings; --real-embeddings is a demo option.")
        if args.demo and (args.results or args.models_file):
            raise ValueError("Demo mode uses bundled fixture models and prompts, not imported results/model configurations.")
        if args.analyze_only and args.reviewer:
            raise ValueError("--reviewer is only used when generating revisions.")
        output = Path(args.output_dir or ("runs/demo" if args.demo else "runs/live"))
        if args.demo:
            from demo_fixtures import FixtureClient, FixtureEncoder, MODELS
            if args.reviewer and args.reviewer != "fixture-reviewer":
                raise ValueError("Use --reviewer fixture-reviewer in the synthetic demo.")
            client = FixtureClient()
            models = MODELS
            prompts_path = Path(__file__).parent / "examples" / "prompts.csv"
            encoder = SentenceEncoder(args.encoder) if args.real_embeddings else FixtureEncoder()
            provenance = "SYNTHETIC responses and reviewer; " + ("REAL sentence embeddings" if args.real_embeddings else "SYNTHETIC vectors; no LLM or learned embedding inference")
        else:
            client = client_from_args(args)
            models = load_models(args.models_file)
            prompts_path = args.prompts
            if not args.results:
                client.preflight(list(models.values()) + ([args.reviewer] if args.reviewer else []))
            elif not args.analyze_only:
                imported = read_results(args.results)
                target_ids = [row["model_id"] for row in imported if row["status"] == "success"]
                client.preflight(target_ids + ([args.reviewer] if args.reviewer else []))
            encoder = SentenceEncoder(args.encoder)
            provenance = "imported responses; real embeddings" if args.results else "live Ollama responses; real embeddings"
        if args.results:
            rows = read_results(args.results)
        else:
            rows = run_batch(prompts_path, output / "results.csv", models=models, client=client,
                             resume=args.resume, retry_failed=args.retry_failed)
        entries = evaluate_and_correct(rows, None if args.analyze_only else client, encoder, output,
                                       threshold=args.threshold, top_k=args.top_k, reviewer=args.reviewer,
                                       resume=args.resume, retry_failed=args.retry_failed,
                                       analyze_only=args.analyze_only, provenance=provenance)
        revised = sum(entry["status"] == "corrected" for entry in entries)
        errors = sum(entry["status"] in {"excluded_api_error", "correction_error"} or
                     (entry.get("review") or {}).get("status") == "error" for entry in entries)
        print(f"Evaluated {len(entries)} responses; revised {revised}; errors {errors}.")
        print(f"Report: {output / 'report.md'}")
        return int(errors > 0)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Checkpoints are saved; rerun with --resume and the same settings.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
