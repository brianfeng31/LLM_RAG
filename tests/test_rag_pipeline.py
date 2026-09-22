import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from batch_llm_runner import CallResult, read_results, run_batch
from demo_fixtures import CASES, MODELS, FixtureClient, FixtureEncoder
from rag_core import evaluate_and_correct, retrieve_references
from RAG import run_rag_correction


class RecordingClient(FixtureClient):
    def __init__(self, interrupt_at=None, bad_review=False):
        self.calls = []
        self.interrupt_at = interrupt_at
        self.bad_review = bad_review

    def generate(self, model, prompt, *, json_mode=False):
        self.calls.append((model, prompt, json_mode))
        if len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt()
        if json_mode and self.bad_review:
            return CallResult('{"verdict": "certainly_correct"}', "success", attempts=1)
        return super().generate(model, prompt, json_mode=json_mode)


class RagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.prompts = Path(__file__).resolve().parents[1] / "examples" / "prompts.csv"
        self.results = self.directory / "results.csv"
        self.rows = run_batch(self.prompts, self.results, models=MODELS, client=FixtureClient())
        self.encoder = FixtureEncoder()

    def evaluate(self, client, **kwargs):
        return evaluate_and_correct(self.rows, client, self.encoder, self.directory / "report", **kwargs)

    def test_full_correction_flow_uses_peer_text_and_fixed_before_after_comparison(self):
        client = RecordingClient()
        entries = self.evaluate(client)
        corrected = [entry for entry in entries if entry["status"] == "corrected"]
        self.assertEqual(len(corrected), 3)
        self.assertEqual(len(client.calls), 3)
        for entry in corrected:
            self.assertEqual(entry["model"], "fixture_outlier")
            self.assertAlmostEqual(entry["original_agreement"], -1)
            self.assertAlmostEqual(entry["revised_agreement"], 1)
            self.assertEqual({r["model"] for r in entry["references"]}, {"fixture_a", "fixture_b"})
            self.assertNotIn(entry["original_answer"], next(call[1] for call in client.calls if entry["prompt"] in call[1]))
        report = json.loads((self.directory / "report" / "report.json").read_text())
        self.assertEqual(len(report["entries"]), 9)
        self.assertIn("not factual accuracy", report["limitation"])

    def test_completed_corrections_are_not_repeated_on_resume(self):
        self.evaluate(RecordingClient())
        self.rows = read_results(self.results)
        resumed = RecordingClient()
        self.evaluate(resumed, resume=True)
        self.assertEqual(resumed.calls, [])

    def test_interrupted_correction_run_resumes_from_saved_revision(self):
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate(RecordingClient(interrupt_at=2))
        state = json.loads((self.directory / "report" / "corrections.checkpoint.json").read_text())
        self.assertEqual(len(state["entries"]), 1)
        client = RecordingClient()
        self.evaluate(client, resume=True)
        self.assertEqual(len(client.calls), 2)

    def test_invalid_reviewer_json_is_separate_and_retry_does_not_regenerate_answers(self):
        entries = self.evaluate(RecordingClient(bad_review=True), reviewer="fixture-reviewer")
        revised = [entry for entry in entries if entry["status"] == "corrected"]
        self.assertTrue(all(entry["review"]["status"] == "error" for entry in revised))
        client = RecordingClient()
        entries = self.evaluate(client, reviewer="fixture-reviewer", resume=True, retry_failed=True)
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all(call[2] for call in client.calls))
        self.assertTrue(all(entry["review"]["status"] == "success" for entry in entries if entry["status"] == "corrected"))

    def test_analysis_only_makes_no_generation_calls(self):
        entries = self.evaluate(None, analyze_only=True)
        self.assertEqual(sum(entry["status"] == "flagged" for entry in entries), 3)
        self.assertTrue(all(entry["revised_answer"] is None for entry in entries))

    def test_changed_threshold_refuses_to_mix_results(self):
        self.evaluate(RecordingClient())
        with self.assertRaisesRegex(ValueError, "settings changed"):
            self.evaluate(RecordingClient(), threshold=0.9, resume=True)

    def test_api_failure_leaves_too_few_peers_and_is_not_embedded(self):
        self.rows = self.rows[:3]
        self.rows[0].update(status="error", response="", error="failed")
        client = RecordingClient()
        entries = self.evaluate(client)
        self.assertEqual(client.calls, [])
        self.assertEqual(entries[0]["status"], "excluded_api_error")
        self.assertTrue(all(entry["reason"] == "insufficient_successful_models" for entry in entries[1:]))

    def test_two_equally_supported_clusters_abstain(self):
        class SplitEncoder:
            name = "test-split"
            def encode(self, texts):
                return [[1, 0] if text.startswith("A") else [0, 1] for text in texts]
        rows = [{"prompt": "question", "model": str(i), "model_id": str(i), "status": "success", "response": answer}
                for i, answer in enumerate(["A1", "A2", "B1", "B2"])]
        client = RecordingClient()
        entries = evaluate_and_correct(rows, client, SplitEncoder(), self.directory / "split")
        self.assertEqual(client.calls, [])
        self.assertTrue(all(entry["reason"] == "ambiguous_reference_groups" for entry in entries))

    def test_retrieval_uses_question_and_limits_references(self):
        class TrackingEncoder:
            def encode(self, texts):
                self.texts = texts
                return [{"question": [1, 0], "near": [1, 0], "far": [0, 1]}[text] for text in texts]
        encoder = TrackingEncoder()
        chosen = retrieve_references("question", [{"model": "far", "response": "far"},
                                                   {"model": "near", "response": "near"}], encoder, top_k=1)
        self.assertEqual(encoder.texts[0], "question")
        self.assertEqual([item["model"] for item in chosen], ["near"])
        with self.assertRaisesRegex(ValueError, "actual question"):
            retrieve_references("", [], encoder)

    def test_manual_entry_point_writes_actual_selected_models_and_rejects_self_reference(self):
        output = self.directory / "manual.csv"
        answer = run_rag_correction(CASES[0][0], "fixture_outlier", ["fixture_a", "fixture_b"],
                                    self.results, output, client=FixtureClient(), encoder=self.encoder, top_k=1)
        self.assertEqual(answer, CASES[0][1])
        self.assertIn('fixture_a', output.read_text())
        self.assertNotIn('fixture_b', output.read_text())
        with self.assertRaisesRegex(ValueError, "exclude the target"):
            run_rag_correction(CASES[0][0], "fixture_outlier", ["fixture_outlier"], self.results, output)

    def test_cli_demo_runs_without_dependencies_or_credentials_and_resumes(self):
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(root / "pipeline.py"), "--demo", "--output-dir", str(self.directory / "cli")]
        first = subprocess.run(command, cwd=self.directory, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("revised 3", first.stdout)
        second = subprocess.run(command + ["--resume"], cwd=self.directory, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stderr)


if __name__ == "__main__":
    unittest.main()
