import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from batch_llm_runner import CallResult, OllamaClient, load_models, read_prompts, read_results, run_batch


class CountingClient:
    def __init__(self, fail_at=None, error=False):
        self.calls = []
        self.fail_at = fail_at
        self.error = error

    def settings(self):
        return {"mode": "test"}

    def generate(self, model, prompt):
        self.calls.append((model, prompt))
        if len(self.calls) == self.fail_at:
            raise KeyboardInterrupt()
        return CallResult("" if self.error else "answer", "error" if self.error else "success",
                          attempts=1, error="temporary failure" if self.error else "")


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.prompts = self.directory / "prompts.csv"
        self.prompts.write_text('prompt\n"Question one?"\n"Question two?"\n')
        self.output = self.directory / "results.csv"
        self.models = {"a": "model-a", "b": "model-b"}

    def run_batch(self, client, **kwargs):
        return run_batch(self.prompts, self.output, models=self.models, client=client, **kwargs)

    def test_interrupt_preserves_completed_response_and_resume_skips_it(self):
        client = CountingClient(fail_at=2)
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch(client)
        self.assertEqual(len(read_results(self.output)), 1)
        resumed = CountingClient()
        rows = self.run_batch(resumed, resume=True)
        self.assertEqual(len(resumed.calls), 3)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["request_id"] for row in rows}), 4)
        self.assertEqual(self.run_batch(CountingClient(), resume=True), read_results(self.output))

    def test_only_failed_records_are_retried_when_requested(self):
        rows = self.run_batch(CountingClient(error=True))
        self.assertTrue(all(row["status"] == "error" and not row["response"] for row in rows))
        skipped = CountingClient()
        self.run_batch(skipped, resume=True)
        self.assertEqual(skipped.calls, [])
        recovered = CountingClient()
        self.run_batch(recovered, resume=True, retry_failed=True)
        self.assertEqual(len(recovered.calls), 4)
        self.assertTrue(all(row["status"] == "success" for row in read_results(self.output)))

    def test_changed_inputs_cannot_silently_reuse_checkpoint(self):
        self.run_batch(CountingClient())
        original = self.output.read_bytes()
        self.prompts.write_text("prompt\nChanged question\n")
        with self.assertRaisesRegex(ValueError, "differ"):
            self.run_batch(CountingClient(), resume=True)
        self.assertEqual(original, self.output.read_bytes())

    def test_existing_output_is_not_overwritten_without_resume(self):
        self.run_batch(CountingClient())
        with self.assertRaisesRegex(ValueError, "already"):
            self.run_batch(CountingClient())

    def test_duplicate_questions_are_collected_once_and_ids_are_distinct(self):
        self.prompts.write_text("prompt\nRepeated question\nRepeated question\n")
        client = CountingClient()
        self.assertEqual(len(self.run_batch(client)), 2)
        self.assertEqual(len(client.calls), 2)

    def test_input_validation_and_same_model_aliases(self):
        self.prompts.write_text("question\nWrong column\n")
        with self.assertRaisesRegex(ValueError, "prompt"):
            read_prompts(self.prompts)
        self.prompts.write_text('prompt\n""\n')
        with self.assertRaisesRegex(ValueError, "Empty"):
            read_prompts(self.prompts)
        model_file = self.directory / "models.json"
        model_file.write_text('{"a": "same-model", "b": "same-model"}')
        with self.assertRaisesRegex(ValueError, "distinct"):
            load_models(model_file)

    def test_legacy_error_strings_are_excluded_and_duplicates_rejected(self):
        self.output.write_text("prompt,model,response,latency_sec\nQuestion,a,OLLAMA_CLOUD_ERROR: failed,1\n")
        self.assertEqual(read_results(self.output)[0]["status"], "error")
        self.output.write_text("prompt,model,response\nQuestion,a,first\nQuestion,a,second\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_results(self.output)

    def test_input_cannot_be_output(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            run_batch(self.prompts, self.prompts, models=self.models, client=CountingClient())


class HttpTests(unittest.TestCase):
    def transport(self, outcomes, recorded):
        def perform(request, timeout):
            recorded.append((request, timeout))
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return io.StringIO(json.dumps(outcome))
        return perform

    def client(self, outcomes, recorded, **kwargs):
        return OllamaClient(api_key="test-key-not-a-real-secret", transport=self.transport(outcomes, recorded), **kwargs)

    def test_http_payload_timeout_and_json_review(self):
        seen = []
        client = self.client([{"message": {"content": "answer"}}], seen, timeout=7, retries=0)
        result = client.generate("test-model", "test question", json_mode=True)
        self.assertEqual(result.status, "success")
        request, timeout = seen[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://ollama.com/api/chat")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key-not-a-real-secret")
        self.assertEqual(timeout, 7)
        self.assertEqual(payload["messages"][0]["content"], "test question")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], "json")

    def test_429_respects_retry_after_then_succeeds(self):
        seen, sleeps = [], []
        error = HTTPError("https://ollama.com/api/chat", 429, "Rate limited", {"Retry-After": "3"}, None)
        client = self.client([error, {"message": {"content": "recovered"}}], seen, sleeper=sleeps.append)
        result = client.generate("model", "question")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.response, "recovered")
        self.assertEqual(sleeps, [3])

    def test_permanent_errors_are_not_retried(self):
        seen = []
        error = HTTPError("https://ollama.com/api/chat", 401, "Unauthorized", {}, None)
        result = self.client([error], seen).generate("model", "question")
        self.assertEqual(result.status, "error")
        self.assertEqual(len(seen), 1)
        self.assertEqual(result.response, "")

    def test_network_failure_is_bounded_and_secret_is_redacted(self):
        seen = []
        error = URLError("test-key-not-a-real-secret")
        client = self.client([error, error, error], seen, retries=2, sleeper=lambda _: None)
        result = client.generate("model", "question")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.status, "error")
        self.assertNotIn("test-key-not-a-real-secret", result.error)

    def test_empty_or_truncated_outputs_do_not_become_answers(self):
        for payload in ({"message": {"content": " "}},
                        {"message": {"content": "partial"}, "done_reason": "length"},
                        {"message": {"content": "partial"}, "done": False}):
            with self.subTest(payload=payload):
                result = self.client([payload], []).generate("model", "question")
                self.assertEqual(result.status, "error")
                self.assertEqual(result.response, "")

    def test_model_preflight_uses_provider_ids(self):
        client = self.client([{"models": [{"name": "available"}]}], [])
        with self.assertRaisesRegex(ValueError, "not advertised"):
            client.preflight(["retired-cloud-id"])

    def test_missing_key_fails_before_any_cloud_call(self):
        seen = []
        with patch.dict(os.environ, {}, clear=True):
            client = OllamaClient(transport=self.transport([], seen))
            with self.assertRaisesRegex(ValueError, "OLLAMA_API_KEY"):
                client.generate("model", "question")
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
