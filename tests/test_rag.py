import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pandas as pd
from ollama import Client

import RAG


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "results.csv"
        self.output = Path(self.tmp.name) / "corrections.csv"
        self.question = "What is the capital of France?"
        pd.DataFrame([
            {"prompt": self.question, "model": "gemma", "response": "London"},
            {"prompt": self.question, "model": "gpt_oss_120b", "response": "Paris"},
            {"prompt": self.question, "model": "deepseek", "response": "The capital is Paris"},
        ]).to_csv(self.source, index=False)

    def run_correction(self, references=None):
        return RAG.run_rag_correction(
            self.question, "gemma",
            references if references is not None else ["gpt_oss_120b", "deepseek"],
            str(self.source), str(self.output))

    def test_cosine_ranking_uses_question_and_returns_top_k(self):
        encoder = Mock()
        encoder.encode.side_effect = [[[1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]]
        with patch.object(RAG, "encoder", encoder):
            result = RAG.retrieve_top_k_similar(self.question, ["unrelated", "relevant", "opposite"], top_k=1)
        self.assertEqual(result, "relevant")
        self.assertEqual(encoder.encode.call_args_list[0].args[0], [self.question])

    def test_full_original_workflow_and_append(self):
        encoder = Mock()
        encoder.encode.side_effect = [
            [[1.0, 0.0]], [[1.0, 0.0], [0.8, 0.2]],
            [[1.0, 0.0]], [[1.0, 0.0], [0.8, 0.2]],
        ]
        with patch.object(RAG, "encoder", encoder), patch.object(RAG, "call_ollama_cloud", return_value="Paris") as chat:
            self.assertEqual(self.run_correction(), "Paris")
            self.run_correction()
        prompt, model = chat.call_args.args
        self.assertIn(self.question, prompt)
        self.assertIn("The capital is Paris", prompt)
        self.assertNotIn("London", prompt)
        self.assertEqual(model, RAG.MODEL_MAP["gemma"])
        self.assertEqual(encoder.encode.call_args_list[0].args[0], [self.question])
        rows = pd.read_csv(self.output)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows.original_answer.tolist(), ["London", "London"])
        self.assertEqual(rows.num_correct_references.tolist(), [2, 2])

    def test_self_missing_and_failed_references_do_not_generate(self):
        rows = pd.read_csv(self.source)
        rows.loc[rows.model == "deepseek", "response"] = "OLLAMA_CLOUD_ERROR: unavailable"
        rows.to_csv(self.source, index=False)
        for references in [["gemma"], ["missing"], ["deepseek"], []]:
            with self.subTest(references=references), patch.object(RAG, "call_ollama_cloud") as chat:
                with self.assertRaises(ValueError):
                    self.run_correction(references)
                chat.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_regeneration_does_not_write_a_correction(self):
        with patch.object(RAG, "retrieve_top_k_similar", return_value="Paris"):
            with patch.object(RAG, "client") as client:
                client.chat.side_effect = RuntimeError("unavailable")
                with self.assertRaisesRegex(RuntimeError, "Ollama request failed"):
                    self.run_correction()
        self.assertFalse(self.output.exists())

    def test_duplicate_answers_and_missing_columns_are_rejected(self):
        rows = pd.read_csv(self.source)
        for invalid in [pd.concat([rows, rows.iloc[:1]]), rows.drop(columns=["response"])]:
            invalid.to_csv(self.source, index=False)
            with self.assertRaises(ValueError):
                RAG.load_results(str(self.source))

    def test_empty_and_failed_answers_are_not_references(self):
        rows = pd.DataFrame([
            {"prompt": "Q", "model": "a", "response": "OLLAMA_CLOUD_ERROR: x"},
            {"prompt": "Q", "model": "b", "response": ""},
            {"prompt": "Q", "model": "c", "response": "Good"},
        ])
        self.assertEqual(RAG.get_answers_for_prompt(rows, "Q"), {"c": "Good"})

    def test_missing_key_and_empty_query_are_clear_errors(self):
        with patch.object(RAG, "client", None), patch.dict(os.environ, {}, clear=True):
            with patch.object(RAG, "Client") as factory:
                with self.assertRaisesRegex(ValueError, "OLLAMA_API_KEY"):
                    RAG.call_ollama_cloud("Q", "M")
                factory.assert_not_called()
        with self.assertRaises(ValueError):
            RAG.retrieve_top_k_similar("", ["Answer"])

    def test_actual_sdk_response_and_failure(self):
        def respond(request):
            self.assertEqual(json.loads(request.content)["messages"][0]["content"], "Question")
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "Paris"}, "done": True})
        client = Client(host="https://example.invalid", transport=httpx.MockTransport(respond))
        with patch.object(RAG, "client", client):
            self.assertEqual(RAG.call_ollama_cloud("Question", "test"), "Paris")
        for response in [
            {"message": {"content": ""}},
            {"message": {"content": "Partial"}, "done_reason": "length"},
        ]:
            with patch.object(RAG, "client") as client:
                client.chat.return_value = response
                with self.assertRaises(RuntimeError):
                    RAG.call_ollama_cloud("Q", "M")


if __name__ == "__main__":
    unittest.main()
