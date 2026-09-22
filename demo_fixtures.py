"""Synthetic fixtures for a network-free software demo, not research results."""
import json

from batch_llm_runner import CallResult


CASES = [
    ("What is the capital of France? Answer in one sentence.",
     "Paris is the capital of France.", "France's capital city is Paris.",
     "A bicycle has two wheels and is powered by pedaling."),
    ("What do plants use sunlight for during photosynthesis? Answer in one sentence.",
     "Plants use sunlight to turn water and carbon dioxide into sugars, releasing oxygen.",
     "Sunlight powers the conversion of carbon dioxide and water into sugar during photosynthesis.",
     "A piano typically has eighty-eight keys."),
    ("What does a computer's CPU do? Answer in one sentence.",
     "The CPU executes program instructions and performs calculations.",
     "A CPU processes instructions and carries out arithmetic and logical operations.",
     "The Pacific is the largest ocean on Earth."),
]
MODELS = {"fixture_a": "fixture/a", "fixture_b": "fixture/b", "fixture_outlier": "fixture/outlier"}


class FixtureEncoder:
    name = "synthetic-fixture-vectors-NOT-a-learned-embedding-model"
    revision = "1"

    def encode(self, texts):
        mapping = {}
        for index, (question, first, second, outlier) in enumerate(CASES):
            vector = [0.0] * len(CASES)
            vector[index] = 1.0
            for text in (question, first, second):
                mapping[text] = vector
            mapping[outlier] = [-value for value in vector]
        return [mapping[text] for text in texts]


class FixtureClient:
    def settings(self):
        return {"mode": "synthetic fixture responses", "version": 1}

    def generate(self, model, prompt, *, json_mode=False):
        if json_mode:
            return CallResult(json.dumps({"verdict": "uncertain", "explanation":
                "This is a synthetic reviewer fixture, not an actual LLM assessment."}), "success", attempts=1)
        for question, first, second, outlier in CASES:
            if prompt == question:
                answer = {"fixture/a": first, "fixture/b": second, "fixture/outlier": outlier}[model]
                return CallResult(answer, "success", attempts=1)
            if f"Original question:\n{question}\n" in prompt:
                return CallResult(first, "success", attempts=1)
        raise ValueError("The fixture client only supports the three bundled demo questions.")
