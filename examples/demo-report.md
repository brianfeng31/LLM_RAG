# Peer-answer comparison and RAG report

Run type: **SYNTHETIC responses and reviewer; SYNTHETIC vectors; no LLM or learned embedding inference**

Agreement measures similarity, not factual accuracy. Reference answers are not verified ground truth.

Embedding model: `synthetic-fixture-vectors-NOT-a-learned-embedding-model`. Threshold: 0.85. Top-k: 3.

Before and after scores use the same original peer answers and exclude the target's original answer.

## fixture_a

Question:
```text
What is the capital of France? Answer in one sentence.
```
Status: **skipped**. Reason: already_in_reference_group.

Original answer:
```text
Paris is the capital of France.
```
Original mean peer agreement: **0.0000**

## fixture_b

Question:
```text
What is the capital of France? Answer in one sentence.
```
Status: **skipped**. Reason: already_in_reference_group.

Original answer:
```text
France's capital city is Paris.
```
Original mean peer agreement: **0.0000**

## fixture_outlier

Question:
```text
What is the capital of France? Answer in one sentence.
```
Status: **corrected**. Reason: —.

Original answer:
```text
A bicycle has two wheels and is powered by pedaling.
```
Original mean peer agreement: **-1.0000**

Revised answer:
```text
Paris is the capital of France.
```
Revised mean peer agreement: **1.0000**
Change: **+2.0000**

Reference: fixture_a (question similarity 1.0000)
```text
Paris is the capital of France.
```

Reference: fixture_b (question similarity 1.0000)
```text
France's capital city is Paris.
```

Optional LLM assessment (separate from the similarity metric):
```text
{
  "model_id": "fixture-reviewer",
  "response": "{\"verdict\": \"uncertain\", \"explanation\": \"This is a synthetic reviewer fixture, not an actual LLM assessment.\"}",
  "status": "success",
  "latency_sec": 0.0,
  "attempts": 1,
  "error": "",
  "verdict": "uncertain",
  "explanation": "This is a synthetic reviewer fixture, not an actual LLM assessment."
}
```

## fixture_a

Question:
```text
What do plants use sunlight for during photosynthesis? Answer in one sentence.
```
Status: **skipped**. Reason: already_in_reference_group.

Original answer:
```text
Plants use sunlight to turn water and carbon dioxide into sugars, releasing oxygen.
```
Original mean peer agreement: **0.0000**

## fixture_b

Question:
```text
What do plants use sunlight for during photosynthesis? Answer in one sentence.
```
Status: **skipped**. Reason: already_in_reference_group.

Original answer:
```text
Sunlight powers the conversion of carbon dioxide and water into sugar during photosynthesis.
```
Original mean peer agreement: **0.0000**

## fixture_outlier

Question:
```text
What do plants use sunlight for during photosynthesis? Answer in one sentence.
```
Status: **corrected**. Reason: —.

Original answer:
```text
A piano typically has eighty-eight keys.
```
Original mean peer agreement: **-1.0000**

Revised answer:
```text
Plants use sunlight to turn water and carbon dioxide into sugars, releasing oxygen.
```
Revised mean peer agreement: **1.0000**
Change: **+2.0000**

Reference: fixture_a (question similarity 1.0000)
```text
Plants use sunlight to turn water and carbon dioxide into sugars, releasing oxygen.
```

Reference: fixture_b (question similarity 1.0000)
```text
Sunlight powers the conversion of carbon dioxide and water into sugar during photosynthesis.
```

Optional LLM assessment (separate from the similarity metric):
```text
{
  "model_id": "fixture-reviewer",
  "response": "{\"verdict\": \"uncertain\", \"explanation\": \"This is a synthetic reviewer fixture, not an actual LLM assessment.\"}",
  "status": "success",
  "latency_sec": 0.0,
  "attempts": 1,
  "error": "",
  "verdict": "uncertain",
  "explanation": "This is a synthetic reviewer fixture, not an actual LLM assessment."
}
```

## fixture_a

Question:
```text
What does a computer's CPU do? Answer in one sentence.
```
Status: **skipped**. Reason: already_in_reference_group.

Original answer:
```text
The CPU executes program instructions and performs calculations.
```
Original mean peer agreement: **0.0000**

## fixture_b

Question:
```text
What does a computer's CPU do? Answer in one sentence.
```
Status: **skipped**. Reason: already_in_reference_group.

Original answer:
```text
A CPU processes instructions and carries out arithmetic and logical operations.
```
Original mean peer agreement: **0.0000**

## fixture_outlier

Question:
```text
What does a computer's CPU do? Answer in one sentence.
```
Status: **corrected**. Reason: —.

Original answer:
```text
The Pacific is the largest ocean on Earth.
```
Original mean peer agreement: **-1.0000**

Revised answer:
```text
The CPU executes program instructions and performs calculations.
```
Revised mean peer agreement: **1.0000**
Change: **+2.0000**

Reference: fixture_a (question similarity 1.0000)
```text
The CPU executes program instructions and performs calculations.
```

Reference: fixture_b (question similarity 1.0000)
```text
A CPU processes instructions and carries out arithmetic and logical operations.
```

Optional LLM assessment (separate from the similarity metric):
```text
{
  "model_id": "fixture-reviewer",
  "response": "{\"verdict\": \"uncertain\", \"explanation\": \"This is a synthetic reviewer fixture, not an actual LLM assessment.\"}",
  "status": "success",
  "latency_sec": 0.0,
  "attempts": 1,
  "error": "",
  "verdict": "uncertain",
  "explanation": "This is a synthetic reviewer fixture, not an actual LLM assessment."
}
```
