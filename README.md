# LLM RAG Correction System

A Python script that uses Retrieval-Augmented Generation (RAG) to correct hallucinated or incorrect answers from LLM models by using verified correct answers from other models as context.

## Features

- Loads results from batch LLM runs
- Uses semantic similarity (sentence transformers) to retrieve relevant correct answers
- Creates RAG-enhanced prompts with verified context
- Corrects incorrect model responses using verified answers from other models
- Saves correction results to CSV

## Setup

1. Install dependencies:
```bash
pip install ollama pandas sentence-transformers scikit-learn
```

2. Set your Ollama API key as an environment variable:

**Windows PowerShell:**
```powershell
$env:OLLAMA_API_KEY="your-api-key-here"
```

**Windows CMD:**
```cmd
set OLLAMA_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OLLAMA_API_KEY="your-api-key-here"
```

## Usage

1. First, run your batch LLM runner to generate `results.csv` with model responses.

2. Run the RAG correction script:
```python
from RAG import run_rag_correction

run_rag_correction(
    prompt="What is the capital of France?",
    wrong_model="gemma",  # Model that gave incorrect answer
    correct_models=["gpt_oss_120b", "deepseek", "qwen3"],  # Models with correct answers
    results_csv="results.csv",
    output_csv="rag_corrections.csv"
)
```

Or run directly:
```bash
python RAG.py
```

## How It Works

1. **Load Results**: Reads the batch results CSV containing all model responses
2. **Retrieve Context**: Uses sentence transformers to find the most similar correct answers to the wrong answer
3. **Create RAG Prompt**: Builds an enhanced prompt that includes verified correct answers as context
4. **Correct Answer**: Sends the RAG-enhanced prompt back to the model that gave the wrong answer
5. **Save Results**: Stores the original and corrected answers in a CSV file

## Output

The script generates `rag_corrections.csv` with:
- `prompt`: The original question
- `wrong_model`: The model that gave the incorrect answer
- `original_answer`: The incorrect/hallucinated answer
- `corrected_answer`: The corrected answer using RAG
- `correct_models_used`: Which models' answers were used as reference
- `num_correct_references`: Number of correct answers used as context

## Supported Models

- `gpt_oss_120b`: GPT-OSS 120B model
- `deepseek`: DeepSeek V3.1 671B Cloud
- `qwen3`: Qwen3 Coder 480B Cloud
- `gemma`: Gemma3 4B Cloud

