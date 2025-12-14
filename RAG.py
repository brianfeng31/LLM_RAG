import os
import pandas as pd
from ollama import Client
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIGURATION
# -----------------------------

# Get API key from environment variable
# Set it with: export OLLAMA_API_KEY="your-key-here" (Linux/Mac)
# Or: $env:OLLAMA_API_KEY="your-key-here" (Windows PowerShell)
API_KEY = os.getenv("OLLAMA_API_KEY", "")

if not API_KEY:
    raise ValueError("OLLAMA_API_KEY environment variable is not set. Please set it before running.")

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + API_KEY}
)

encoder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# -----------------------------
# MODEL MAPPING
# -----------------------------

MODEL_MAP = {
    "gpt_oss_120b": "gpt-oss:120b",
    "deepseek": "deepseek-v3.1:671b-cloud",
    "qwen3": "qwen3-coder:480b-cloud",
    "gemma": "gemma3:4b-cloud",
}

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def call_ollama_cloud(prompt, model):
    try:
        response = client.chat(
            model,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response["message"]["content"]
    except Exception as e:
        return f"OLLAMA_CLOUD_ERROR: {e}"


def load_results(results_csv="results.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_csv = os.path.join(script_dir, results_csv)
    return pd.read_csv(results_csv)


def get_answers_for_prompt(df, prompt):
    rows = df[df["prompt"] == prompt]
    return {row["model"]: row["response"] for _, row in rows.iterrows()}


def retrieve_top_k_similar(query_answer, correct_answers, top_k=3):
    if not correct_answers:
        return ""
    
    query_emb = encoder.encode([query_answer], convert_to_tensor=False)
    corpus_emb = encoder.encode(correct_answers, convert_to_tensor=False)
    
    sims = cosine_similarity(query_emb, corpus_emb)[0]
    top_indices = sims.argsort()[::-1][:min(top_k, len(correct_answers))]
    
    return "\n\n---\n\n".join([correct_answers[i] for i in top_indices])


def create_rag_prompt(original_prompt, context):
    rag_prompt = f"""You previously answered a question, but your answer may have contained errors or hallucinations.

Here are verified correct answers from other models for reference:

{context}

---

Now, based on these verified answers, please provide an accurate response to the original question:

{original_prompt}

Provide a corrected, factual answer that aligns with the verified information above."""
    
    return rag_prompt


def correct_model_answer(wrong_model_name, original_prompt, correct_answers):
    context = retrieve_top_k_similar("", correct_answers, top_k=len(correct_answers))
    rag_prompt = create_rag_prompt(original_prompt, context)
    ollama_model = MODEL_MAP[wrong_model_name]
    
    print(f"\nSending RAG-enhanced prompt to {wrong_model_name}...")
    corrected_answer = call_ollama_cloud(rag_prompt, ollama_model)
    
    return corrected_answer


# -----------------------------
# MAIN CORRECTION WORKFLOW
# -----------------------------

def run_rag_correction(
    prompt,
    wrong_model,
    correct_models,
    results_csv="results.csv",
    output_csv="rag_corrections.csv"
):
    df = load_results(results_csv)
    all_answers = get_answers_for_prompt(df, prompt)
    correct_answers = [all_answers[model] for model in correct_models if model in all_answers]
    
    if not correct_answers:
        print("No correct answers found!")
        return
    
    print(f"\nOriginal wrong answer from {wrong_model}:")
    print(all_answers.get(wrong_model, "N/A")[:200] + "...")
    
    print(f"\nUsing {len(correct_answers)} correct answers as RAG context")
    
    corrected = correct_model_answer(wrong_model, prompt, correct_answers)
    
    print(f"\nCorrected answer from {wrong_model}:")
    print(corrected[:200] + "...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(script_dir, output_csv)
    
    correction_result = {
        "prompt": prompt,
        "wrong_model": wrong_model,
        "original_answer": all_answers.get(wrong_model, "N/A"),
        "corrected_answer": corrected,
        "correct_models_used": ", ".join(correct_models),
        "num_correct_references": len(correct_answers)
    }
    
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        updated = pd.concat([existing, pd.DataFrame([correction_result])], ignore_index=True)
        updated.to_csv(output_csv, index=False)
    else:
        pd.DataFrame([correction_result]).to_csv(output_csv, index=False)
    
    print(f"\nSaved correction to {output_csv}")
    
    return corrected


# -----------------------------
# USAGE
# -----------------------------

if __name__ == "__main__":
    
    problem_prompt = "What is the capital of France?"
    
    hallucinated_model = "gemma"
    
    correct_models = ["gpt_oss_120b", "deepseek", "qwen3"]
    
    run_rag_correction(
        prompt=problem_prompt,
        wrong_model=hallucinated_model,
        correct_models=correct_models,
        results_csv="results.csv",
        output_csv="rag_corrections.csv"
    )
    
    print("\nRAG correction complete")
    print("Check rag_corrections.csv for before/after comparison")