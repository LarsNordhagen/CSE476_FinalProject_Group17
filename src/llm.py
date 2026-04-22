# %% Minimal setup
# If needed (uncomment in a notebook):
# !pip install requests python-dotenv

import os
import requests
from dotenv import load_dotenv
import re

load_dotenv()

API_KEY  = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")  
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507") 

def run_agent (question: str, past_feedback: str) -> str:

    subtasks = decompose(question)

    subanswers = []

    for subtask in subtasks:

        category = classify(subtask)

        context = generate_synthetic_context(subtask)
        
        answer_candidates = []
        self_consistency_count = 3
        for i in range(self_consistency_count):
            answer_candidate = use_cot_and_context_injection(subtask, context)
            answer_candidates.append(answer_candidate)
        
        best_candidate = choose_best(answer_candidates)
        subanswers.append(best_candidate)
        print(best_candidate)
    
    full_answer = combine_subanswers(subanswers)

    judge_feedback = llm_judge(question, full_answer)

    refined_answer = self_refine(full_answer, judge_feedback)

    return (refined_answer)[:4900]




# Break up the question into a list of subtasks
def decompose(question: str):
    return [question]

# Determine what type of task this is (e.g. reasoning, math, retrievel, etc.)
def classify(subtask: str):
    return "temp_classification"

# Ask the LLM to provide any context that could be relevent to answering the question
def generate_synthetic_context(subtask: str):
    return "no context"

# Get the answer using CoT reasoning. Use synthetic context to help
def use_cot_and_context_injection(subtask: str, context: str):
    prompt = f"""You are solving a question carefully.
    Question: {subtask}
    Helpful context:
    {context}
    
   Think carefully step by step about the question, using the helpful context when relevant. Return only the final answer. Do not include any explanation or reasoning."""

    result = call_model_chat_completions(
        prompt=prompt,
        system = "You are a helpful reasoning assistant. Think carefully internally, but ouput only the final answer with no explanation."
    )

    if not result["ok"]:
        print("Model call failed")
        
        print("status:", result["status"])
        print("error:", result["error"])
        return "ERROR: model call failed"


    if not result["text"]:
        return "ERROR: empty model response"


    return result["text"].strip()

    #answer = call_model_chat_completions(subtask)['text']
    #return answer

# Choose the best answer as part of self-consistency
def choose_best(candidates: list[str]):
    if not candidates:
        return "ERROR: no candidates"
    
    if len(candidates) == 1:
        return candidates[0]
    
    vote_counts = {}
    for candidate in candidates:
        vote_counts[candidate] = vote_counts.get(candidate, 0) + 1
    best_candidate = max(vote_counts, key=vote_counts.get)
    best_votes = vote_counts[best_candidate]
    if list(vote_counts.values()).count(best_votes) == 1:
        return best_candidate
        
    # tie break
    tied_candidates = [c for c, ct in vote_counts.items() if ct == best_votes]

    numbered_candidates = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(tied_candidates)
    )
    tie_break_prompt = f"""Select the single best final answer to the same question from the options below. Return only the number of the best option (e.g., 2), with no explanation. 
    
Options: 
{numbered_candidates}"""
    result = call_model_chat_completions(
        prompt=tie_break_prompt,
        system="You are an answer quality judge. Return only one option number.",
        temperature=0.0,
    )

    if result["ok"] and result["text"]:
        match = re.search(r"\d+", result["text"].strip())
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(tied_candidates):
                return tied_candidates[idx]
    return tied_candidates[0]

# Combine all subanswers into a coherent, full answer
def combine_subanswers(subanswers: list[str]):
    return subanswers[0]

# Provides feedback if the answer isn't accurate to the question
def llm_judge(question: str, answer: str):
    prompt = f"""Evaluate whether the answer correctly and sufficiently addresses the question.
Question:
{question}

Answer:
{answer}

Return exactly one of the following:
1) If the answer is correct and sufficient, return exactly: no feedback
2) Otherwise, return brief corrective feedback (1-3 sentences) focused on what is wrong or missing."""

    result = call_model_chat_completions(
        prompt=prompt,
        system="You are a strict but fair answer-quality judge. Follow the output format exactly.",
        temperature=0.0,
    )

    if not result["ok"]:
        return "Feedback unavailable: model call failed."

    if not result["text"]:
        return "Feedback unavailable: empty model response."

    feedback = result["text"].strip()
    if "no feedback" in feedback.lower():
        return "no feedback"
    return feedback

# Use self-refine
def self_refine(full_answer: str, judge_feedback: str):
    return full_answer







def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.15,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}