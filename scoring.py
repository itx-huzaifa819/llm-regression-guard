from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "and", "or",
    "for", "on", "in", "at", "their", "this", "that", "with", "customer",
    "customer's", "wants", "want",
}


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def score_summary_relevance(expected_summary: str, predicted_summary: str) -> int:
    """
    1-5 relevance score. In production, replace the body of this function
    with a real LLM-as-judge call, e.g.:

        prompt = JUDGE_PROMPT.format(expected=expected_summary, predicted=predicted_summary)
        response = await judge_client.chat.completions.create(...)
        return int(response.choices[0].message.content.strip())

    Keeping the signature identical means eval_runner.py never needs to
    change when you swap the scoring backend.
    """
    expected_tokens = _tokenize(expected_summary)
    predicted_tokens = _tokenize(predicted_summary)

    if not expected_tokens or not predicted_tokens:
        return 1

    overlap = len(expected_tokens & predicted_tokens)
    union = len(expected_tokens | predicted_tokens)
    jaccard = overlap / union if union else 0.0

    if jaccard >= 0.55:
        return 5
    if jaccard >= 0.35:
        return 4
    if jaccard >= 0.20:
        return 3
    if jaccard >= 0.08:
        return 2
    return 1


JUDGE_PROMPT = """You are grading a customer-support email summary.

Reference summary (ground truth): {expected}
Model-generated summary: {predicted}

On a scale of 1-5, how well does the model-generated summary capture the
same core issue as the reference summary? Respond with ONLY the integer.
5 = same issue, same key facts. 1 = unrelated or wrong issue.
"""
