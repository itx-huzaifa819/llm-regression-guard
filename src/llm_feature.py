from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Tuple

from src.config import ClassificationOutput, PromptConfig

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "mock")


async def classify_email(
    email_text: str, prompt_config: PromptConfig
) -> Tuple[ClassificationOutput, float, int, int]:
    """
    Run one email through the classifier feature.

    Returns (output, latency_ms, prompt_tokens, completion_tokens).
    """
    if LLM_PROVIDER == "openai":
        return await _classify_openai(email_text, prompt_config)
    return await _classify_mock(email_text, prompt_config)


async def _classify_openai(
    email_text: str, prompt_config: PromptConfig
) -> Tuple[ClassificationOutput, float, int, int]:
    """
    Real provider. Requires `pip install openai` and OPENAI_API_KEY set.
    Kept isolated here so it's the only place that needs editing to point
    at a different vendor (Azure OpenAI, Anthropic, a local model, etc).
    """
    from openai import AsyncOpenAI  # imported lazily so mock mode has no dependency

    client = AsyncOpenAI()
    messages = [{"role": "system", "content": prompt_config.system_prompt}]
    for ex in prompt_config.few_shot_examples:
        messages.append({"role": "user", "content": ex.input})
        messages.append({"role": "assistant", "content": json.dumps(ex.output)})
    messages.append({"role": "user", "content": email_text})

    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=prompt_config.model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    output = ClassificationOutput(**parsed)
    usage = response.usage
    return output, latency_ms, usage.prompt_tokens, usage.completion_tokens


# --- Mock provider -----------------------------------------------------
# Deterministic keyword-based classifier that behaves *worse* on
# prompts/email_classifier_v2.yaml than on v1, since v2 drops the
# disambiguation rules and most of the few-shot examples. This is what
# lets the pipeline demonstrate a genuine, reproducible regression.

_BILLING_KEYWORDS = {"charge", "charged", "refund", "invoice", "payment", "billed", "subscription", "price"}
_TECHNICAL_KEYWORDS = {"crash", "crashes", "bug", "error", "broken", "not working", "outage", "freeze", "fails"}
_ACCOUNT_KEYWORDS = {"password", "login", "log in", "account", "reset", "profile", "username", "locked out"}


def _keyword_score(text: str, keywords: set) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


async def _classify_mock(
    email_text: str, prompt_config: PromptConfig
) -> Tuple[ClassificationOutput, float, int, int]:
    await asyncio.sleep(0.01)  # simulate network latency
    start = time.perf_counter()

    scores = {
        "billing": _keyword_score(email_text, _BILLING_KEYWORDS),
        "technical": _keyword_score(email_text, _TECHNICAL_KEYWORDS),
        "account": _keyword_score(email_text, _ACCOUNT_KEYWORDS),
    }
    best_category, best_score = max(scores.items(), key=lambda kv: kv[1])
    category = best_category if best_score > 0 else "general"

    # v2 has no disambiguation rules and fewer few-shot examples, so it
    # loses precision on short / ambiguous / mixed-signal emails -- a
    # deterministic stand-in for "a weaker prompt does worse on edge cases".
    is_leaner_prompt = len(prompt_config.few_shot_examples) <= 2
    word_count = len(email_text.split())
    is_edge_case = word_count <= 6 or sum(1 for s in scores.values() if s > 0) >= 2

    rng = random.Random(f"{prompt_config.version_id}:{email_text}")
    if is_leaner_prompt and is_edge_case and rng.random() < 0.55:
        alt_categories = [c for c in ("billing", "technical", "account", "general") if c != category]
        category = rng.choice(alt_categories)

    if is_leaner_prompt:
        # v2 has no instruction to write a *tailored* one-sentence summary,
        # so the mock falls back to lazily truncating the raw email.
        summary = email_text.strip().split(".")[0][:90]
    else:
        summary = _template_summary(category, email_text)

    latency_ms = (time.perf_counter() - start) * 1000 + rng.uniform(80, 220)
    prompt_tokens = len(prompt_config.system_prompt.split()) + len(email_text.split())
    completion_tokens = len(summary.split()) + 5

    output = ClassificationOutput(category=category, summary=summary)
    return output, latency_ms, prompt_tokens, completion_tokens


def _template_summary(category: str, email_text: str) -> str:
    templates = {
        "billing": "Customer has a billing or payment concern regarding their account.",
        "technical": "Customer is reporting a technical issue or bug with the product.",
        "account": "Customer needs help accessing or managing their account.",
        "general": "Customer has a general question or comment.",
    }
    return templates[category]
