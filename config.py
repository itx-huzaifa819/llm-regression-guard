from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account", "general"]


class FewShotExample(BaseModel):
    input: str
    output: dict


class PromptConfig(BaseModel):
    """A single versioned prompt definition, loaded from /prompts/*.yaml."""

    version_id: str
    timestamp: datetime
    model: str
    description: str = ""
    system_prompt: str
    few_shot_examples: List[FewShotExample] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ClassificationOutput(BaseModel):
    """Structured output contract for the LLM feature under test."""

    category: Category
    summary: str = Field(..., max_length=300)


class TestCase(BaseModel):
    """One golden-dataset record: a hand-verified input/expected pair."""

    id: str
    input_email: str
    expected_category: Category
    expected_summary: str
    expected_difficulty: Literal["easy", "medium", "hard"] = "easy"
    notes: str = ""


class CaseResult(BaseModel):
    """The outcome of running one TestCase through one PromptConfig."""

    case_id: str
    input_email: str
    expected_category: Category
    predicted_category: Category
    expected_summary: str
    predicted_summary: str
    category_match: bool
    summary_relevance_score: int  # 1-5, LLM-as-judge
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    difficulty: str


class RunResult(BaseModel):
    """The full output of one eval run — this is what gets diffed and stored."""

    run_id: str
    prompt_version: str
    model: str
    timestamp: datetime
    results: List[CaseResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.category_match for r in self.results) / len(self.results)

    @property
    def avg_summary_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.summary_relevance_score for r in self.results) / len(self.results)
