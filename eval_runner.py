from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.config import CaseResult, PromptConfig, RunResult, TestCase
from src.llm_feature import classify_email
from src.scoring import score_summary_relevance

DEFAULT_CONCURRENCY = 8


def load_golden_dataset(path: str | Path) -> List[TestCase]:
    with open(path, "r") as f:
        data = json.load(f)
    return [TestCase(**c) for c in data["cases"]]


async def _run_one_case(
    case: TestCase, prompt_config: PromptConfig, semaphore: asyncio.Semaphore
) -> CaseResult:
    async with semaphore:
        output, latency_ms, prompt_tokens, completion_tokens = await classify_email(
            case.input_email, prompt_config
        )
        relevance = score_summary_relevance(case.expected_summary, output.summary)
        return CaseResult(
            case_id=case.id,
            input_email=case.input_email,
            expected_category=case.expected_category,
            predicted_category=output.category,
            expected_summary=case.expected_summary,
            predicted_summary=output.summary,
            category_match=output.category == case.expected_category,
            summary_relevance_score=relevance,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            difficulty=case.expected_difficulty,
        )


async def run_eval(
    prompt_config: PromptConfig,
    golden_dataset: List[TestCase],
    concurrency: int = DEFAULT_CONCURRENCY,
) -> RunResult:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_run_one_case(case, prompt_config, semaphore) for case in golden_dataset]
    results = await asyncio.gather(*tasks)

    return RunResult(
        run_id=str(uuid.uuid4())[:8],
        prompt_version=prompt_config.version_id,
        model=prompt_config.model,
        timestamp=datetime.now(timezone.utc),
        results=list(results),
    )


def save_run(run: RunResult, runs_dir: str | Path = "runs") -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(exist_ok=True)
    out_path = runs_dir / f"{run.timestamp.strftime('%Y%m%dT%H%M%S')}_{run.prompt_version}_{run.run_id}.json"
    with open(out_path, "w") as f:
        f.write(run.model_dump_json(indent=2))
    return out_path


def load_run(path: str | Path) -> RunResult:
    with open(path, "r") as f:
        return RunResult.model_validate_json(f.read())


def latest_runs(runs_dir: str | Path = "runs", n: int | None = None) -> List[Path]:
    """Return run files sorted oldest -> newest (filenames are timestamp-prefixed)."""
    runs_dir = Path(runs_dir)
    files = sorted(runs_dir.glob("*.json"))
    return files if n is None else files[-n:]
