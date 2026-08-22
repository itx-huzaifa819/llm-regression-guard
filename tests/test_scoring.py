from src.config import CaseResult, RunResult
from src.diff_engine import diff_runs
from src.scoring import score_summary_relevance
from datetime import datetime, timezone


def test_score_summary_relevance_identical_is_high():
    score = score_summary_relevance(
        "Customer was double-charged and wants a refund.",
        "Customer was double-charged and wants a refund.",
    )
    assert score == 5


def test_score_summary_relevance_unrelated_is_low():
    score = score_summary_relevance(
        "Customer was double-charged and wants a refund.",
        "The weather today is sunny with a light breeze.",
    )
    assert score <= 2


def _make_case_result(case_id, expected, predicted):
    match = expected == predicted
    return CaseResult(
        case_id=case_id,
        input_email="test email",
        expected_category=expected,
        predicted_category=predicted,
        expected_summary="s",
        predicted_summary="s",
        category_match=match,
        summary_relevance_score=5 if match else 2,
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=5,
        difficulty="easy",
    )


def _make_run(run_id, version, results):
    return RunResult(
        run_id=run_id,
        prompt_version=version,
        model="mock",
        timestamp=datetime.now(timezone.utc),
        results=results,
    )


def test_diff_detects_regression():
    baseline = _make_run(
        "base1",
        "v1",
        [
            _make_case_result("tc1", "billing", "billing"),
            _make_case_result("tc2", "technical", "technical"),
        ],
    )
    current = _make_run(
        "cur1",
        "v2",
        [
            _make_case_result("tc1", "billing", "general"),  # flipped pass -> fail
            _make_case_result("tc2", "technical", "technical"),
        ],
    )
    diff = diff_runs(baseline, current)
    assert len(diff.regressions) == 1
    assert diff.regressions[0].case_id == "tc1"
    assert diff.pass_rate_delta < 0


def test_diff_severity_thresholds():
    # 1 flip out of 2 cases = 50% drop, well past critical
    baseline = _make_run(
        "base1", "v1",
        [_make_case_result(f"tc{i}", "billing", "billing") for i in range(2)],
    )
    current = _make_run(
        "cur1", "v2",
        [
            _make_case_result("tc0", "billing", "general"),
            _make_case_result("tc1", "billing", "billing"),
        ],
    )
    diff = diff_runs(baseline, current, warning_threshold_pct=3.0, critical_threshold_pct=8.0)
    assert diff.severity == "critical"
