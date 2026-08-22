from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel

from src.config import CaseResult, RunResult

Severity = Literal["ok", "warning", "critical"]


class FlippedCase(BaseModel):
    case_id: str
    input_email: str
    expected_category: str
    baseline_predicted: str
    current_predicted: str
    difficulty: str


class CategoryDelta(BaseModel):
    category: str
    baseline_accuracy: float
    current_accuracy: float
    delta: float


class DiffReport(BaseModel):
    baseline_run_id: str
    current_run_id: str
    baseline_prompt_version: str
    current_prompt_version: str
    baseline_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float
    baseline_avg_summary_score: float
    current_avg_summary_score: float
    category_deltas: List[CategoryDelta]
    regressions: List[FlippedCase]
    improvements: List[FlippedCase]
    severity: Severity


def diff_runs(
    baseline: RunResult,
    current: RunResult,
    warning_threshold_pct: float = 3.0,
    critical_threshold_pct: float = 8.0,
) -> DiffReport:
    baseline_by_id: Dict[str, CaseResult] = {r.case_id: r for r in baseline.results}
    current_by_id: Dict[str, CaseResult] = {r.case_id: r for r in current.results}
    shared_ids = set(baseline_by_id) & set(current_by_id)

    regressions: List[FlippedCase] = []
    improvements: List[FlippedCase] = []

    for case_id in sorted(shared_ids):
        base_r = baseline_by_id[case_id]
        cur_r = current_by_id[case_id]
        if base_r.category_match and not cur_r.category_match:
            regressions.append(
                FlippedCase(
                    case_id=case_id,
                    input_email=cur_r.input_email,
                    expected_category=cur_r.expected_category,
                    baseline_predicted=base_r.predicted_category,
                    current_predicted=cur_r.predicted_category,
                    difficulty=cur_r.difficulty,
                )
            )
        elif not base_r.category_match and cur_r.category_match:
            improvements.append(
                FlippedCase(
                    case_id=case_id,
                    input_email=cur_r.input_email,
                    expected_category=cur_r.expected_category,
                    baseline_predicted=base_r.predicted_category,
                    current_predicted=cur_r.predicted_category,
                    difficulty=cur_r.difficulty,
                )
            )

    category_deltas = _category_deltas(baseline.results, current.results)

    pass_rate_delta_pct = (current.pass_rate - baseline.pass_rate) * 100
    severity = _classify_severity(pass_rate_delta_pct, warning_threshold_pct, critical_threshold_pct)

    return DiffReport(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        baseline_prompt_version=baseline.prompt_version,
        current_prompt_version=current.prompt_version,
        baseline_pass_rate=baseline.pass_rate,
        current_pass_rate=current.pass_rate,
        pass_rate_delta=current.pass_rate - baseline.pass_rate,
        baseline_avg_summary_score=baseline.avg_summary_score,
        current_avg_summary_score=current.avg_summary_score,
        category_deltas=category_deltas,
        regressions=regressions,
        improvements=improvements,
        severity=severity,
    )


def _category_deltas(
    baseline_results: List[CaseResult], current_results: List[CaseResult]
) -> List[CategoryDelta]:
    categories = sorted({r.expected_category for r in baseline_results + current_results})
    deltas = []
    for cat in categories:
        base_cat = [r for r in baseline_results if r.expected_category == cat]
        cur_cat = [r for r in current_results if r.expected_category == cat]
        base_acc = sum(r.category_match for r in base_cat) / len(base_cat) if base_cat else 0.0
        cur_acc = sum(r.category_match for r in cur_cat) / len(cur_cat) if cur_cat else 0.0
        deltas.append(
            CategoryDelta(category=cat, baseline_accuracy=base_acc, current_accuracy=cur_acc, delta=cur_acc - base_acc)
        )
    return deltas


def _classify_severity(delta_pct: float, warning_threshold: float, critical_threshold: float) -> Severity:
    drop = -delta_pct  # a negative delta (regression) is a positive "drop"
    if drop >= critical_threshold:
        return "critical"
    if drop >= warning_threshold:
        return "warning"
    return "ok"
