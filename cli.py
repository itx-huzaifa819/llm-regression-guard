from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from src.config import PromptConfig
from src.diff_engine import diff_runs
from src.drift import compute_drift, drift_alert_text
from src.eval_runner import latest_runs, load_golden_dataset, load_run, run_eval, save_run
from src.report import generate_html_report
from src.alerts import send_slack_alert

GOLDEN_DATASET_PATH = "golden_dataset/test_cases_v1.json"


def _run_and_save(prompt_path: str) -> Path:
    prompt_config = PromptConfig.from_yaml(prompt_path)
    golden_dataset = load_golden_dataset(GOLDEN_DATASET_PATH)
    result = asyncio.run(run_eval(prompt_config, golden_dataset))
    path = save_run(result)
    print(f"Ran {len(result.results)} cases with prompt '{prompt_config.version_id}'")
    print(f"Pass rate: {result.pass_rate:.1%} | Avg summary score: {result.avg_summary_score:.2f}")
    print(f"Saved: {path}")
    return path


def _run(args):
    _run_and_save(args.prompt)
    return 0


def _eval(args):
    """Full pipeline: run -> diff against baseline -> report -> alert -> exit code."""
    current_path = _run_and_save(args.prompt)
    current = load_run(current_path)

    if args.baseline:
        baseline_path = Path(args.baseline)
    else:
        prior_runs = [f for f in latest_runs("runs") if f != current_path]
        if not prior_runs:
            print("No baseline run found — nothing to diff against. Treating as pass.")
            return 0
        baseline_path = prior_runs[-1]

    baseline = load_run(baseline_path)
    diff = diff_runs(
        baseline,
        current,
        warning_threshold_pct=args.warning_threshold,
        critical_threshold_pct=args.critical_threshold,
    )

    trend_run_paths = latest_runs("runs", n=10)
    trend_runs = [load_run(p) for p in trend_run_paths]
    report_path = generate_html_report(diff, current, trend_runs=trend_runs, out_path="report.html")
    print(f"Diff report: {report_path}")

    print(f"Severity: {diff.severity.upper()}")
    print(f"Pass rate: {diff.baseline_pass_rate:.1%} -> {diff.current_pass_rate:.1%} ({diff.pass_rate_delta:+.1%})")
    print(f"Regressions: {len(diff.regressions)} | Improvements: {len(diff.improvements)}")
    for r in diff.regressions:
        print(f"  REGRESSED {r.case_id} [{r.difficulty}]: {r.baseline_predicted} -> {r.current_predicted}")

    if args.alert:
        send_slack_alert(diff, report_url=args.report_url or str(report_path))

    drift = compute_drift("runs")
    if drift and drift.is_drifting:
        print(drift_alert_text(drift))
        if args.alert:
            from src.alerts import build_slack_payload
            import httpx

            webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
            if webhook_url:
                httpx.post(webhook_url, json={"text": drift_alert_text(drift)}, timeout=10)

    if diff.severity == "critical" and args.fail_on_critical:
        print("::error::Critical regression threshold exceeded — blocking merge.")
        return 1
    return 0


def _drift(args):
    drift = compute_drift("runs", window=args.window, threshold=args.threshold)
    if drift is None:
        print("Not enough runs yet to compute drift (need at least 2).")
        return 0
    print(f"Rolling average ({drift.window_size} runs): {drift.rolling_average:.1%}")
    if drift.is_drifting:
        print(drift_alert_text(drift))
    else:
        print("No drift detected.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="LLM eval pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the golden dataset through a prompt version and save the result.")
    p_run.add_argument("--prompt", required=True)
    p_run.set_defaults(func=_run)

    p_eval = sub.add_parser("eval", help="Run + diff against baseline + report + alert + exit code for CI.")
    p_eval.add_argument("--prompt", required=True)
    p_eval.add_argument("--baseline", default=None, help="Path to a specific baseline run JSON. Defaults to the most recent prior run.")
    p_eval.add_argument("--warning-threshold", type=float, default=3.0)
    p_eval.add_argument("--critical-threshold", type=float, default=8.0)
    p_eval.add_argument("--alert", action="store_true", help="Send a Slack alert.")
    p_eval.add_argument("--report-url", default=None, help="Public URL to link in the Slack alert.")
    p_eval.add_argument("--fail-on-critical", action="store_true", help="Exit non-zero on critical regressions, to block CI merge.")
    p_eval.set_defaults(func=_eval)

    p_drift = sub.add_parser("drift", help="Check the rolling-average drift signal.")
    p_drift.add_argument("--window", type=int, default=7)
    p_drift.add_argument("--threshold", type=float, default=0.90)
    p_drift.set_defaults(func=_drift)

    args = parser.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
