from __future__ import annotations

import os

import httpx

from src.diff_engine import DiffReport

SEVERITY_EMOJI = {"ok": "✅", "warning": "⚠️", "critical": "🚨"}


def build_slack_payload(diff: DiffReport, report_url: str = "") -> dict:
    emoji = SEVERITY_EMOJI[diff.severity]
    status_word = {"ok": "PASS", "warning": "WARN", "critical": "FAIL"}[diff.severity]

    regressions_line = (
        f"{len(diff.regressions)} regression(s) detected"
        if diff.regressions
        else "No regressions"
    )
    accuracy_line = (
        f"accuracy {diff.baseline_pass_rate:.0%} -> {diff.current_pass_rate:.0%} "
        f"({diff.pass_rate_delta:+.1%})"
    )

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{emoji} *Eval {status_word}* — "
                    f"`{diff.baseline_prompt_version}` -> `{diff.current_prompt_version}`\n"
                    f"{regressions_line}, {accuracy_line}"
                ),
            },
        }
    ]

    if diff.regressions:
        sample = ", ".join(r.case_id for r in diff.regressions[:5])
        more = f" (+{len(diff.regressions) - 5} more)" if len(diff.regressions) > 5 else ""
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Flipped to fail:* {sample}{more}"},
            }
        )

    if report_url:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"<{report_url}|View full diff report>"},
            }
        )

    fallback_text = f"{emoji} Eval {status_word}: {regressions_line}, {accuracy_line}"
    return {"text": fallback_text, "blocks": blocks}


def send_slack_alert(diff: DiffReport, report_url: str = "", webhook_url: str | None = None) -> bool:
    """Returns True if the alert was sent (or would be, in dry-run mode)."""
    webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    payload = build_slack_payload(diff, report_url)

    if not webhook_url:
        print("[alerts] SLACK_WEBHOOK_URL not set — dry run, printing payload instead:")
        print(payload["text"])
        return False

    response = httpx.post(webhook_url, json=payload, timeout=10)
    response.raise_for_status()
    return True
