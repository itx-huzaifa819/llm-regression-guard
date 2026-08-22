from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.config import RunResult
from src.diff_engine import DiffReport

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eval Diff Report — {baseline_version} vs {current_version}</title>
<style>
  :root {{
    --ok: #1a7f37; --warn: #9a6700; --crit: #cf222e;
    --bg: #f6f8fa; --border: #d0d7de; --text: #1f2328;
  }}
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 0; padding: 32px; background: #fff; color: var(--text); }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  .meta {{ color: #57606a; font-size: 13px; margin-bottom: 24px; }}
  .scorecard {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
  .card {{ border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; background: var(--bg); min-width: 160px; }}
  .card .label {{ font-size: 12px; color: #57606a; text-transform: uppercase; letter-spacing: 0.03em; }}
  .card .value {{ font-size: 24px; font-weight: 600; margin-top: 4px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; color: #fff; }}
  .badge.ok {{ background: var(--ok); }} .badge.warning {{ background: var(--warn); }} .badge.critical {{ background: var(--crit); }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; font-size: 13px; }}
  th, td {{ border: 1px solid var(--border); padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: var(--bg); font-size: 12px; text-transform: uppercase; letter-spacing: 0.02em; }}
  tr.regression {{ background: #ffebe9; }}
  tr.improvement {{ background: #e6ffec; }}
  .old {{ color: var(--crit); }} .new {{ color: var(--ok); }}
  section {{ margin-bottom: 32px; }}
  h2 {{ font-size: 16px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  canvas {{ max-width: 700px; }}
  code {{ background: var(--bg); padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>

<h1>Eval Diff Report</h1>
<div class="meta">
  <code>{baseline_version}</code> (baseline, run {baseline_run_id}) &rarr;
  <code>{current_version}</code> (current, run {current_run_id})
  &nbsp;|&nbsp; model: {model} &nbsp;|&nbsp; generated: {generated_at}
  &nbsp;|&nbsp; <span class="badge {severity}">{severity_label}</span>
</div>

<div class="scorecard">
  <div class="card">
    <div class="label">Pass rate (baseline)</div>
    <div class="value">{baseline_pass_rate:.0%}</div>
  </div>
  <div class="card">
    <div class="label">Pass rate (current)</div>
    <div class="value">{current_pass_rate:.0%}</div>
  </div>
  <div class="card">
    <div class="label">Delta</div>
    <div class="value">{pass_rate_delta:+.1%}</div>
  </div>
  <div class="card">
    <div class="label">Regressions</div>
    <div class="value">{num_regressions}</div>
  </div>
  <div class="card">
    <div class="label">Improvements</div>
    <div class="value">{num_improvements}</div>
  </div>
  <div class="card">
    <div class="label">Avg summary score</div>
    <div class="value">{baseline_summary_score:.1f} &rarr; {current_summary_score:.1f}</div>
  </div>
</div>

<section>
  <h2>Per-category accuracy</h2>
  <table>
    <tr><th>Category</th><th>Baseline</th><th>Current</th><th>Delta</th></tr>
    {category_rows}
  </table>
</section>

<section>
  <h2>Regressed cases ({num_regressions})</h2>
  {regression_table}
</section>

<section>
  <h2>Improved cases ({num_improvements})</h2>
  {improvement_table}
</section>

<section>
  <h2>Trend — last {num_trend_runs} runs</h2>
  <canvas id="trendChart" width="700" height="260"></canvas>
</section>

<script>
  const trendData = {trend_json};
  const canvas = document.getElementById('trendChart');
  const ctx = canvas.getContext('2d');
  function drawChart() {{
    const w = canvas.width, h = canvas.height, pad = 40;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#d0d7de'; ctx.beginPath();
    ctx.moveTo(pad, pad); ctx.lineTo(pad, h - pad); ctx.lineTo(w - pad, h - pad); ctx.stroke();
    ctx.fillStyle = '#57606a'; ctx.font = '11px sans-serif';
    ctx.fillText('100%', 4, pad + 4); ctx.fillText('0%', 10, h - pad + 4);
    if (trendData.length === 0) return;
    const stepX = (w - 2 * pad) / Math.max(trendData.length - 1, 1);
    ctx.strokeStyle = '#1a7f37'; ctx.lineWidth = 2; ctx.beginPath();
    trendData.forEach((pt, i) => {{
      const x = pad + i * stepX;
      const y = h - pad - pt.pass_rate * (h - 2 * pad);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
    ctx.fillStyle = '#1a7f37';
    trendData.forEach((pt, i) => {{
      const x = pad + i * stepX;
      const y = h - pad - pt.pass_rate * (h - 2 * pad);
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#57606a'; ctx.font = '10px sans-serif';
      ctx.save(); ctx.translate(x, h - pad + 16); ctx.rotate(-0.5);
      ctx.fillText(pt.label, 0, 0); ctx.restore();
      ctx.fillStyle = '#1a7f37';
    }});
  }}
  drawChart();
</script>

</body>
</html>
"""

_CASE_ROW = """<tr class="{cls}">
  <td>{case_id}</td>
  <td>{difficulty}</td>
  <td>{input_email}</td>
  <td>{expected_category}</td>
  <td class="old">{baseline_predicted}</td>
  <td class="new">{current_predicted}</td>
</tr>"""


def _case_table(cases, cls: str) -> str:
    if not cases:
        return "<p>None.</p>"
    rows = "\n".join(
        _CASE_ROW.format(
            cls=cls,
            case_id=c.case_id,
            difficulty=c.difficulty,
            input_email=c.input_email,
            expected_category=c.expected_category,
            baseline_predicted=c.baseline_predicted,
            current_predicted=c.current_predicted,
        )
        for c in cases
    )
    return (
        "<table><tr><th>Case</th><th>Difficulty</th><th>Input</th>"
        "<th>Expected</th><th>Baseline predicted</th><th>Current predicted</th></tr>"
        f"{rows}</table>"
    )


def generate_html_report(
    diff: DiffReport,
    current_run: RunResult,
    trend_runs: Optional[List[RunResult]] = None,
    out_path: str | Path = "report.html",
) -> Path:
    trend_runs = trend_runs or [current_run]
    trend_points = [
        {"label": f"{r.prompt_version[:12]}", "pass_rate": r.pass_rate} for r in trend_runs
    ]

    category_rows = "\n".join(
        f"<tr><td>{d.category}</td><td>{d.baseline_accuracy:.0%}</td>"
        f"<td>{d.current_accuracy:.0%}</td><td>{d.delta:+.1%}</td></tr>"
        for d in diff.category_deltas
    )

    severity_label = {"ok": "PASS", "warning": "WARNING", "critical": "CRITICAL"}[diff.severity]

    html = _TEMPLATE.format(
        baseline_version=diff.baseline_prompt_version,
        current_version=diff.current_prompt_version,
        baseline_run_id=diff.baseline_run_id,
        current_run_id=diff.current_run_id,
        model=current_run.model,
        generated_at=current_run.timestamp.isoformat(),
        severity=diff.severity,
        severity_label=severity_label,
        baseline_pass_rate=diff.baseline_pass_rate,
        current_pass_rate=diff.current_pass_rate,
        pass_rate_delta=diff.pass_rate_delta,
        num_regressions=len(diff.regressions),
        num_improvements=len(diff.improvements),
        baseline_summary_score=diff.baseline_avg_summary_score,
        current_summary_score=diff.current_avg_summary_score,
        category_rows=category_rows,
        regression_table=_case_table(diff.regressions, "regression"),
        improvement_table=_case_table(diff.improvements, "improvement"),
        num_trend_runs=len(trend_points),
        trend_json=json.dumps(trend_points),
    )

    out_path = Path(out_path)
    out_path.write_text(html)
    return out_path
