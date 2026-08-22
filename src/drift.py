from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from src.config import RunResult
from src.eval_runner import latest_runs, load_run

DEFAULT_WINDOW = 7


class DriftReport(BaseModel):
    window_size: int
    run_ids: List[str]
    pass_rates: List[float]
    rolling_average: float
    previous_rolling_average: Optional[float]
    is_drifting: bool
    threshold: float


def compute_drift(
    runs_dir: str | Path = "runs",
    window: int = DEFAULT_WINDOW,
    threshold: float = 0.90,
) -> Optional[DriftReport]:
    """
    threshold: if the rolling average pass rate over `window` runs drops
    below this, flag a slow-drift warning. Tune per how strict your bar is.
    """
    files = latest_runs(runs_dir, n=window + 1)  # +1 so we can compare against the prior window too
    if len(files) < 2:
        return None

    runs: List[RunResult] = [load_run(f) for f in files]
    current_window = runs[-window:] if len(runs) >= window else runs
    prior_window = runs[:-1][-window:] if len(runs) > window else None

    current_avg = sum(r.pass_rate for r in current_window) / len(current_window)
    prior_avg = (
        sum(r.pass_rate for r in prior_window) / len(prior_window) if prior_window else None
    )

    return DriftReport(
        window_size=len(current_window),
        run_ids=[r.run_id for r in current_window],
        pass_rates=[r.pass_rate for r in current_window],
        rolling_average=current_avg,
        previous_rolling_average=prior_avg,
        is_drifting=current_avg < threshold,
        threshold=threshold,
    )


def drift_alert_text(drift: DriftReport) -> str:
    trend = ""
    if drift.previous_rolling_average is not None:
        change = drift.rolling_average - drift.previous_rolling_average
        trend = f" ({change:+.1%} vs prior {drift.window_size}-run window)"
    return (
        f"🐢 Slow drift warning: {drift.window_size}-run rolling average pass rate is "
        f"{drift.rolling_average:.1%}, below the {drift.threshold:.0%} threshold{trend}. "
        f"No single run tripped a regression alert, but the trend is degrading."
    )
