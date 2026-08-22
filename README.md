# LLM Regression Guard

A CI/CD-style evaluation pipeline for LLM-powered features. It runs a golden
dataset against a prompt on every change, scores the output on multiple
dimensions, diffs the result against a baseline run, and alerts the team
through Slack when quality drops — before a bad prompt change reaches
production.

The feature under test is a customer-support email classifier (billing /
technical / account / general), but the pipeline itself doesn't care what
the feature is. Swap `src/llm_feature.py` for whatever LLM-powered function
you're shipping and the rest of the system — scoring, diffing, alerting,
CI — keeps working unchanged.

## Why

Prompt changes usually ship the same way a copy edit does: someone tweaks
the wording, it looks fine on a couple of manual checks, and it merges.
There's rarely a test suite behind it the way there would be for
application code, so a prompt that quietly gets worse on an entire category
of inputs can sit in production for weeks before anyone notices.

This project treats a prompt the way you'd treat any other code: it gets
tested against a fixed dataset on every change, and a regression blocks the
merge instead of reaching users.

## How it works

1. **Prompts are versioned.** Each one lives in `prompts/*.yaml` with a
   version id, model, system prompt, and few-shot examples, loaded into a
   typed `PromptConfig`.
2. **A golden dataset holds the ground truth.** `golden_dataset/test_cases_v1.json`
   has 50 hand-written support emails, each with a verified expected
   category and summary. A handful are deliberately hard — short, sarcastic,
   mixed-language, or genuinely ambiguous — and tagged by difficulty.
3. **Every run scores every case** on two dimensions: exact category match,
   and summary relevance (1–5, scored by an LLM-as-judge — see note below).
4. **Every run gets diffed against a baseline**: overall pass-rate delta,
   per-category deltas, and the specific cases that flipped pass → fail
   (regressions) or fail → pass (improvements).
5. **Severity is threshold-based.** A pass-rate drop under 3% is `ok`,
   3–8% is a `warning`, 8%+ is `critical` and can block a merge.
   Thresholds are configurable.
6. **A separate drift check watches the trend.** A slow decline across
   several runs won't trip any single-run threshold, so a rolling average
   over the last N runs is tracked independently and flagged if it drops
   below its own threshold.
7. **Output:** an HTML diff report (scorecard, per-category table,
   side-by-side regressions, trend chart) and a Slack message with the
   headline numbers.

## Setup

```bash
pip install -r requirements.txt
```

By default the pipeline runs against a mock LLM provider — a deterministic,
offline stand-in for the classifier, so the whole thing runs for free with
no API key. It's built to be noticeably worse on `prompts/email_classifier_v2.yaml`
(a weaker prompt with the disambiguation rules and most few-shot examples
stripped out) than on `v1`, so there's a real regression for the pipeline
to catch out of the box.

To point it at a real model:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

To send real Slack alerts:

```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

Without a webhook set, alerts print to stdout instead of failing.

## Usage

```bash
# Run one prompt version through the golden dataset and save the result
python -m src.cli run --prompt prompts/email_classifier_v1.yaml

# Run a new version, diff against the most recent prior run,
# generate the HTML report, and send a Slack alert
python -m src.cli eval --prompt prompts/email_classifier_v2.yaml --alert

# Same, but exit non-zero on a critical regression (for blocking CI)
python -m src.cli eval --prompt prompts/email_classifier_v2.yaml --alert --fail-on-critical

# Check the rolling-average drift signal directly
python -m src.cli drift
```

Run history is saved to `runs/*.json`, one file per run, timestamp-prefixed.
The repo ships with two runs already recorded (`v1` baseline, `v2`
regression) so running `eval` against them produces a real diff immediately.

## Project layout

```
prompts/                    versioned prompt configs
golden_dataset/              hand-verified test cases
src/
  config.py                  typed models: PromptConfig, TestCase, RunResult, ...
  llm_feature.py               the feature under test, pluggable provider (mock / openai)
  eval_runner.py                async-batched test runner
  scoring.py                     category match + summary relevance scoring
  diff_engine.py                  regression/improvement diffing + significance thresholds
  drift.py                         rolling-average drift detection
  report.py                        HTML diff report generator
  alerts.py                        Slack webhook integration
  cli.py                           entrypoint used by CI and Docker
runs/                        saved run history
tests/                       unit tests
.github/workflows/eval.yml    CI: runs the pipeline on every PR touching prompts/
Dockerfile                   packages the pipeline for a portable/offline run
```

## Adding a test case

Add an entry to `golden_dataset/test_cases_v1.json` with a unique `id`,
the `input_email`, the correct `expected_category` and `expected_summary`,
a difficulty tag, and a short note on why the case matters. Test cases are
written by hand, not generated by an LLM — the dataset's value depends on
it being independent ground truth, not the system grading itself.

## Design notes

- **The summary-relevance scorer is a stand-in for a real LLM judge.**
  `score_summary_relevance()` in `src/scoring.py` uses a word-overlap
  heuristic instead of an actual judge API call, so the pipeline runs free
  and offline. The function signature is the seam — swap the body for a
  real call and nothing downstream changes.
- **The mock LLM provider is intentionally prompt-sensitive.** Its accuracy
  depends on how much of the system prompt (disambiguation rules, few-shot
  examples) survives — which is what makes `v1` → `v2` a real, reproducible
  regression rather than a synthetic example.
- **Per-run diffing and drift detection are two separate systems.** A bad
  prompt change is a different failure mode from a model provider that
  degrades gradually over weeks; conflating the two into one check means
  one failure mode ends up masking the other.

## Running tests

```bash
pytest tests/
```

## License

MIT
