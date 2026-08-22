FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY prompts/ prompts/
COPY golden_dataset/ golden_dataset/

# Runs are written here; mount a volume to persist history across container runs
RUN mkdir -p runs

# Configurable at run time:
#   LLM_PROVIDER      - "mock" (default, offline) or "openai"
#   OPENAI_API_KEY     - required if LLM_PROVIDER=openai
#   SLACK_WEBHOOK_URL  - required to actually send Slack alerts (omit for dry-run/print)
ENV LLM_PROVIDER=mock

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["eval", "--prompt", "prompts/email_classifier_v1.yaml", "--alert", "--fail-on-critical"]
