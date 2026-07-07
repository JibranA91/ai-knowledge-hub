# ── test stage ────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS test

WORKDIR /app

ENV LANGCHAIN_TRACING_V2=false \
    LANGSMITH_TRACING=false \
    LANGCHAIN_CALLBACKS_BACKGROUND=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt requirements-test.txt ./
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt -r requirements-test.txt

COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY data/schema/ ./data/schema/
COPY tests/ ./tests/
COPY pytest.ini .

RUN pytest tests/unit -q

# ── production stage ───────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Disable LangSmith's background tracer thread — it tries to connect to langsmith.com
# on import and spawns threads even when tracing is not configured.
ENV LANGCHAIN_TRACING_V2=false \
    LANGSMITH_TRACING=false \
    LANGCHAIN_CALLBACKS_BACKGROUND=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY data/schema/ ./data/schema/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]
