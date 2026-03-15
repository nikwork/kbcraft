# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir poetry==1.8.3

# Copy lock files first — deps layer is cached until lock file changes
COPY pyproject.toml poetry.lock README.md ./

RUN poetry config virtualenvs.in-project true \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Build a wheel and install it non-editably so the runtime stage needs no src/
COPY src/ src/

RUN poetry build -f wheel \
    && .venv/bin/pip install --no-deps dist/*.whl

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

CMD ["kbcraft", "--help"]
