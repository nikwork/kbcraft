FROM python:3.12-slim

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copy dependency files first for layer caching
COPY pyproject.toml poetry.lock README.md ./

# Install runtime dependencies only (no pytest, black, ruff, mypy)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Copy source
COPY src/ src/
COPY validate_setup.py ./

# Install the kbcraft package itself
RUN pip install --no-cache-dir -e .

# Default: show help
CMD ["kbcraft", "--help"]
