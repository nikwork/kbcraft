# Docker Architecture

## Overview

The Docker setup runs three services via Docker Compose: an Ollama embedding server, a one-shot model puller, and the kbcraft application itself.

## Services

### `ollama`
- **Image:** `ollama/ollama:latest`
- **Purpose:** Serves the embedding model over HTTP on port `11434`.
- **Persistence:** Model weights are stored in a named volume (`ollama-models`) so pulled models survive container restarts.
- **Health check:** `ollama list` polled every 10 s; other services wait for healthy status before starting.

### `ollama-pull`
- **Image:** `ollama/ollama:latest` (reused)
- **Purpose:** One-shot init container — runs `ollama pull all-minilm` once, then exits (`restart: no`).
- **Model:** `all-minilm` — 384-dimensional embeddings, ~22 MB, CPU-friendly.
- **Idempotency:** `ollama pull` is a no-op when the model is already in the shared volume, so re-running `docker compose up` is safe.

### `kbcraft`
- **Image:** Built from local `Dockerfile`.
- **Purpose:** Runs the kbcraft CLI. Default command is `kbcraft --help`; override with `docker compose run kbcraft bash` for interactive use.
- **Mounts:**
  - `./docs → /app/docs` — local Markdown files to index.
  - `./scripts → /app/scripts` — helper scripts accessible inside the container.
- **Env:** `OLLAMA_HOST=http://ollama:11434` wires it to the Ollama service.

## Dockerfile

| Step | What it does |
|------|-------------|
| `FROM python:3.12-slim` | Minimal base image |
| Install Poetry 1.8.3 | Dependency manager |
| Copy `pyproject.toml` + `poetry.lock` first | Layer cache: deps only rebuilt when lock file changes |
| `poetry install --only main --no-root` | Runtime deps only — no dev tools (pytest, ruff, black, mypy) |
| Copy `src/` | Application source |
| `pip install -e .` | Install kbcraft package in editable mode |
| `CMD ["kbcraft", "--help"]` | Default entrypoint |

## Simplification Proposals

1. **Remove the `ollama-pull` sidecar** — replace with an `entrypoint` script on the `ollama` service itself that pulls the model after the server starts. Eliminates one container and removes the dependency chain.

2. **Use a multi-stage build** — add a builder stage that installs Poetry and resolves deps, then copy only the installed site-packages into the final image. Avoids shipping Poetry in the runtime image.

3. **Pin the Ollama image tag** — `ollama/ollama:latest` will silently break on a future incompatible release. Pin to a specific version (e.g. `ollama/ollama:0.3`).

4. **Drop `pip install -e .`** — since `poetry install --only main` already handles the package, the extra `pip install -e .` step is redundant. Use `poetry install --only main` (without `--no-root`) to install the package in one step.

5. **Use `COPY . .` guard with `.dockerignore`** — currently `pyproject.toml`, `poetry.lock`, `README.md`, and `src/` are copied separately. A well-configured `.dockerignore` (excluding `tests/`, `kb/`, `.git/`, etc.) plus a single `COPY . .` achieves the same layer-cache benefit more cleanly.

6. **`stdin_open: true` + `tty: true` on kbcraft** — these are only needed for interactive sessions. They add no value when the service is run non-interactively and can be confusing. Move them to a `docker-compose.override.yml` for local dev.
