# kbcraft

[![CI](https://github.com/yourusername/kbcraft/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/kbcraft/actions/workflows/ci.yml)
[![Lint and Test](https://github.com/yourusername/kbcraft/actions/workflows/lint-and-test.yml/badge.svg)](https://github.com/yourusername/kbcraft/actions/workflows/lint-and-test.yml)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Build structured, RAG-ready knowledge bases from Markdown — straight from your terminal.**

`kbcraft` is a Python command-line utility for authoring, organizing, and indexing collections of Markdown files into vector stores for use in Retrieval-Augmented Generation (RAG) pipelines. It bridges the gap between human-readable documentation and machine-queryable knowledge.

---

## Features

- 📝 **Scaffold knowledge bases** — generate structured sets of `.md` files from templates or outlines
- 🗂️ **Organize & validate** — enforce consistent frontmatter, naming conventions, and directory layout
- 🔍 **Chunk & embed** — split documents intelligently and generate vector embeddings
- 🗄️ **Push to vector stores** — built-in support for Chroma, Qdrant, Pinecone, and FAISS
- 🔄 **Incremental sync** — only re-index documents that have changed
- 🖥️ **Pure CLI** — scriptable, CI-friendly, no GUI required

---

## File Selector

`selector.py` is the component responsible for deciding **which files from a project get collected** for chunking, embedding, and insertion into a vector store. It is always the first step in the ingestion pipeline.

### Language presets

Twenty built-in presets cover the most common file types. Combine as many as you need:

| Preset | Extensions |
|---|---|
| `markdown` | `.md`, `.mdx` |
| `python` | `.py` |
| `javascript` | `.js`, `.jsx`, `.mjs`, `.cjs` |
| `typescript` | `.ts`, `.tsx` |
| `shell` | `.sh`, `.bash`, `.zsh`, `.fish` |
| `go` | `.go` |
| `rust` | `.rs` |
| `java` | `.java` |
| `c` / `cpp` | `.c`, `.h` / `.cpp`, `.hpp`, … |
| `ruby` | `.rb` |
| `sql` | `.sql` |
| `yaml` / `toml` / `json` | config & data files |
| `html` / `css` | web assets |

List all presets and their exact patterns:

```bash
kbcraft presets
```

### CLI usage

```bash
# Default — collect all Markdown files
kbcraft collect ./docs

# Python project — source and shell scripts, skip tests and caches
kbcraft collect ./myproject \
  --lang python \
  --lang shell \
  --exclude 'tests/**' \
  --exclude '__pycache__/**'

# Python + Markdown docs, exclude private files and drafts
kbcraft collect ./myproject \
  --lang python \
  --lang markdown \
  --exclude '_*' \
  --exclude 'drafts/**'

# Add a custom extension on top of a preset
kbcraft collect ./myproject --lang markdown --include '**/*.rst'

# Use a .kbignore file for persistent exclude rules
kbcraft collect ./myproject --lang python --kbignore .kbignore
```

### Python API

```python
from kbcraft.selector import FileFilter, LANGUAGE_PRESETS

# --- Preset-based (recommended) ---

# Single language
f = FileFilter.from_presets(["python"])

# Multiple languages combined
f = FileFilter.from_presets(["python", "shell", "markdown"])

# With excludes
f = FileFilter.from_presets(
    ["python", "javascript"],
    exclude_patterns=["tests/**", "node_modules/", "__pycache__/"],
)

files = f.collect_files("./myproject")
for path in files:
    print(path)

# --- Custom glob patterns ---

f = FileFilter(
    include_patterns=["**/*.py", "**/*.md", "**/*.rst"],
    exclude_patterns=["drafts/**", "_*"],
)
files = f.collect_files("./myproject")

# --- .kbignore file ---

# Reads exclude patterns from .kbignore (gitignore-style)
f = FileFilter.from_kbignore(
    ".kbignore",
    include_patterns=FileFilter.from_presets(["python"]).include_patterns,
)
files = f.collect_files("./myproject")

# --- Check a single file ---

f = FileFilter.from_presets(["python"])
from pathlib import Path
root = Path("./myproject").resolve()
print(f.should_include(root / "src/main.py", root))   # True
print(f.should_include(root / "tests/test_main.py", root))  # True (not excluded)
```

### .kbignore file

Place a `.kbignore` file in your project root to permanently record which paths to skip. The format is identical to `.gitignore`:

```
# Ignore generated and cache directories
__pycache__/
.mypy_cache/
.pytest_cache/
*.pyc

# Ignore drafts and private files
drafts/**
_*

# Ignore test fixtures
tests/fixtures/**

# Un-ignore a specific file (! prefix removes an earlier exclude)
!drafts/approved.md
```

Pass it to the CLI with `--kbignore .kbignore`, or load it in Python with `FileFilter.from_kbignore(".kbignore")`.

---

## Running with Docker Compose

The Docker Compose setup starts a local [Ollama](https://ollama.com) inference server alongside the kbcraft application container. No Python ML dependencies or GPU required — Ollama handles all embedding inference as a REST service.

### Services

| Service | Image | Role |
|---|---|---|
| `ollama` | `ollama/ollama:latest` | Embedding inference server on port `11434` |
| `ollama-pull` | `ollama/ollama:latest` | One-shot init: pulls `all-minilm`, then exits |
| `kbcraft` | local build | kbcraft app, starts after Ollama healthcheck passes |

**Default model: `all-minilm`** — 384 dimensions, ~22 MB, fast on CPU. See [Switching the embedding model](#switching-the-embedding-model) to use a heavier model.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) ≥ 2 (included in Docker Desktop)

### Quick start

**1. Start all services:**

```bash
docker compose up -d
```

On first run this builds the kbcraft image and pulls `all-minilm` (~22 MB). Subsequent starts reuse the cached image and model volume.

**2. Watch the model pull finish:**

```bash
docker compose logs -f ollama-pull
# → Model ready.
```

**3. Verify all services are running:**

```bash
docker compose ps
```

Expected output:

```
NAME                    STATUS
kbcraft-ollama          running (healthy)
kbcraft-ollama-pull     exited (0)
kbcraft-app             running
```

**4. Check the kbcraft CLI:**

```bash
docker compose run kbcraft kbcraft --help
```

**5. Stop everything:**

```bash
docker compose down
```

> Pulled models are stored in the `ollama-models` Docker volume and survive `down`. Run `docker compose down -v` to also delete the volume.

---

## Testing OllamaEmbedder

### Run the smoke test (via Docker Compose)

After `docker compose up -d`, run the full smoke test against the live Ollama server:

```bash
docker compose run kbcraft python scripts/test_ollama.py
```

Expected output:

```
============================================================
  kbcraft — OllamaEmbedder smoke test
============================================================
  Ollama host : http://ollama:11434
  Model       : all-minilm

────────────────────────────────────────────────────────────
  1. Embedding dimension
────────────────────────────────────────────────────────────
  ✓  embedding_dim == 384

────────────────────────────────────────────────────────────
  2. encode() — plain batch
────────────────────────────────────────────────────────────
  ✓  returns 3 vectors
  ✓  each vector has 384 dims
  ✓  all values are floats

────────────────────────────────────────────────────────────
  3. encode_query()
────────────────────────────────────────────────────────────
  ✓  returns a single vector
  ✓  has 384 dims

────────────────────────────────────────────────────────────
  4. encode_documents()
────────────────────────────────────────────────────────────
  ✓  returns 2 vectors
  ✓  each has 384 dims

────────────────────────────────────────────────────────────
  5. Semantic similarity
────────────────────────────────────────────────────────────
  ✓  related pair scores higher than unrelated

────────────────────────────────────────────────────────────
  6. Auto-batching (batch_size=2)
────────────────────────────────────────────────────────────
  ✓  5 texts → 5 vectors

────────────────────────────────────────────────────────────
  7. ChromaDB adapter (as_chroma_ef)
────────────────────────────────────────────────────────────
  ✓  returns list of vectors
  ✓  vector has 384 dims

────────────────────────────────────────────────────────────
  8. FAISS adapter (as_faiss_matrix)
────────────────────────────────────────────────────────────
  ✓  shape is (2, 384)
  ✓  dtype is float32

============================================================
  All checks passed ✓
============================================================
```

### Run the smoke test locally (without Docker)

Requires Ollama installed and running on your machine:

```bash
# Install Ollama: https://ollama.com/download
ollama pull all-minilm
ollama serve          # if not already running as a service

python scripts/test_ollama.py
```

### Use OllamaEmbedder in Python

```python
from kbcraft.embedders import OllamaEmbedder

# Connect to Ollama (default: http://localhost:11434)
embedder = OllamaEmbedder(model="all-minilm")

# Encode a search query (no prefix needed for all-minilm)
query_vec = embedder.encode_query("how to handle errors in Python")

# Encode documents for indexing
doc_vecs = embedder.encode_documents([
    "try:\n    risky_call()\nexcept Exception as e:\n    handle(e)",
    "Use try/except to catch exceptions in Python.",
])

# Plain batch encode (no prefix, for clustering / deduplication)
vecs = embedder.encode(["hello world", "foo bar"])

# ChromaDB integration
import chromadb
client = chromadb.Client()
collection = client.get_or_create_collection(
    "my_docs",
    embedding_function=embedder.as_chroma_ef(),
)

# FAISS integration
import faiss, numpy as np
index = faiss.IndexFlatL2(embedder.embedding_dim)   # 384
index.add(embedder.as_faiss_matrix(["doc1", "doc2"]))

q = np.array([embedder.encode_query("search term")], dtype=np.float32)
distances, indices = index.search(q, k=3)
```

### Switching the embedding model

To use a different Ollama model, change two lines:

**`docker-compose.yml`** — update the pull command:
```yaml
entrypoint: >
  sh -c "ollama pull nomic-embed-text && echo 'Model ready.'"
```

**`scripts/test_ollama.py`** — update the model name:
```python
embedder = OllamaEmbedder(model="nomic-embed-text", host=OLLAMA_HOST)
```

Available CPU-friendly models:

| Model | Size | Dims | EN | RU | Code |
|---|---|---|---|---|---|
| `all-minilm` | ~22 MB | 384 | ✓ | limited | limited |
| `nomic-embed-text` | ~274 MB | 768 | ✓✓ | ✓ | ✓✓ |
| `mxbai-embed-large` | ~670 MB | 1024 | ✓✓ | ✓ | ✓ |
| `bge-m3` | ~1.2 GB | 1024 | ✓✓ | ✓✓ | ✓✓ |

### Troubleshooting

**`Cannot reach Ollama` error**

```bash
# Check Ollama is healthy
docker compose ps ollama
docker compose logs ollama

# Restart just the Ollama service
docker compose restart ollama
```

**Model not found**

```bash
# Re-run the pull service
docker compose run ollama-pull
```

**Rebuild kbcraft image after code changes**

```bash
docker compose build kbcraft
docker compose up -d
```

---

## Documentation

- **[Development Setup Guide](#development-setup-guide)** - Complete guide for setting up the project
- **[Testing and CI/CD](kb/TESTING_AND_CI.md)** - Comprehensive testing and CI/CD documentation
- **[Project Structure](#project-structure)** - Overview of the codebase organization

---

## Development Setup Guide

This guide walks you through setting up the kbcraft project for development on your local machine from scratch.

### Prerequisites

Before starting, ensure you have the following installed on your system:

1. **Git** - For cloning the repository
   ```bash
   git --version  # Should show version 2.x or higher
   ```

2. **Python 3.8+** - The project supports Python 3.8 through 3.12
   ```bash
   python3 --version  # Should show 3.8 or higher
   ```

3. **pyenv** (Recommended) - For managing Python versions
   ```bash
   # Install on macOS/Linux
   curl https://pyenv.run | bash

   # Or on macOS with Homebrew
   brew install pyenv

   # Verify installation
   pyenv --version
   ```

4. **Poetry** - Modern Python dependency management
   ```bash
   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -

   # Or with pipx (recommended)
   pipx install poetry

   # Verify installation
   poetry --version  # Should show 1.x or higher
   ```

### Step-by-Step Setup

#### Step 1: Install Python with pyenv

If you don't have Python 3.12.12 installed:

```bash
# Install Python 3.12.12
pyenv install 3.12.12

# Verify it was installed
pyenv versions
```

#### Step 2: Clone the Repository

```bash
# Clone from your git repository
git clone https://github.com/yourusername/kbcraft.git
# Or if using a local path:
git clone /path/to/kbcraft.git

# Navigate to the project directory
cd kbcraft

# Verify you're in the right directory
ls -la  # Should see pyproject.toml, src/, tests/, etc.
```

#### Step 3: Create Virtual Environment with pyenv

```bash
# Create a dedicated virtual environment for this project
pyenv virtualenv 3.12.12 kbcraft-env

# Set it as the local Python version (creates .python-version file)
pyenv local kbcraft-env

# Verify the environment is active
python --version  # Should show Python 3.12.12
which python      # Should point to kbcraft-env
```

#### Step 4: Install Dependencies with Poetry

```bash
# Install all project dependencies
poetry install

# This will:
# - Create a virtual environment (if not already created by pyenv)
# - Install all dependencies from pyproject.toml
# - Install the kbcraft package in editable mode
# - Install dev dependencies (pytest, black, ruff, mypy)
```

**Optional: Install with Vector Store Support**

```bash
# Install with specific vector store
poetry install --extras "chroma"

# Or install with all extras
poetry install --all-extras
```

#### Step 5: Verify Installation

**Quick Validation (Recommended):**

Run the automated validation script to check all setup requirements:

```bash
poetry run python validate_setup.py
```

This script will check:
- Module imports
- Project structure
- Python version
- Development tools (pytest, black, ruff, mypy)

Expected output:
```
============================================================
kbcraft Development Environment Validation
============================================================

✓ Import kbcraft module
✓ Import all submodules
✓ Project structure is correct
✓ pytest is installed
✓ black is installed
✓ ruff is installed
✓ mypy is installed
✓ Python version is 3.8+ (current: 3.12.12)

============================================================
Results: 8/8 checks passed
============================================================

🎉 Success! Your development environment is ready to use.
```

**Manual Validation (Optional):**

If you prefer to validate manually, run these commands:

**1. Check Python environment:**
```bash
poetry env info

# Expected output should show:
# - Python version: 3.12.12
# - Path to virtualenv
# - Valid: True
```

**2. Check installed package:**
```bash
poetry run python -c "import kbcraft; print(kbcraft.__version__)"

# Expected output: 0.1.0
```

**3. Test CLI command:**
```bash
poetry run kbcraft --help

# Should run without errors (even if output is empty for now)
```

**4. Verify module imports:**
```bash
poetry run python -c "
import kbcraft
from kbcraft import cli, scaffold, organize, chunker, embedder, sync, utils
from kbcraft.vector_stores import chroma, qdrant, pinecone, faiss
print('✓ All modules imported successfully')
"

# Expected output: ✓ All modules imported successfully
```

**5. Check development tools:**
```bash
# Test pytest
poetry run pytest --version

# Test black
poetry run black --version

# Test ruff
poetry run ruff --version

# Test mypy
poetry run mypy --version
```

#### Step 6: Run Tests

```bash
# Run the test suite
poetry run pytest

# Expected output:
# - Tests should be collected (even if there are no tests yet)
# - No import errors
# - Exit code 0 or 5 (no tests collected)
```

#### Step 7: Run Code Quality Checks

```bash
# Format code with black
poetry run black .

# Lint with ruff
poetry run ruff check .

# Type check with mypy
poetry run mypy src/kbcraft

# All commands should complete without critical errors
```

### Validation Checklist

**Quick Check:**

Run the validation script and ensure all checks pass:

```bash
poetry run python validate_setup.py
```

You should see: `Results: 8/8 checks passed` and `🎉 Success!`

**Detailed Checklist:**

If you prefer manual validation, use this checklist:

- [ ] Git repository cloned successfully
- [ ] Python 3.12.12 installed via pyenv
- [ ] Virtual environment `kbcraft-env` created and active
- [ ] Poetry installed and accessible
- [ ] `poetry install` completed without errors
- [ ] `poetry env info` shows correct Python version and valid environment
- [ ] `poetry run python -c "import kbcraft; print(kbcraft.__version__)"` prints `0.1.0`
- [ ] `poetry run kbcraft --help` runs without import errors
- [ ] All modules can be imported without errors
- [ ] Dev tools (pytest, black, ruff, mypy) are installed and working
- [ ] `poetry run pytest` runs without import errors
- [ ] Code formatting and linting tools work
- [ ] **OR** validation script passes with 8/8 checks ✓

### Troubleshooting

#### Issue: `poetry install` fails with "pyenv: python: command not found"

**Solution:**
```bash
# Ensure pyenv is properly initialized in your shell
# Add to ~/.bashrc or ~/.zshrc:
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Reload shell configuration
source ~/.bashrc  # or source ~/.zshrc
```

#### Issue: "No module named 'kbcraft'"

**Solution:**
```bash
# Reinstall the package
poetry install

# Or ensure you're using poetry run
poetry run python -c "import kbcraft"
```

#### Issue: Poetry creates its own virtualenv instead of using pyenv

**Solution:**
```bash
# Configure Poetry to create virtualenvs in the project directory
poetry config virtualenvs.in-project false
poetry config virtualenvs.prefer-active-python true

# Remove existing virtualenv and reinstall
poetry env remove python
poetry install
```

#### Issue: Permission denied when installing Poetry

**Solution:**
```bash
# Use pipx instead
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry
```

#### Issue: Tests fail to run

**Solution:**
```bash
# Verify pytest is installed
poetry show pytest

# Reinstall dev dependencies
poetry install --with dev
```

### Next Steps

Once your environment is validated and ready:

1. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

2. **Start coding:**
   - Implement features in `src/kbcraft/`
   - Write tests in `tests/`
   - Run tests frequently with `pytest`

3. **Before committing:**
   ```bash
   poetry run black .           # Format code
   poetry run ruff check .      # Lint code
   poetry run mypy src/kbcraft  # Type check
   poetry run pytest            # Run tests
   ```

4. **Useful commands during development:**
   ```bash
   poetry add <package>         # Add a new dependency
   poetry add --group dev <pkg> # Add a dev dependency
   poetry update                # Update dependencies
   poetry show --tree           # Show dependency tree
   ```

---

## Installation

### Using Poetry (Recommended)

**1. Clone the repository:**
```bash
git clone /path/to/kbcraft.git
cd kbcraft
```

**2. Set up Python environment with pyenv (optional but recommended):**
```bash
# Create a dedicated virtual environment
pyenv virtualenv 3.12.12 kbcraft-env

# Set it as the local Python version for this project
pyenv local kbcraft-env
```

**3. Install with Poetry:**
```bash
# Install all dependencies (will create a virtual environment automatically)
poetry install

# Install with specific vector store extras
poetry install --extras "chroma"
poetry install --extras "qdrant"
poetry install --extras "pinecone"
poetry install --extras "faiss"

# Install with all extras
poetry install --all-extras
```

**4. Run commands:**
```bash
# Activate the virtual environment
poetry shell

# Or run commands directly
poetry run kbcraft --help

# Import the module
poetry run python -c "import kbcraft; print(kbcraft.__version__)"
```

**5. Development commands:**
```bash
poetry run pytest              # Run tests
poetry run black .             # Format code
poetry run ruff check .        # Lint code
poetry run mypy kbcraft        # Type check
```

### Using pip

**1. Clone the repository:**
```bash
git clone /path/to/kbcraft.git
cd kbcraft
```

**2. Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install in development mode:**
```bash
# Basic installation
pip install -e .

# Install with dev tools (pytest, black, ruff, mypy)
pip install -e ".[dev]"

# Install with vector store support
pip install -e ".[chroma]"     # ChromaDB
pip install -e ".[qdrant]"     # Qdrant
pip install -e ".[pinecone]"   # Pinecone
pip install -e ".[faiss]"      # FAISS
```

**5. Install directly from local git repo (without cloning):**
```bash
# Install from local path
pip install git+file:///path/to/kbcraft.git

# Install in editable mode from local path
pip install -e git+file:///path/to/kbcraft.git#egg=kbcraft
```

**6. Verify installation:**
```bash
kbcraft --help
```

### Development Commands

**Using Poetry:**
```bash
poetry install                  # Install package with dependencies
poetry install --with dev       # Install with dev dependencies
poetry run pytest               # Run tests
poetry run ruff check .         # Run linting
poetry run mypy kbcraft         # Run type checking
poetry run black .              # Format code
poetry shell                    # Activate virtual environment
poetry env info                 # Show environment info
```

**Using Make:**
```bash
make install    # Install package
make dev        # Install with dev dependencies
make test       # Run tests
make lint       # Run linting (ruff + mypy)
make format     # Format code (black + ruff)
make clean      # Clean build artifacts
```

---

## Project Structure

```
kbcraft/
├── src/kbcraft/              # Main source code directory
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # Command-line interface entry point
│   ├── scaffold.py          # Scaffolding for generating .md files
│   ├── selector.py          # File selection for vector store ingestion
│   ├── chunker.py           # Document chunking functionality
│   ├── embedder.py          # Embedding generation
│   ├── sync.py              # Incremental sync functionality
│   ├── utils.py             # Utility functions
│   └── vector_stores/       # Vector store integrations
│       ├── __init__.py
│       ├── chroma.py        # ChromaDB integration
│       ├── qdrant.py        # Qdrant integration
│       ├── pinecone.py      # Pinecone integration
│       └── faiss.py         # FAISS integration
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_scaffold.py
│   ├── test_chunker.py
│   └── test_embedder.py
├── pyproject.toml           # Modern Python package configuration (Poetry)
├── poetry.lock              # Lock file for exact dependency versions
├── validate_setup.py        # Environment validation script
├── requirements.txt         # Python dependencies (legacy support)
├── Makefile                 # Development commands
├── MANIFEST.in              # Package manifest
├── .gitignore               # Git ignore patterns
├── .python-version          # pyenv Python version file
├── .github/workflows/       # GitHub Actions CI/CD workflows
│   ├── ci.yml              # Comprehensive CI pipeline
│   └── lint-and-test.yml   # Lint and test workflow
├── kb/                      # Knowledge base (CI guides, troubleshooting)
│   ├── TESTING_AND_CI.md   # Testing and CI/CD documentation
│   ├── CI_SETUP.md         # CI setup summary
│   ├── CI_FIXES.md         # CI fixes log
│   ├── CI_QUICK_REFERENCE.md  # CI quick reference card
│   └── TROUBLESHOOTING_CI.md  # CI troubleshooting guide
└── README.md                # This file
```

---

## Continuous Integration (CI)

> **📖 For comprehensive testing and CI/CD documentation, see [TESTING_AND_CI.md](kb/TESTING_AND_CI.md)**

The project uses GitHub Actions for automated testing and code quality checks.

### CI Pipeline

The CI pipeline runs automatically on:
- Push to `main` branch
- Pull requests to `main` branch
- Manual workflow dispatch

### Workflow Jobs

**1. Lint Job** - Code Quality Checks
- Runs ruff linter with main rules (E, F, W)
- Checks code formatting with black
- Runs on Python 3.12

**2. Test Job** - Unit Tests
- Runs pytest with coverage reporting
- Tests on multiple Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
- Uploads coverage report to Codecov (Python 3.12 only)

**3. Validate Job** - Environment Validation
- Runs the validation script
- Confirms all checks pass
- Runs after lint and test jobs complete

### Running CI Checks Locally

Before pushing code, run the same checks locally:

```bash
# Run linting
poetry run ruff check src/ tests/

# Check formatting
poetry run black --check src/ tests/

# Auto-fix issues
poetry run ruff check --fix src/ tests/
poetry run black src/ tests/

# Run tests
poetry run pytest -v --cov=kbcraft

# Run validation
poetry run python validate_setup.py
```

### CI Configuration Files

- `.github/workflows/ci.yml` - Main CI pipeline with matrix testing
- `.github/workflows/lint-and-test.yml` - Simplified lint and test workflow

### Linting Rules

The project uses **main rules only** for ruff:
- **E** - pycodestyle errors
- **F** - pyflakes (undefined names, unused imports)
- **W** - pycodestyle warnings

To see all linting issues:
```bash
poetry run ruff check src/ tests/
```

To auto-fix linting issues:
```bash
poetry run ruff check --fix src/ tests/
```

### Code Coverage

Test coverage is tracked and reported in CI. To generate a local coverage report:

```bash
# Terminal report
poetry run pytest --cov=kbcraft --cov-report=term-missing

# HTML report
poetry run pytest --cov=kbcraft --cov-report=html
open htmlcov/index.html
```
