# GitHub Actions Workflows

This directory contains GitHub Actions workflows for CI/CD automation.

## Available Workflows

### `ci.yml` - Comprehensive CI Pipeline

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch
- Manual workflow dispatch

**Jobs:**
- **Lint and Test:** Runs on Python 3.8-3.12 matrix
  - Code linting with ruff (main rules only: E, F, W)
  - Code formatting check with black
  - Type checking with mypy (non-blocking)
  - Unit tests with pytest
  - Coverage reporting with Codecov

**Features:**
- Dependency caching for faster runs
- Multi-Python version testing
- Code coverage tracking

---

### `lint-and-test.yml` - Simplified Workflow

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch
- Manual workflow dispatch

**Jobs:**

1. **Lint Job** (Python 3.12)
   - Ruff linting
   - Black formatting check

2. **Test Job** (Python 3.8-3.12 matrix)
   - Unit tests with pytest
   - Coverage reporting

3. **Validate Job**
   - Runs validation script
   - Confirms environment setup

**Features:**
- Parallel job execution
- Clear separation of concerns
- Validation after tests pass

---

## Workflow Status

You can view workflow runs at:
`https://github.com/yourusername/kbcraft/actions`

## Running Checks Locally

To run the same checks that CI runs:

```bash
# Linting
poetry run ruff check src/ tests/
poetry run black --check src/ tests/

# Auto-fix
poetry run ruff check --fix src/ tests/
poetry run black src/ tests/

# Tests
poetry run pytest -v --cov=kbcraft

# Validation
poetry run python validate_setup.py
```

## Configuration

Linting rules are configured in `pyproject.toml`:

```toml
[tool.ruff.lint]
select = ["E", "F", "W"]  # Main rules only
ignore = ["E501"]         # Line too long (handled by black)
```

## Adding New Workflows

1. Create a new `.yml` file in this directory
2. Use the existing workflows as templates
3. Test locally before pushing
4. Update this README with the new workflow details
