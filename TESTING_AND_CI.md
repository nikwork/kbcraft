# Testing and CI/CD Documentation

This document provides comprehensive information about testing strategies, continuous integration, and continuous deployment for the kbcraft project.

---

## Table of Contents

- [Testing Strategy](#testing-strategy)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Code Coverage](#code-coverage)
- [Continuous Integration](#continuous-integration)
- [CI/CD Workflows](#cicd-workflows)
- [Pre-commit Checks](#pre-commit-checks)
- [Troubleshooting](#troubleshooting)

---

## Testing Strategy

The kbcraft project uses **pytest** as the primary testing framework with the following approach:

### Test Organization

```
tests/
├── __init__.py
├── test_kbcraft.py      # Core module and import tests
├── test_scaffold.py     # Scaffolding functionality tests
├── test_chunker.py      # Document chunking tests
├── test_embedder.py     # Embedding generation tests
└── ...                  # Additional test modules
```

### Test Types

1. **Unit Tests**
   - Test individual functions and classes in isolation
   - Fast execution
   - High coverage of edge cases

2. **Integration Tests**
   - Test interactions between modules
   - Verify data flow through components

3. **Import Tests**
   - Verify all modules can be imported
   - Check package structure integrity
   - Validate version information

### Test Naming Conventions

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

Example:
```python
def test_function_name():
    """Test description."""
    assert result == expected

class TestClassName:
    """Test suite for ClassName."""

    def test_method_name(self):
        """Test description."""
        assert result == expected
```

---

## Running Tests

### Basic Test Execution

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_kbcraft.py

# Run specific test function
poetry run pytest tests/test_kbcraft.py::test_version_exists

# Run specific test class
poetry run pytest tests/test_scaffold.py::TestScaffold
```

### Running Tests with Coverage

```bash
# Run tests with coverage report
poetry run pytest --cov=kbcraft

# Run with detailed missing lines report
poetry run pytest --cov=kbcraft --cov-report=term-missing

# Generate HTML coverage report
poetry run pytest --cov=kbcraft --cov-report=html
open htmlcov/index.html

# Generate XML coverage report (for CI)
poetry run pytest --cov=kbcraft --cov-report=xml
```

### Test Options

```bash
# Stop on first failure
poetry run pytest -x

# Show local variables in tracebacks
poetry run pytest -l

# Disable output capture (see print statements)
poetry run pytest -s

# Run tests in parallel (requires pytest-xdist)
poetry run pytest -n auto

# Run only failed tests from last run
poetry run pytest --lf

# Run tests matching a keyword expression
poetry run pytest -k "test_import"
```

---

## Writing Tests

### Basic Test Structure

```python
"""
Tests for module_name functionality.
"""

import pytest
from kbcraft import module_name


def test_basic_functionality():
    """Test basic functionality."""
    result = module_name.function()
    assert result == expected_value


class TestModuleName:
    """Test suite for module_name."""

    def test_method_one(self):
        """Test method one."""
        obj = module_name.ClassName()
        assert obj.method_one() is not None

    def test_method_two(self):
        """Test method two with edge case."""
        obj = module_name.ClassName()
        with pytest.raises(ValueError):
            obj.method_two(invalid_input)
```

### Using Fixtures

```python
import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        "key": "value",
        "items": [1, 2, 3]
    }


def test_with_fixture(sample_data):
    """Test using fixture."""
    assert "key" in sample_data
    assert len(sample_data["items"]) == 3
```

### Parametrized Tests

```python
import pytest


@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply_by_two(input, expected):
    """Test multiplication by two with multiple inputs."""
    assert input * 2 == expected
```

### Testing Exceptions

```python
import pytest


def test_raises_exception():
    """Test that function raises expected exception."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_raises(invalid_value)
```

---

## Code Coverage

### Current Coverage Status

As of the latest run:
- **Total Coverage:** 60%
- **Fully Covered Modules:** All modules except cli.py
- **Lines to Cover:** 23 total statements, 2 missing

### Coverage Goals

- **Minimum:** 80% overall coverage
- **Target:** 90%+ coverage for core modules
- **Critical Modules:** 100% coverage required

### Checking Coverage

```bash
# Quick coverage check
poetry run pytest --cov=kbcraft --cov-report=term

# Detailed coverage with missing lines
poetry run pytest --cov=kbcraft --cov-report=term-missing

# Branch coverage (includes conditional branches)
poetry run pytest --cov=kbcraft --cov-branch

# Coverage for specific module
poetry run pytest --cov=kbcraft.scaffold tests/test_scaffold.py
```

### Coverage Configuration

Coverage settings in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-v --cov=kbcraft --cov-report=term-missing"
```

---

## Continuous Integration

The project uses **GitHub Actions** for automated CI/CD with multiple workflows.

### CI Triggers

All workflows trigger on:
- **Push to main branch**
- **Pull requests to main branch**
- **Manual workflow dispatch** (via GitHub UI)

### CI Philosophy

- **Fast feedback:** Quick linting before running full test suite
- **Multi-version testing:** Ensure compatibility across Python 3.8-3.12
- **Fail fast:** Stop on critical errors to save CI minutes
- **Comprehensive reporting:** Coverage reports and test results

---

## CI/CD Workflows

### Workflow 1: `ci.yml` - Comprehensive CI Pipeline

**Purpose:** Full validation with dependency caching and matrix testing

**Jobs:**
```yaml
lint-and-test:
  - Runs on: ubuntu-latest
  - Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
  - Uses dependency caching for speed
```

**Steps:**
1. Checkout code
2. Set up Python (matrix version)
3. Install Poetry
4. Load cached dependencies (if available)
5. Install project dependencies
6. Run ruff linting
7. Check code formatting with black
8. Type check with mypy (non-blocking)
9. Run pytest with coverage
10. Upload coverage to Codecov (Python 3.12 only)

**Features:**
- ✓ Dependency caching for faster runs
- ✓ Multi-version Python testing
- ✓ Coverage tracking with Codecov
- ✓ Type checking (informational)

**When to Use:**
- Primary CI check for all commits
- Default workflow for pull requests

---

### Workflow 2: `lint-and-test.yml` - Modular Workflow

**Purpose:** Separate linting, testing, and validation into distinct jobs

**Job 1: Lint** (Python 3.12)
```bash
- poetry run ruff check src/ tests/
- poetry run black --check src/ tests/
```

**Job 2: Test** (Python 3.8-3.12 matrix)
```bash
- poetry run pytest -v --cov=kbcraft --cov-report=xml
```

**Job 3: Validate** (runs after lint and test pass)
```bash
- poetry run python validate_setup.py
```

**Features:**
- ✓ Parallel job execution
- ✓ Clear separation of concerns
- ✓ Validation confirms environment setup

**When to Use:**
- When you need granular CI step visibility
- Debugging specific CI failures
- Manual workflow dispatch for validation

---

## Linting Rules

### Ruff Configuration

The project uses **main rules only** for ruff linting:

```toml
[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "W",   # pycodestyle warnings
]
ignore = [
    "E501",  # Line too long (handled by black)
]
```

### Rule Categories

**E - pycodestyle errors**
- PEP 8 style compliance
- Indentation errors
- Whitespace issues
- Statement formatting

**F - pyflakes**
- Undefined names
- Unused imports
- Unused variables
- Import errors

**W - pycodestyle warnings**
- Deprecated features
- Potential issues
- Style warnings

### Running Linting

```bash
# Check all files
poetry run ruff check src/ tests/

# Auto-fix issues
poetry run ruff check --fix src/ tests/

# Show all violations
poetry run ruff check --no-cache src/ tests/

# Check specific file
poetry run ruff check src/kbcraft/cli.py
```

### Code Formatting

The project uses **Black** for consistent code formatting:

```bash
# Check formatting
poetry run black --check src/ tests/

# Format code
poetry run black src/ tests/

# Check specific file
poetry run black --check src/kbcraft/cli.py

# Show diff without formatting
poetry run black --diff src/ tests/
```

**Black Configuration:**

```toml
[tool.black]
line-length = 100
target-version = ['py38']
```

---

## Pre-commit Checks

### Recommended Pre-commit Workflow

Before committing code, run these checks locally:

```bash
# 1. Format code
poetry run black src/ tests/

# 2. Fix linting issues
poetry run ruff check --fix src/ tests/

# 3. Run tests
poetry run pytest -v

# 4. Check coverage
poetry run pytest --cov=kbcraft --cov-report=term-missing

# 5. Validate setup
poetry run python validate_setup.py
```

### One-liner Pre-commit Check

```bash
poetry run black src/ tests/ && \
poetry run ruff check src/ tests/ && \
poetry run pytest -v --cov=kbcraft && \
echo "✓ All checks passed - ready to commit!"
```

### Using Make Commands

```bash
# Format and lint
make format
make lint

# Run tests
make test

# All checks
make format && make lint && make test
```

---

## CI Status Badges

The project README includes the following CI badges:

```markdown
[![CI](https://github.com/yourusername/kbcraft/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/kbcraft/actions/workflows/ci.yml)
[![Lint and Test](https://github.com/yourusername/kbcraft/actions/workflows/lint-and-test.yml/badge.svg)](https://github.com/yourusername/kbcraft/actions/workflows/lint-and-test.yml)
```

These badges show:
- ✓ Green: All checks passing
- ✗ Red: One or more checks failing
- ○ Yellow: Checks in progress

---

## Troubleshooting

### Common CI Failures

#### 1. Linting Failures

**Error:** `ruff check` fails with unused imports

**Solution:**
```bash
# View the specific errors
poetry run ruff check src/ tests/

# Auto-fix
poetry run ruff check --fix src/ tests/
```

#### 2. Formatting Failures

**Error:** `black --check` fails

**Solution:**
```bash
# Format the code
poetry run black src/ tests/

# Verify
poetry run black --check src/ tests/
```

#### 3. Test Failures

**Error:** Tests fail in CI but pass locally

**Possible Causes:**
- Different Python version
- Missing dependencies
- Environment-specific issues

**Solution:**
```bash
# Test with specific Python version
poetry env use python3.8
poetry install
poetry run pytest

# Check for missing dependencies
poetry install --sync
```

#### 4. Import Errors

**Error:** `ModuleNotFoundError` in CI

**Solution:**
```bash
# Ensure package is installed
poetry install

# Verify package structure
poetry run python -c "import kbcraft; print(kbcraft.__version__)"

# Run validation
poetry run python validate_setup.py
```

#### 5. Coverage Upload Failures

**Error:** Codecov upload fails

**Solution:**
- Ensure `CODECOV_TOKEN` is set in GitHub secrets
- Check coverage file exists: `coverage.xml`
- Set `fail_ci_if_error: false` to make uploads optional

### Local vs CI Differences

| Aspect | Local | CI |
|--------|-------|-----|
| Python Version | Your system version | Matrix: 3.8-3.12 |
| Dependencies | May be stale | Fresh install each run |
| Caching | None by default | Enabled in CI |
| Environment | Your machine | Clean Ubuntu container |

### Getting Help

If CI continues to fail:

1. **Check CI logs:** View full output in GitHub Actions
2. **Run locally:** Reproduce the exact CI commands
3. **Use validation script:** `poetry run python validate_setup.py`
4. **Check this doc:** Review relevant troubleshooting section

---

## Best Practices

### For Developers

1. **Run tests before committing**
   ```bash
   poetry run pytest -v
   ```

2. **Check coverage for new code**
   ```bash
   poetry run pytest --cov=kbcraft --cov-report=term-missing
   ```

3. **Format code automatically**
   ```bash
   poetry run black src/ tests/
   ```

4. **Fix linting issues**
   ```bash
   poetry run ruff check --fix src/ tests/
   ```

### For Maintainers

1. **Require CI passing before merge**
   - Enable branch protection on `main`
   - Require status checks to pass

2. **Review coverage reports**
   - Monitor coverage trends
   - Require tests for new features

3. **Update dependencies regularly**
   ```bash
   poetry update
   poetry lock
   ```

4. **Keep workflows up to date**
   - Use latest action versions
   - Test with new Python versions

---

## Quick Reference

### Essential Commands

```bash
# Testing
poetry run pytest                           # Run all tests
poetry run pytest -v                        # Verbose output
poetry run pytest --cov=kbcraft            # With coverage

# Linting
poetry run ruff check src/ tests/          # Check code
poetry run ruff check --fix src/ tests/    # Auto-fix

# Formatting
poetry run black src/ tests/               # Format code
poetry run black --check src/ tests/       # Check formatting

# Validation
poetry run python validate_setup.py        # Validate environment

# Pre-commit
poetry run black src/ tests/ && \
poetry run ruff check src/ tests/ && \
poetry run pytest -v
```

### CI Workflow Files

- `.github/workflows/ci.yml` - Main CI pipeline
- `.github/workflows/lint-and-test.yml` - Modular workflow
- `.github/workflows/README.md` - Workflow documentation

### Configuration Files

- `pyproject.toml` - All tool configurations
- `.python-version` - pyenv Python version
- `poetry.lock` - Locked dependencies

---

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [ruff documentation](https://docs.astral.sh/ruff/)
- [black documentation](https://black.readthedocs.io/)
- [Poetry documentation](https://python-poetry.org/docs/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)

---

**Last Updated:** 2026-03-09
**Maintained by:** kbcraft team
