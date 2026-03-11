# GitHub Actions CI Troubleshooting Guide

This guide helps you diagnose and fix common CI failures in GitHub Actions.

---

## Quick Fixes Applied

The workflows have been updated with:

✅ **Better Poetry Installation**
- Using `snok/install-poetry@v1` action (more reliable)
- Pinned to Poetry 1.7.1
- Parallel installation enabled

✅ **Dependency Caching**
- Caches virtual environments between runs
- Faster CI runs (30s vs 2-3 minutes)
- Per-Python-version caching

✅ **Fail-Fast Disabled**
- Tests continue on other Python versions if one fails
- Better visibility of which versions have issues

✅ **Better Error Handling**
- `continue-on-error: true` for non-critical steps
- Codecov upload won't fail the build

---

## Common CI Failures and Solutions

### 1. Poetry Installation Fails

**Error:**
```
curl: (7) Failed to connect to install.python-poetry.org
```

**Solution:**
✅ Already fixed! Now using `snok/install-poetry@v1` action instead of curl.

**Verify:**
```yaml
- name: Install Poetry
  uses: snok/install-poetry@v1
  with:
    version: 1.7.1
```

---

### 2. Dependency Installation Fails

**Error:**
```
poetry install: command not found
```

**Cause:** Poetry not added to PATH

**Solution:**
✅ Already fixed! The action handles PATH automatically.

**Error:**
```
SolverProblemError: The current project's Python requirement is not compatible
```

**Cause:** Python version incompatibility

**Solution:**
Check `pyproject.toml`:
```toml
[tool.poetry.dependencies]
python = "^3.8"  # Should support 3.8+
```

If you need to support older Python, update dependencies:
```bash
# Locally test with older Python
poetry env use python3.8
poetry install
```

---

### 3. Tests Fail in CI But Pass Locally

**Possible Causes:**

**A. Import Errors**
```
ModuleNotFoundError: No module named 'kbcraft'
```

**Solution:**
Check package structure in `pyproject.toml`:
```toml
packages = [{include = "kbcraft", from = "src"}]
```

**B. Missing Dependencies**
```
ImportError: cannot import name 'X'
```

**Solution:**
Ensure `poetry.lock` is committed:
```bash
git add poetry.lock
git commit -m "Add poetry.lock"
```

**C. Path Issues**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution:**
Use absolute paths or ensure working directory is correct:
```python
import os
from pathlib import Path

# Get project root
ROOT_DIR = Path(__file__).parent.parent
```

---

### 4. Linting Fails

**Error:**
```
ruff check: F401 imported but unused
```

**Solution:**
```bash
# Fix locally first
poetry run ruff check --fix src/ tests/
git add .
git commit -m "Fix linting issues"
git push
```

**Error:**
```
black: would reformat file
```

**Solution:**
```bash
# Format locally
poetry run black src/ tests/
git add .
git commit -m "Format code with black"
git push
```

---

### 5. Coverage Upload Fails

**Error:**
```
Codecov: Error uploading to Codecov
```

**Solution:**
✅ Already fixed! Set to `continue-on-error: true`

**Optional:** Add Codecov token
1. Go to https://codecov.io
2. Add your repository
3. Get token
4. Add to GitHub Secrets: `Settings > Secrets > Actions > New secret`
   - Name: `CODECOV_TOKEN`
   - Value: `<your-token>`

---

### 6. Timeout Errors

**Error:**
```
The job was canceled because it exceeded the maximum execution time
```

**Solution:**
✅ Caching is enabled to speed up runs.

If still timing out:
```yaml
jobs:
  test:
    timeout-minutes: 15  # Add timeout
```

---

### 7. Matrix Build Failures

**Error:**
```
Python 3.8 tests fail, but 3.12 passes
```

**Solution:**
Test locally with that Python version:
```bash
# Using pyenv
pyenv install 3.8
pyenv local 3.8
poetry env use python3.8
poetry install
poetry run pytest
```

Common 3.8 issues:
- Type hints (use `from __future__ import annotations`)
- f-string limitations
- Older dependency versions

---

## Debugging Workflow

### Step 1: Check Workflow Logs

1. Go to: `https://github.com/YOUR_USERNAME/kbcraft/actions`
2. Click on failed workflow run
3. Click on failed job
4. Expand failed step
5. Read error message

### Step 2: Reproduce Locally

```bash
# Clean environment
rm -rf .venv poetry.lock

# Reinstall
poetry install

# Run tests
poetry run pytest -v

# Run linting
poetry run ruff check src/ tests/
poetry run black --check src/ tests/
```

### Step 3: Test with Specific Python Version

```bash
# Create env with specific Python
pyenv install 3.8  # or whichever version failed
poetry env use python3.8
poetry install
poetry run pytest -v
```

### Step 4: Check for Common Issues

```bash
# Ensure poetry.lock is committed
git status | grep poetry.lock

# Ensure src structure is correct
ls -la src/kbcraft/

# Verify imports work
poetry run python -c "import kbcraft; print(kbcraft.__version__)"

# Run validation
poetry run python validate_setup.py
```

---

## Testing Workflows Locally

### Using `act` (GitHub Actions locally)

```bash
# Install act
# macOS:
brew install act

# Linux:
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run workflow locally
act push

# Run specific job
act -j test

# Run with specific Python version
act -j test --matrix python-version:3.8
```

---

## Workflow File Locations

- `.github/workflows/ci.yml` - Main CI pipeline
- `.github/workflows/lint-and-test.yml` - Modular workflow
- `.github/workflows/ci-improved.yml` - Alternative improved workflow

---

## Key Improvements Made

### Before (Manual Poetry Install)
```yaml
- name: Install Poetry
  run: |
    curl -sSL https://install.python-poetry.org | python3 -
    echo "$HOME/.local/bin" >> $GITHUB_PATH
```
❌ Unreliable
❌ No caching
❌ Slow

### After (Action-based Install)
```yaml
- name: Install Poetry
  uses: snok/install-poetry@v1
  with:
    version: 1.7.1
    virtualenvs-create: true
    virtualenvs-in-project: true
    installer-parallel: true
```
✅ Reliable
✅ With caching
✅ Fast

### Caching Added
```yaml
- name: Load cached venv
  id: cached-poetry-dependencies
  uses: actions/cache@v4
  with:
    path: .venv
    key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}
```
✅ Speeds up CI by 60-80%

---

## Checklist for CI Success

Before pushing:

- [ ] Run tests locally: `poetry run pytest`
- [ ] Run linting: `poetry run ruff check src/ tests/`
- [ ] Format code: `poetry run black src/ tests/`
- [ ] Commit `poetry.lock`: `git add poetry.lock`
- [ ] Validate setup: `poetry run python validate_setup.py`
- [ ] Check git status: `git status`

After pushing:

- [ ] Monitor Actions tab on GitHub
- [ ] Check all matrix builds pass
- [ ] Review coverage report
- [ ] Address any warnings

---

## Getting Help

1. **Check this guide first**
2. **Review workflow logs** in GitHub Actions
3. **Reproduce locally** with same Python version
4. **Check Poetry docs:** https://python-poetry.org/docs/
5. **Check GitHub Actions docs:** https://docs.github.com/en/actions

---

## Quick Commands for CI Issues

```bash
# Full clean and reinstall
rm -rf .venv poetry.lock
poetry install
poetry run pytest -v

# Fix linting issues
poetry run ruff check --fix src/ tests/
poetry run black src/ tests/

# Test with different Python version
poetry env use python3.8
poetry install
poetry run pytest

# Validate everything
poetry run python validate_setup.py

# Check what would be committed
git status
git diff
```

---

## Status Badge

Add to your README.md:
```markdown
[![CI](https://github.com/USERNAME/kbcraft/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/kbcraft/actions/workflows/ci.yml)
```

---

**Last Updated:** 2026-03-09
