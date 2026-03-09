# CI/CD Quick Reference Card

Quick reference for common CI/CD and testing commands.

---

## 🚀 Quick Commands

### Run All CI Checks Locally

```bash
# One-liner to run all checks
poetry run black src/ tests/ && \
poetry run ruff check src/ tests/ && \
poetry run pytest -v --cov=kbcraft && \
echo "✓ All CI checks passed!"
```

### Individual Checks

```bash
# Lint code
poetry run ruff check src/ tests/

# Check formatting
poetry run black --check src/ tests/

# Run tests
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=kbcraft --cov-report=term-missing

# Validate environment
poetry run python validate_setup.py
```

---

## 🔧 Auto-fix Issues

```bash
# Fix linting issues
poetry run ruff check --fix src/ tests/

# Format code
poetry run black src/ tests/

# Both at once
poetry run ruff check --fix src/ tests/ && poetry run black src/ tests/
```

---

## 📊 Coverage Commands

```bash
# Basic coverage
poetry run pytest --cov=kbcraft

# With missing lines
poetry run pytest --cov=kbcraft --cov-report=term-missing

# HTML report
poetry run pytest --cov=kbcraft --cov-report=html
open htmlcov/index.html

# XML for CI
poetry run pytest --cov=kbcraft --cov-report=xml
```

---

## 🧪 Test Variations

```bash
# Stop on first failure
poetry run pytest -x

# Show print statements
poetry run pytest -s

# Run specific test
poetry run pytest tests/test_kbcraft.py::test_version_exists

# Run tests matching keyword
poetry run pytest -k "import"

# Verbose output
poetry run pytest -v
```

---

## 🔍 Linting Rules

### Ruff (Main Rules Only)
- **E** - pycodestyle errors
- **F** - pyflakes
- **W** - pycodestyle warnings

### Ignored
- **E501** - Line too long (black handles this)

---

## 📁 Key Files

- `.github/workflows/ci.yml` - Main CI pipeline
- `.github/workflows/lint-and-test.yml` - Modular workflow
- `pyproject.toml` - Tool configurations
- `TESTING_AND_CI.md` - Full documentation

---

## 🎯 Pre-commit Checklist

```bash
□ poetry run black src/ tests/
□ poetry run ruff check src/ tests/
□ poetry run pytest -v
□ poetry run pytest --cov=kbcraft
□ git add .
□ git commit -m "Your message"
```

---

## 🚨 Common Fixes

### "Unused import 'pytest'"
```bash
# Remove unused pytest import or use it
# Auto-fix:
poetry run ruff check --fix tests/
```

### "would reformat file"
```bash
# Format the file:
poetry run black src/ tests/
```

### "Test failed"
```bash
# Run with more details:
poetry run pytest -v -s
```

---

## 📖 More Info

See [TESTING_AND_CI.md](../TESTING_AND_CI.md) for comprehensive documentation.

---

**Print this card or bookmark it for quick access!**
