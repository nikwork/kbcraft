#!/bin/bash
# Test CI checks locally before pushing to GitHub

set -e  # Exit on error

echo "=============================================="
echo "  Testing CI Checks Locally"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print success
success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to print info
info() {
    echo -e "${YELLOW}→${NC} $1"
}

echo "Step 1: Check Poetry is installed"
if command -v poetry &> /dev/null; then
    success "Poetry is installed ($(poetry --version))"
else
    error "Poetry is not installed"
    exit 1
fi
echo ""

echo "Step 2: Check Python version"
python_version=$(python --version)
info "Python version: $python_version"
echo ""

echo "Step 3: Install dependencies"
info "Running: poetry install"
poetry install --no-interaction
success "Dependencies installed"
echo ""

echo "Step 4: Run linting with ruff"
info "Running: poetry run ruff check src/ tests/"
if poetry run ruff check src/ tests/; then
    success "Ruff linting passed"
else
    error "Ruff linting failed"
    echo ""
    info "Fix with: poetry run ruff check --fix src/ tests/"
    exit 1
fi
echo ""

echo "Step 5: Check code formatting with black"
info "Running: poetry run black --check src/ tests/"
if poetry run black --check src/ tests/ > /dev/null 2>&1; then
    success "Black formatting check passed"
else
    error "Black formatting check failed"
    echo ""
    info "Fix with: poetry run black src/ tests/"
    exit 1
fi
echo ""

echo "Step 6: Run tests with pytest"
info "Running: poetry run pytest -v"
if poetry run pytest -v; then
    success "All tests passed"
else
    error "Tests failed"
    exit 1
fi
echo ""

echo "Step 7: Check test coverage"
info "Running: poetry run pytest --cov=kbcraft --cov-report=term-missing"
poetry run pytest --cov=kbcraft --cov-report=term-missing
echo ""

echo "Step 8: Run validation script"
info "Running: poetry run python validate_setup.py"
if poetry run python validate_setup.py; then
    success "Validation passed"
else
    error "Validation failed"
    exit 1
fi
echo ""

echo "=============================================="
echo -e "${GREEN}✓ All CI checks passed locally!${NC}"
echo "=============================================="
echo ""
echo "You can now push to GitHub with confidence."
echo ""
echo "To push:"
echo "  git add ."
echo "  git commit -m 'Your commit message'"
echo "  git push"
echo ""
