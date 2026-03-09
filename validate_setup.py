#!/usr/bin/env python
"""
Validation script to check if the kbcraft development environment is properly set up.
Run this after completing the setup steps to ensure everything is working.

Usage:
    poetry run python validate_setup.py
"""

import sys
import subprocess
from pathlib import Path


def check(description, test_func):
    """Run a test and print the result."""
    try:
        test_func()
        print(f"✓ {description}")
        return True
    except Exception as e:
        print(f"✗ {description}")
        print(f"  Error: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("kbcraft Development Environment Validation")
    print("=" * 60)
    print()

    checks_passed = 0
    checks_total = 0

    # Check 1: Import kbcraft
    checks_total += 1
    def test_import_kbcraft():
        import kbcraft
        assert kbcraft.__version__ == "0.1.0"

    if check("Import kbcraft module", test_import_kbcraft):
        checks_passed += 1

    # Check 2: Import all submodules
    checks_total += 1
    def test_import_submodules():
        from kbcraft import cli, scaffold, organize, chunker, embedder, sync, utils
        from kbcraft.vector_stores import chroma, qdrant, pinecone, faiss

    if check("Import all submodules", test_import_submodules):
        checks_passed += 1

    # Check 3: Project structure
    checks_total += 1
    def test_project_structure():
        required_files = [
            "pyproject.toml",
            "src/kbcraft/__init__.py",
            "src/kbcraft/cli.py",
            "tests/__init__.py",
        ]
        for file in required_files:
            assert Path(file).exists(), f"Missing: {file}"

    if check("Project structure is correct", test_project_structure):
        checks_passed += 1

    # Check 4: pyproject.toml is valid
    def test_pyproject():
        import tomli
        with open("pyproject.toml", "rb") as f:
            data = tomli.load(f)
        assert "tool" in data
        assert "poetry" in data["tool"]
        assert data["tool"]["poetry"]["name"] == "kbcraft"

    try:
        import tomli
        checks_total += 1
        if check("pyproject.toml is valid", test_pyproject):
            checks_passed += 1
    except ImportError:
        print("⊘ pyproject.toml validation (tomli not installed, skipping)")

    # Check 5: pytest is available
    checks_total += 1
    def test_pytest():
        result = subprocess.run(
            ["poetry", "run", "pytest", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        assert "pytest" in result.stdout

    if check("pytest is installed", test_pytest):
        checks_passed += 1

    # Check 6: black is available
    checks_total += 1
    def test_black():
        result = subprocess.run(
            ["poetry", "run", "black", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        assert "black" in result.stdout

    if check("black is installed", test_black):
        checks_passed += 1

    # Check 7: ruff is available
    checks_total += 1
    def test_ruff():
        result = subprocess.run(
            ["poetry", "run", "ruff", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        assert "ruff" in result.stdout

    if check("ruff is installed", test_ruff):
        checks_passed += 1

    # Check 8: mypy is available
    checks_total += 1
    def test_mypy():
        result = subprocess.run(
            ["poetry", "run", "mypy", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        assert "mypy" in result.stdout

    if check("mypy is installed", test_mypy):
        checks_passed += 1

    # Check 9: Python version
    checks_total += 1
    def test_python_version():
        assert sys.version_info >= (3, 8), f"Python {sys.version_info} is too old"

    if check(f"Python version is 3.8+ (current: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})", test_python_version):
        checks_passed += 1

    # Print summary
    print()
    print("=" * 60)
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)

    if checks_passed == checks_total:
        print()
        print("🎉 Success! Your development environment is ready to use.")
        print()
        print("Next steps:")
        print("  - Activate the environment: poetry shell")
        print("  - Start coding in src/kbcraft/")
        print("  - Run tests: poetry run pytest")
        print("  - Format code: poetry run black .")
        return 0
    else:
        print()
        print("⚠️  Some checks failed. Please review the errors above.")
        print("   See the 'Troubleshooting' section in README.md for help.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
