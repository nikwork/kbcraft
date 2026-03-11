"""
Tests for FileFilter in selector.py.
"""

from pathlib import Path

import pytest

from kbcraft.selector import LANGUAGE_PRESETS, FileFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tree(tmp_path: Path, files: list) -> None:
    """Create a set of empty files under tmp_path."""
    for rel in files:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


# ---------------------------------------------------------------------------
# LANGUAGE_PRESETS
# ---------------------------------------------------------------------------


class TestLanguagePresets:
    def test_all_expected_presets_present(self):
        for name in ["markdown", "python", "javascript", "typescript", "shell", "go", "rust"]:
            assert name in LANGUAGE_PRESETS

    def test_each_preset_has_at_least_one_pattern(self):
        for name, patterns in LANGUAGE_PRESETS.items():
            assert patterns, f"Preset '{name}' has no patterns"

    def test_markdown_includes_md(self):
        assert any("*.md" in p for p in LANGUAGE_PRESETS["markdown"])

    def test_shell_includes_sh(self):
        assert any("*.sh" in p for p in LANGUAGE_PRESETS["shell"])

    def test_python_includes_py(self):
        assert any("*.py" in p for p in LANGUAGE_PRESETS["python"])


# ---------------------------------------------------------------------------
# FileFilter.from_presets
# ---------------------------------------------------------------------------


class TestFromPresets:
    def test_single_preset(self):
        f = FileFilter.from_presets(["python"])
        assert any("*.py" in p for p in f.include_patterns)

    def test_multiple_presets_merged(self):
        f = FileFilter.from_presets(["python", "shell"])
        patterns = f.include_patterns
        assert any("*.py" in p for p in patterns)
        assert any("*.sh" in p for p in patterns)

    def test_no_duplicate_patterns(self):
        f = FileFilter.from_presets(["javascript", "javascript"])
        assert len(f.include_patterns) == len(set(f.include_patterns))

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            FileFilter.from_presets(["cobol"])

    def test_unknown_preset_lists_available(self):
        with pytest.raises(ValueError, match="Available:"):
            FileFilter.from_presets(["cobol"])

    def test_exclude_patterns_passed_through(self):
        f = FileFilter.from_presets(["python"], exclude_patterns=["tests/**"])
        assert "tests/**" in f.exclude_patterns

    def test_preset_collects_correct_files(self, tmp_path):
        make_tree(tmp_path, ["main.py", "run.sh", "readme.md", "data.json"])
        f = FileFilter.from_presets(["python", "shell"])
        files = f.collect_files(tmp_path)
        names = {p.name for p in files}
        assert names == {"main.py", "run.sh"}


# ---------------------------------------------------------------------------
# FileFilter.should_include
# ---------------------------------------------------------------------------


class TestShouldInclude:
    def _f(self, include=None, exclude=None):
        return FileFilter(include_patterns=include, exclude_patterns=exclude)

    def test_default_includes_md(self, tmp_path):
        p = tmp_path / "guide.md"
        p.touch()
        assert self._f().should_include(p, tmp_path) is True

    def test_default_excludes_py(self, tmp_path):
        p = tmp_path / "main.py"
        p.touch()
        assert self._f().should_include(p, tmp_path) is False

    def test_custom_include_txt(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.touch()
        assert self._f(include=["**/*.txt"]).should_include(p, tmp_path) is True

    def test_exclude_overrides_include(self, tmp_path):
        p = tmp_path / "drafts" / "wip.md"
        p.parent.mkdir()
        p.touch()
        f = self._f(include=["**/*.md"], exclude=["drafts/**"])
        assert f.should_include(p, tmp_path) is False

    def test_filename_wildcard_exclude(self, tmp_path):
        p = tmp_path / "sub" / "_private.md"
        p.parent.mkdir()
        p.touch()
        assert self._f(include=["**/*.md"], exclude=["_*"]).should_include(p, tmp_path) is False

    def test_directory_slash_pattern(self, tmp_path):
        p = tmp_path / "node_modules" / "readme.md"
        p.parent.mkdir()
        p.touch()
        assert self._f(include=["**/*.md"], exclude=["node_modules/"]).should_include(p, tmp_path) is False

    def test_non_excluded_file_still_included(self, tmp_path):
        p = tmp_path / "docs" / "guide.md"
        p.parent.mkdir()
        p.touch()
        assert self._f(include=["**/*.md"], exclude=["drafts/**"]).should_include(p, tmp_path) is True


# ---------------------------------------------------------------------------
# FileFilter.collect_files
# ---------------------------------------------------------------------------


class TestCollectFiles:
    def test_collects_md_only_by_default(self, tmp_path):
        make_tree(tmp_path, ["a.md", "sub/b.md", "c.py", "d.txt"])
        files = FileFilter().collect_files(tmp_path)
        names = {p.name for p in files}
        assert names == {"a.md", "b.md"}

    def test_collect_python_and_shell(self, tmp_path):
        make_tree(tmp_path, ["app.py", "deploy.sh", "readme.md"])
        f = FileFilter.from_presets(["python", "shell"])
        names = {p.name for p in f.collect_files(tmp_path)}
        assert names == {"app.py", "deploy.sh"}

    def test_excludes_directory(self, tmp_path):
        make_tree(tmp_path, ["src/main.py", "tests/test_main.py"])
        f = FileFilter.from_presets(["python"], exclude_patterns=["tests/**"])
        names = {p.name for p in f.collect_files(tmp_path)}
        assert names == {"main.py"}

    def test_returns_sorted_paths(self, tmp_path):
        make_tree(tmp_path, ["z.md", "a.md", "m.md"])
        files = FileFilter().collect_files(tmp_path)
        assert files == sorted(files)

    def test_empty_directory(self, tmp_path):
        assert FileFilter().collect_files(tmp_path) == []

    def test_accepts_string_path(self, tmp_path):
        (tmp_path / "a.md").touch()
        files = FileFilter().collect_files(str(tmp_path))
        assert len(files) == 1


# ---------------------------------------------------------------------------
# FileFilter.from_kbignore
# ---------------------------------------------------------------------------


class TestFromKbignore:
    def test_loads_exclude_patterns(self, tmp_path):
        kbi = tmp_path / ".kbignore"
        kbi.write_text("drafts/**\n_*\n")
        make_tree(tmp_path, ["docs/guide.md", "drafts/wip.md", "_hidden.md"])
        f = FileFilter.from_kbignore(kbi)
        names = {p.name for p in f.collect_files(tmp_path)}
        assert names == {"guide.md"}

    def test_ignores_comments_and_blank_lines(self, tmp_path):
        kbi = tmp_path / ".kbignore"
        kbi.write_text("# comment\n\ndrafts/**\n")
        make_tree(tmp_path, ["guide.md", "drafts/wip.md"])
        names = {p.name for p in FileFilter.from_kbignore(kbi).collect_files(tmp_path)}
        assert "guide.md" in names
        assert "wip.md" not in names

    def test_negation_removes_exclude(self, tmp_path):
        kbi = tmp_path / ".kbignore"
        kbi.write_text("drafts/**\n!drafts/**\n")
        make_tree(tmp_path, ["guide.md", "drafts/wip.md"])
        names = {p.name for p in FileFilter.from_kbignore(kbi).collect_files(tmp_path)}
        assert "wip.md" in names

    def test_missing_file_silently_ignored(self, tmp_path):
        make_tree(tmp_path, ["guide.md"])
        f = FileFilter.from_kbignore(tmp_path / ".kbignore")
        assert len(f.collect_files(tmp_path)) == 1

    def test_extra_excludes_combined_with_file(self, tmp_path):
        kbi = tmp_path / ".kbignore"
        kbi.write_text("drafts/**\n")
        make_tree(tmp_path, ["guide.md", "drafts/wip.md", "_hidden.md"])
        f = FileFilter.from_kbignore(kbi, extra_excludes=["_*"])
        names = {p.name for p in f.collect_files(tmp_path)}
        assert names == {"guide.md"}

    def test_custom_include_presets_via_from_kbignore(self, tmp_path):
        kbi = tmp_path / ".kbignore"
        kbi.write_text("")
        make_tree(tmp_path, ["main.py", "run.sh", "readme.md"])
        f = FileFilter.from_kbignore(kbi, include_patterns=["**/*.py", "**/*.sh"])
        names = {p.name for p in f.collect_files(tmp_path)}
        assert names == {"main.py", "run.sh"}
