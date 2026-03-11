"""
File selection for knowledge base ingestion.

Determines which files from a project are collected for chunking, embedding,
and insertion into a vector store.
"""

import fnmatch
import re
from functools import lru_cache
from pathlib import Path, PurePath
from typing import Dict, List, Optional


@lru_cache(maxsize=256)
def _compile_glob(pattern: str) -> re.Pattern:
    """Compile a glob pattern (with ``**`` support) to a regex.

    ``**`` matches zero or more path segments (including their separators).
    ``*``  matches any character sequence within a single path segment.
    ``?``  matches any single character within a segment.
    """
    parts = []
    i = 0
    while i < len(pattern):
        if pattern[i: i + 3] == "**/":
            parts.append("(?:.*/)?")  # zero or more dir segments
            i += 3
        elif pattern[i: i + 2] == "**":
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(parts) + "$")


# ---------------------------------------------------------------------------
# Language presets
# ---------------------------------------------------------------------------

#: Glob patterns for each supported language/file-type preset.
#: Use ``FileFilter.from_presets()`` to build a filter from one or more names.
LANGUAGE_PRESETS: Dict[str, List[str]] = {
    "markdown":   ["**/*.md", "**/*.mdx"],
    "python":     ["**/*.py"],
    "javascript": ["**/*.js", "**/*.jsx", "**/*.mjs", "**/*.cjs"],
    "typescript": ["**/*.ts", "**/*.tsx"],
    "shell":      ["**/*.sh", "**/*.bash", "**/*.zsh", "**/*.fish"],
    "go":         ["**/*.go"],
    "rust":       ["**/*.rs"],
    "java":       ["**/*.java"],
    "c":          ["**/*.c", "**/*.h"],
    "cpp":        ["**/*.cpp", "**/*.cc", "**/*.cxx", "**/*.hpp", "**/*.hxx"],
    "ruby":       ["**/*.rb"],
    "php":        ["**/*.php"],
    "swift":      ["**/*.swift"],
    "kotlin":     ["**/*.kt", "**/*.kts"],
    "scala":      ["**/*.scala"],
    "yaml":       ["**/*.yaml", "**/*.yml"],
    "toml":       ["**/*.toml"],
    "json":       ["**/*.json"],
    "html":       ["**/*.html", "**/*.htm"],
    "css":        ["**/*.css", "**/*.scss", "**/*.sass", "**/*.less"],
    "sql":        ["**/*.sql"],
}


class FileFilter:
    """Decide which files from a project directory should be included in a vector store.

    Supports:
    - **Language presets** via ``FileFilter.from_presets()``
    - **Custom glob patterns** via ``include_patterns`` / ``exclude_patterns``
    - **A .kbignore file** (gitignore-style) via ``FileFilter.from_kbignore()``

    Exclude patterns are checked first — if any exclude matches, the file is
    rejected regardless of includes.

    Pattern rules:
    - ``**/*.md``       — any .md file at any depth
    - ``*.md``          — .md filename (also matches files in subdirectories)
    - ``drafts/**``     — everything inside a drafts/ directory
    - ``private/``      — any directory named "private" (trailing slash = dir)
    - ``_*.md``         — files whose name starts with underscore

    Example::

        # Include Python and shell files, skip tests and hidden dirs
        f = FileFilter.from_presets(
            ["python", "shell"],
            exclude_patterns=["tests/**", ".*/**"],
        )
        files = f.collect_files("/path/to/project")
    """

    DEFAULT_INCLUDE: List[str] = list(LANGUAGE_PRESETS["markdown"])

    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> None:
        self.include_patterns: List[str] = (
            list(include_patterns) if include_patterns is not None else list(self.DEFAULT_INCLUDE)
        )
        self.exclude_patterns: List[str] = list(exclude_patterns) if exclude_patterns else []

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_presets(
        cls,
        presets: List[str],
        exclude_patterns: Optional[List[str]] = None,
    ) -> "FileFilter":
        """Build a FileFilter from one or more named language presets.

        Args:
            presets: List of preset names (e.g. ``["python", "shell"]``).
                     Use ``list(LANGUAGE_PRESETS)`` to see all available names.
            exclude_patterns: Additional glob patterns to exclude.

        Raises:
            ValueError: If an unknown preset name is given.
        """
        unknown = [p for p in presets if p not in LANGUAGE_PRESETS]
        if unknown:
            available = ", ".join(sorted(LANGUAGE_PRESETS))
            raise ValueError(
                f"Unknown preset(s): {', '.join(unknown)}. "
                f"Available: {available}"
            )

        include_patterns: List[str] = []
        for name in presets:
            for pattern in LANGUAGE_PRESETS[name]:
                if pattern not in include_patterns:
                    include_patterns.append(pattern)

        return cls(include_patterns=include_patterns, exclude_patterns=exclude_patterns)

    @classmethod
    def from_kbignore(
        cls,
        kbignore_path,
        include_patterns: Optional[List[str]] = None,
        extra_excludes: Optional[List[str]] = None,
    ) -> "FileFilter":
        """Create a FileFilter loading exclude patterns from a .kbignore file.

        The .kbignore format is gitignore-style:
        - Blank lines and lines starting with ``#`` are ignored.
        - ``!pattern`` negates (removes) a previously added exclude pattern.
        - All other lines are treated as exclude glob patterns.

        Args:
            kbignore_path: Path to the .kbignore file. A missing file is silently ignored.
            include_patterns: Passed through to FileFilter. Defaults to ``["**/*.md"]``.
            extra_excludes: Additional exclude patterns applied before parsing the file.
        """
        exclude_patterns: List[str] = list(extra_excludes or [])

        path = Path(kbignore_path)
        if path.exists():
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("!"):
                    negated = line[1:].strip()
                    exclude_patterns = [p for p in exclude_patterns if p != negated]
                else:
                    exclude_patterns.append(line)

        return cls(include_patterns=include_patterns, exclude_patterns=exclude_patterns)

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_pattern(rel_path: PurePath, pattern: str) -> bool:
        """Return True if *rel_path* matches *pattern*.

        Handles:
        - ``dir/`` trailing-slash patterns (any ancestor directory named ``dir``)
        - ``**`` recursive wildcards via a compiled regex
        - Simple filename wildcards (``_*``, ``*.pyc``) matched against the filename
          so they work even when the file is inside subdirectories
        """
        if pattern.endswith("/"):
            dir_name = pattern.rstrip("/")
            return any(fnmatch.fnmatch(part, dir_name) for part in rel_path.parts[:-1])

        rel_str = rel_path.as_posix()

        # Full path match with ** support
        if _compile_glob(pattern).match(rel_str):
            return True

        # Filename-only match for patterns without path separators (e.g. "_*", "*.pyc")
        if "/" not in pattern and fnmatch.fnmatch(rel_path.name, pattern):
            return True

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_include(self, file_path: Path, root: Path) -> bool:
        """Return True if *file_path* should be included in the knowledge base.

        Args:
            file_path: Absolute path to the file being evaluated.
            root: The root directory used to compute the relative path.
        """
        rel = PurePath(file_path.relative_to(root))

        for pattern in self.exclude_patterns:
            if self._matches_pattern(rel, pattern):
                return False

        for pattern in self.include_patterns:
            if self._matches_pattern(rel, pattern):
                return True

        return False

    def collect_files(self, root_dir) -> List[Path]:
        """Walk *root_dir* and return all files accepted by this filter.

        Args:
            root_dir: Directory to walk. Accepts str or Path.

        Returns:
            Sorted list of absolute Path objects.
        """
        root = Path(root_dir).resolve()
        results = [
            file_path
            for file_path in root.rglob("*")
            if file_path.is_file() and self.should_include(file_path, root)
        ]
        return sorted(results)
