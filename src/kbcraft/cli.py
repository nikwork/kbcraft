"""
Command-line interface for kbcraft.
"""

import argparse
import sys
from pathlib import Path

from kbcraft.selector import LANGUAGE_PRESETS, FileFilter


def _build_parser() -> argparse.ArgumentParser:
    preset_names = sorted(LANGUAGE_PRESETS)

    parser = argparse.ArgumentParser(
        prog="kbcraft",
        description="Build structured, RAG-ready knowledge bases from Markdown.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------ #
    # collect — resolve which files would be indexed                       #
    # ------------------------------------------------------------------ #
    collect = subparsers.add_parser(
        "collect",
        help="List files that would be added to the vector store.",
        description=(
            "Walk a source directory and print every file that passes the "
            "current include/exclude filters. Useful for previewing what "
            "will be indexed before running a full build."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available language presets:\n  " + "\n  ".join(preset_names) + "\n\n"
            "Examples:\n"
            "  # Markdown only (default)\n"
            "  kbcraft collect ./docs\n\n"
            "  # Python and shell scripts, skip tests\n"
            "  kbcraft collect ./myproject --lang python --lang shell --exclude 'tests/**'\n\n"
            "  # Mix presets with custom patterns\n"
            "  kbcraft collect ./myproject --lang markdown --include '**/*.rst'\n\n"
            "  # Use a .kbignore file\n"
            "  kbcraft collect ./myproject --lang python --kbignore .kbignore\n"
        ),
    )
    collect.add_argument(
        "source_dir",
        metavar="SOURCE_DIR",
        help="Root directory to scan.",
    )
    collect.add_argument(
        "--lang",
        metavar="LANGUAGE",
        action="append",
        dest="languages",
        help=(
            f"Language preset to include. Can be repeated. "
            f"Available: {', '.join(preset_names)}. "
            "When --lang is given, --include defaults are ignored."
        ),
    )
    collect.add_argument(
        "--include",
        metavar="PATTERN",
        action="append",
        dest="include_patterns",
        help=(
            "Extra glob pattern to include. Can be repeated. "
            "Added on top of any --lang presets. "
            "If neither --lang nor --include is given, defaults to **/*.md."
        ),
    )
    collect.add_argument(
        "--exclude",
        metavar="PATTERN",
        action="append",
        dest="exclude_patterns",
        help=(
            "Glob pattern for files to exclude. Can be repeated. "
            "Exclusions are checked before inclusions. "
            "Example: --exclude 'drafts/**' --exclude '_*'"
        ),
    )
    collect.add_argument(
        "--kbignore",
        metavar="FILE",
        default=None,
        help=(
            "Path to a .kbignore file (gitignore-style exclude rules). "
            "Defaults to <SOURCE_DIR>/.kbignore when that file exists."
        ),
    )

    # ------------------------------------------------------------------ #
    # presets — list available language presets                            #
    # ------------------------------------------------------------------ #
    subparsers.add_parser(
        "presets",
        help="List all available language presets and their file patterns.",
    )

    return parser


def _cmd_collect(args: argparse.Namespace) -> int:
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"error: '{source_dir}' is not a directory", file=sys.stderr)
        return 1

    # Build include patterns: start from presets, then append custom patterns.
    include_patterns = None  # None → FileFilter uses its default (**/*.md)

    if args.languages:
        try:
            preset_filter = FileFilter.from_presets(args.languages)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        include_patterns = list(preset_filter.include_patterns)

    if args.include_patterns:
        if include_patterns is None:
            include_patterns = []
        for p in args.include_patterns:
            if p not in include_patterns:
                include_patterns.append(p)

    # Resolve .kbignore
    if args.kbignore:
        kbignore_path = Path(args.kbignore)
    else:
        kbignore_path = source_dir / ".kbignore"

    file_filter = FileFilter.from_kbignore(
        kbignore_path=kbignore_path,
        include_patterns=include_patterns,
        extra_excludes=args.exclude_patterns,
    )

    files = file_filter.collect_files(source_dir)

    if not files:
        print("No files matched.")
        return 0

    for f in files:
        print(f.relative_to(source_dir.resolve()))

    print(f"\n{len(files)} file(s) matched.", file=sys.stderr)
    return 0


def _cmd_presets() -> int:
    col = max(len(name) for name in LANGUAGE_PRESETS) + 2
    print("Available language presets:\n")
    for name, patterns in sorted(LANGUAGE_PRESETS.items()):
        print(f"  {name:<{col}}{', '.join(patterns)}")
    return 0


def main() -> None:
    """Main entry point for the kbcraft CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        sys.exit(_cmd_collect(args))
    elif args.command == "presets":
        sys.exit(_cmd_presets())
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
