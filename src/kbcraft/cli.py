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

    # ------------------------------------------------------------------ #
    # config — manage configs/*.yaml from a single user file               #
    # ------------------------------------------------------------------ #
    config = subparsers.add_parser(
        "config",
        help="Manage kbcraft configs/*.yaml from a single user-supplied file.",
    )
    config_sub = config.add_subparsers(dest="config_command")

    apply = config_sub.add_parser(
        "apply",
        help="Reset configs/*.yaml from a single combined user config file.",
        description=(
            "Read a single yaml file with top-level sections `embedding`, "
            "`storage`, `vector_store` and overwrite the matching files in "
            "configs/. Each section present is written as the *entire* content "
            "of its target file — unspecified options revert to dataclass "
            "defaults at load time. Sections omitted from the user file are "
            "left untouched on disk."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example user config:\n"
            "  embedding:\n"
            "    active_model: nomic-embed-text\n"
            "    backends:\n"
            "      ollama: {host: http://localhost:11434, timeout: 60}\n"
            "  storage:\n"
            "    active_backend: s3\n"
            "  vector_store:\n"
            "    active_backend: faiss\n"
        ),
    )
    apply.add_argument(
        "file",
        metavar="FILE",
        help="Path to the combined user config yaml.",
    )
    apply.add_argument(
        "--configs-dir",
        metavar="DIR",
        default="configs",
        help="Target configs directory. Default: ./configs",
    )
    apply.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying any files.",
    )

    # ------------------------------------------------------------------ #
    # index — full pipeline: select → chunk → embed → FAISS               #
    # ------------------------------------------------------------------ #
    index = subparsers.add_parser(
        "index",
        help="Build a FAISS vector index from a folder and save it to disk.",
        description=(
            "Run the full kbcraft pipeline on SOURCE_DIR:\n"
            "  1. Select files (same filters as 'collect')\n"
            "  2. Chunk files using the model's tokenizer\n"
            "  3. Embed chunks via the chosen backend\n"
            "  4. Build a FAISS flat-L2 index\n"
            "  5. Write index.faiss + chunks.json + meta.json to OUTPUT_DIR"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Embedder backends:\n"
            "  ollama        Local Ollama server (default).  Requires 'ollama serve'.\n"
            "  openai        Any OpenAI-compatible /v1/embeddings endpoint\n"
            "                (Ollama, vLLM, LM Studio, OpenAI, Azure, …).\n"
            "                Qwen3 models are detected automatically and use the\n"
            "                official Qwen3 tokenizer for exact token counting.\n\n"
            "Examples:\n"
            "  # Index Markdown docs with the default Ollama model\n"
            "  kbcraft index ./docs --output ./my_index\n\n"
            "  # Python + YAML with a specific Ollama model\n"
            "  kbcraft index ./src --lang python --lang yaml \\\n"
            "    --embedder ollama --model bge-m3\n\n"
            "  # Qwen3-0.6B via local Ollama OpenAI endpoint\n"
            "  kbcraft index ./docs --embedder openai \\\n"
            "    --model qwen3-embedding:0.6b\n\n"
            "  # Remote OpenAI-compatible server with auth\n"
            "  kbcraft index ./docs --embedder openai \\\n"
            "    --base-url https://my-server/v1 --token sk-... \\\n"
            "    --model text-embedding-3-small\n"
        ),
    )
    index.add_argument(
        "source_dir",
        metavar="SOURCE_DIR",
        help="Root directory to index.",
    )
    index.add_argument(
        "--output",
        metavar="DIR",
        default="kbcraft_index",
        help="Directory to write the index files to. Default: ./kbcraft_index",
    )
    # File selection (same as collect)
    index.add_argument(
        "--lang",
        metavar="LANGUAGE",
        action="append",
        dest="languages",
        help=(
            "Language preset to include. Can be repeated. "
            f"Available: {', '.join(preset_names)}. "
            "Defaults to markdown when no --lang or --include is given."
        ),
    )
    index.add_argument(
        "--include",
        metavar="PATTERN",
        action="append",
        dest="include_patterns",
        help="Extra glob pattern to include. Can be repeated.",
    )
    index.add_argument(
        "--exclude",
        metavar="PATTERN",
        action="append",
        dest="exclude_patterns",
        help="Glob pattern to exclude. Can be repeated.",
    )
    index.add_argument(
        "--kbignore",
        metavar="FILE",
        default=None,
        help="Path to a .kbignore file. Defaults to <SOURCE_DIR>/.kbignore.",
    )
    # Embedder
    index.add_argument(
        "--embedder",
        choices=["ollama", "openai"],
        default="ollama",
        help="Embedding backend. Default: ollama.",
    )
    index.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help=(
            "Model name. " "ollama default: nomic-embed-text. " "openai default: nomic-embed-text."
        ),
    )
    index.add_argument(
        "--host",
        metavar="URL",
        default="http://localhost:11434",
        help="Ollama server URL. Default: http://localhost:11434",
    )
    index.add_argument(
        "--base-url",
        metavar="URL",
        default="",
        help=(
            "Base URL for the OpenAI-compatible endpoint "
            "(e.g. http://localhost:11434/v1 for Ollama). "
            "Omit to use the real OpenAI API."
        ),
    )
    # Chunking
    index.add_argument(
        "--chunk-size",
        metavar="N",
        type=int,
        default=512,
        help="Maximum tokens per chunk. Default: 512.",
    )
    index.add_argument(
        "--chunk-overlap",
        metavar="N",
        type=int,
        default=50,
        help="Token overlap between adjacent chunks. Default: 50.",
    )
    index.add_argument(
        "--name",
        metavar="NAME",
        default="index",
        help=(
            "Base filename for the output files (no extension). "
            "Produces <NAME>.faiss, <NAME>_chunks.json, <NAME>_meta.json. "
            "Default: index"
        ),
    )

    # ------------------------------------------------------------------ #
    # query — search a previously built FAISS index                        #
    # ------------------------------------------------------------------ #
    query = subparsers.add_parser(
        "query",
        help="Search a FAISS index built by 'kbcraft index' and print ranked hits.",
        description=(
            "Load <NAME>.faiss + <NAME>_chunks.json from INDEX_DIR, embed the "
            "query with the same backend used to build the index, and print the "
            "top-k nearest chunks by L2 distance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Query an index built with OpenAI embeddings\n"
            "  kbcraft query ./my_index --name index --embedder openai \\\n"
            "    --model text-embedding-3-small -q 'how does chunking work'\n"
        ),
    )
    query.add_argument(
        "index_dir",
        metavar="INDEX_DIR",
        help="Directory containing the index files.",
    )
    query.add_argument(
        "-q",
        "--query",
        metavar="TEXT",
        required=True,
        help="Query text to search for.",
    )
    query.add_argument(
        "--name",
        metavar="NAME",
        default="index",
        help="Base filename of the index files (no extension). Default: index",
    )
    query.add_argument(
        "-k",
        "--k",
        metavar="N",
        type=int,
        default=3,
        help="Number of results to return. Default: 3.",
    )
    query.add_argument(
        "--json",
        action="store_true",
        help="Emit results as a JSON object on stdout (for scripts/CI).",
    )
    # Embedder options — must match how the index was built.
    query.add_argument(
        "--embedder",
        choices=["ollama", "openai"],
        default="ollama",
        help="Embedding backend. Must match the index. Default: ollama.",
    )
    query.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help="Model name. Must match the model used to build the index.",
    )
    query.add_argument(
        "--host",
        metavar="URL",
        default="http://localhost:11434",
        help="Ollama server URL. Default: http://localhost:11434",
    )
    query.add_argument(
        "--base-url",
        metavar="URL",
        default="",
        help="Base URL for an OpenAI-compatible endpoint. Omit for real OpenAI.",
    )

    return parser


# ── Command handlers ──────────────────────────────────────────────────────────


def _cmd_collect(args: argparse.Namespace) -> int:
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"error: '{source_dir}' is not a directory", file=sys.stderr)
        return 1

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


_CONFIG_SECTIONS = {
    "embedding": "embedding.yaml",
    "storage": "storage.yaml",
    "vector_store": "vector_store.yaml",
}


def _cmd_config_apply(args: argparse.Namespace) -> int:
    try:
        import yaml
    except ImportError:
        print("error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        return 1

    user_file = Path(args.file)
    if not user_file.is_file():
        print(f"error: '{user_file}' is not a file", file=sys.stderr)
        return 1

    configs_dir = Path(args.configs_dir)
    if not configs_dir.is_dir():
        print(f"error: configs dir '{configs_dir}' does not exist", file=sys.stderr)
        return 1

    with open(user_file, encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    if not isinstance(user_cfg, dict):
        print(f"error: '{user_file}' must be a yaml mapping at the top level", file=sys.stderr)
        return 1

    unknown = set(user_cfg) - set(_CONFIG_SECTIONS)
    if unknown:
        print(
            f"error: unknown section(s) in user config: {sorted(unknown)}. "
            f"Allowed: {sorted(_CONFIG_SECTIONS)}",
            file=sys.stderr,
        )
        return 1

    present = [s for s in _CONFIG_SECTIONS if s in user_cfg]
    if not present:
        print(
            f"error: user config has none of {sorted(_CONFIG_SECTIONS)}. Nothing to apply.",
            file=sys.stderr,
        )
        return 1

    verb = "Would write" if args.dry_run else "Wrote"
    for section in present:
        target = configs_dir / _CONFIG_SECTIONS[section]
        body = user_cfg[section] or {}
        rendered = yaml.safe_dump(body, sort_keys=False, default_flow_style=False)
        if args.dry_run:
            print(f"--- {target} ---")
            print(rendered, end="")
        else:
            target.write_text(rendered, encoding="utf-8")
        print(f"{verb} {target}", file=sys.stderr)

    skipped = [s for s in _CONFIG_SECTIONS if s not in user_cfg]
    if skipped:
        print(f"Left untouched: {', '.join(skipped)}", file=sys.stderr)
    return 0


def _cmd_presets() -> int:
    col = max(len(name) for name in LANGUAGE_PRESETS) + 2
    print("Available language presets:\n")
    for name, patterns in sorted(LANGUAGE_PRESETS.items()):
        print(f"  {name:<{col}}{', '.join(patterns)}")
    return 0


_QWEN3_VARIANTS = {"qwen3-embedding:0.6b", "qwen3-embedding:4b", "qwen3-embedding:8b"}
_QWEN3_VARIANT_MAP = {
    "qwen3-embedding:0.6b": "0.6b",
    "qwen3-embedding:4b": "4b",
    "qwen3-embedding:8b": "8b",
}


def _build_embedder(embedder_kind: str, model, host: str, base_url: str, api_token: str):
    """Construct an embedder from CLI options.

    Returns ``(embedder, model_name, tokenize_fn, backend_label, tokenizer_label)``.
    Shared by the ``index`` and ``query`` commands so both resolve the backend,
    model defaults, and tokenizer the same way.
    """
    model_name = model
    tokenize_fn = None  # set below after embedder is created

    if embedder_kind == "ollama":
        from kbcraft.embedders.ollama import OllamaEmbedder
        from kbcraft.tokenizer import get_tokenizer

        model_name = model_name or "nomic-embed-text"
        embedder = OllamaEmbedder(model=model_name, host=host)
        tok = get_tokenizer(model_name)
        tokenize_fn = tok.tokenize
        backend_label = f"ollama  host={host}"
        tokenizer_label = f"{tok.backend} ({tok.model_name})"

    else:  # openai
        if model_name in _QWEN3_VARIANTS:
            from kbcraft.embedders.qwen import Qwen3Embedder

            model_name = model_name or list(_QWEN3_VARIANTS)[0]
            variant = _QWEN3_VARIANT_MAP[model_name]
            embedder = Qwen3Embedder(variant=variant, base_url=base_url, token=api_token)
            _ = embedder.tokenizer  # load now for accurate timing later
            tokenize_fn = lambda text: embedder.tokenizer.tokenize(text)  # noqa: E731
            tokenizer_label = f"transformers ({model_name})"
        elif base_url:
            from kbcraft.embedder import OpenAICompatibleEmbedder
            from kbcraft.tokenizer import WhitespaceTokenizer

            model_name = model_name or "nomic-embed-text"
            embedder = OpenAICompatibleEmbedder(
                model=model_name, token=api_token, base_url=base_url
            )
            tok = WhitespaceTokenizer(model=model_name)
            tokenize_fn = tok.tokenize
            tokenizer_label = "whitespace (approximate)"
        else:
            from kbcraft.embedders.openai import OpenAIEmbedder

            model_name = model_name or "text-embedding-3-small"
            embedder = OpenAIEmbedder(model=model_name, token=api_token)
            # Chunk with the model's real tiktoken tokenizer so chunk token
            # counts are exact and no chunk can overflow the model's context
            # window (which would force the embedder to sub-split and break the
            # one-vector-per-chunk contract). encode() returns token ids; len()
            # of that list is the exact token count the Chunker needs.
            _enc = embedder.tokenizer
            tokenize_fn = lambda text: _enc.encode(text)  # noqa: E731
            tokenizer_label = f"tiktoken ({model_name})"

        backend_label = f"openai  base_url={base_url or 'https://api.openai.com/v1'}"

    return embedder, model_name, tokenize_fn, backend_label, tokenizer_label


def _cmd_index(args: argparse.Namespace) -> int:
    import json
    import os
    import time

    api_token = os.environ.get("KBCRAFT_API_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.is_dir():
        print(f"error: '{source_dir}' is not a directory", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    def log(msg: str) -> None:
        print(msg, flush=True)

    def log_section(title: str) -> None:
        print(f"\n{'─' * 60}", flush=True)
        print(f"  {title}", flush=True)
        print("─" * 60, flush=True)

    # ── Imports ───────────────────────────────────────────────────────────────
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        print(f"error: {exc}\nInstall with: pip install faiss-cpu numpy", file=sys.stderr)
        return 1

    from kbcraft.chunker import Chunker

    # ── Build embedder ────────────────────────────────────────────────────────
    embedder, model_name, tokenize_fn, backend_label, tokenizer_label = _build_embedder(
        args.embedder, args.model, args.host, args.base_url, api_token
    )

    log("\n" + "=" * 60)
    log("  kbcraft — build FAISS index")
    log("=" * 60)
    log(f"  Source dir   : {source_dir}")
    log(f"  Output dir   : {output_dir}")
    log(f"  Backend      : {backend_label}")
    log(f"  Model        : {model_name}")
    log(f"  Tokenizer    : {tokenizer_label}")
    log(f"  Chunk size   : {args.chunk_size} tokens")
    log(f"  Chunk overlap: {args.chunk_overlap} tokens")

    # ── 1. Select files ───────────────────────────────────────────────────────
    log_section("1. Selecting files")

    include_patterns = None
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

    kbignore_path = Path(args.kbignore) if args.kbignore else source_dir / ".kbignore"
    file_filter = FileFilter.from_kbignore(
        kbignore_path=kbignore_path,
        include_patterns=include_patterns,
        extra_excludes=args.exclude_patterns,
    )
    files = file_filter.collect_files(source_dir)

    if not files:
        log("  No files matched. Check --lang / --include / --exclude.")
        return 1

    log(f"  Found {len(files)} file(s):")
    for f in files:
        log(f"    {f.relative_to(source_dir)}")

    # ── 2. Chunk files ────────────────────────────────────────────────────────
    log_section("2. Chunking files")

    chunker = Chunker(
        max_chunk_tokens=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        prepend_source=True,
        tokenize=tokenize_fn,
    )
    t0 = time.time()
    chunks = chunker.chunk_files(files, base_dir=source_dir)
    log(f"  {len(chunks)} chunks from {len(files)} file(s)  ({time.time() - t0:.1f}s)")

    if not chunks:
        log("  No chunks produced — files may be empty.")
        return 1

    texts = [c.text for c in chunks]

    # ── 3. Embed chunks ───────────────────────────────────────────────────────
    log_section("3. Embedding chunks")

    dim = embedder.embedding_dim
    log(f"  Embedding dim: {dim}")
    log(f"  Embedding {len(texts)} chunk(s) …")

    t0 = time.time()
    try:
        # encode_documents applies any document-side instruction prefix the
        # backend defines (e.g. nomic/e5 "search_document: "); it falls back to
        # encode() for backends without asymmetric prefixes (OpenAI).
        vectors = embedder.encode_documents(texts)
    except Exception as exc:
        print(f"error: embedding failed: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    log(f"  Done  ({elapsed:.1f}s, {len(texts) / max(elapsed, 0.001):.1f} chunks/s)")

    if len(vectors) != len(chunks):
        # The pipeline's contract is one vector per chunk. A mismatch means a
        # chunk exceeded the model's context window and the embedder sub-split
        # it, which would misalign chunk metadata from index vectors. Chunks are
        # sized with the model's own tokenizer to prevent this, so a mismatch
        # here is a real error rather than something to paper over.
        print(
            f"error: embedder returned {len(vectors)} vectors for {len(chunks)} "
            "chunks. A chunk exceeded the model context window and was split. "
            "Lower --chunk-size so every chunk fits the model limit.",
            file=sys.stderr,
        )
        return 1

    # ── 4. Build FAISS index ──────────────────────────────────────────────────
    log_section("4. Building FAISS index")

    matrix = np.array(vectors, dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)
    log(f"  IndexFlatL2  dim={dim}  vectors={index.ntotal}")

    # ── 5. Save to disk ───────────────────────────────────────────────────────
    log_section("5. Saving to disk")

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / f"{args.name}.faiss"
    faiss.write_index(index, str(index_path))
    log(f"  Wrote {index_path}")

    chunks_path = output_dir / f"{args.name}_chunks.json"
    chunks_path.write_text(
        json.dumps(
            [
                {
                    "text": c.text,
                    "source": c.source,
                    "index": c.index,
                    "token_count": c.token_count,
                }
                for c in chunks
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"  Wrote {chunks_path}")

    meta_path = output_dir / f"{args.name}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "embedder": args.embedder,
                "embedding_dim": dim,
                "total_chunks": len(chunks),
                "total_files": len(files),
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "files": [str(f.relative_to(source_dir)) for f in files],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"  Wrote {meta_path}")

    log("\n" + "=" * 60)
    log(f"  Index built   ({len(chunks)} chunks, dim={dim})")
    log(f"  Output: {output_dir}/")
    log("=" * 60 + "\n")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    import json
    import os

    api_token = os.environ.get("KBCRAFT_API_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

    index_dir = Path(args.index_dir)
    index_path = index_dir / f"{args.name}.faiss"
    chunks_path = index_dir / f"{args.name}_chunks.json"

    for path in (index_path, chunks_path):
        if not path.is_file():
            print(f"error: '{path}' not found — build it with 'kbcraft index'", file=sys.stderr)
            return 1

    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        print(f"error: {exc}\nInstall with: pip install faiss-cpu numpy", file=sys.stderr)
        return 1

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    embedder, model_name, _, backend_label, _ = _build_embedder(
        args.embedder, args.model, args.host, args.base_url, api_token
    )

    index = faiss.read_index(str(index_path))

    # In --json mode stdout must carry only the JSON payload, so decorative
    # logging goes to stderr.
    info = sys.stderr if args.json else sys.stdout

    print("\n" + "=" * 60, file=info, flush=True)
    print("  kbcraft — query FAISS index", file=info, flush=True)
    print("=" * 60, file=info, flush=True)
    print(f"  Index      : {index_path}", file=info, flush=True)
    print(f"  Backend    : {backend_label}", file=info, flush=True)
    print(f"  Model      : {model_name}", file=info, flush=True)
    print(f"  Vectors    : {index.ntotal}", file=info, flush=True)
    print(f"  Query      : {args.query!r}", file=info, flush=True)
    print(f"  Top-k      : {args.k}", file=info, flush=True)

    try:
        # encode_query applies any query-side instruction prefix the backend
        # defines (e.g. "search_query: "); falls back to encode() for OpenAI.
        q_vec = np.array([embedder.encode_query(args.query)], dtype=np.float32)
    except Exception as exc:
        print(f"error: embedding query failed: {exc}", file=sys.stderr)
        return 1

    k = min(args.k, index.ntotal)
    distances, ids = index.search(q_vec, k=k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], ids[0]), start=1):
        idx = int(idx)
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        source = chunk.get("source") or ""
        results.append(
            {
                "rank": rank,
                "id": idx,
                "source": source,
                "l2": float(dist),
                "preview": chunk["text"][:200],
            }
        )

    if args.json:
        print(
            json.dumps(
                {"query": args.query, "k": k, "count": len(results), "results": results},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    print(f"\n  Top-{k} results:", flush=True)
    for r in results:
        name = Path(r["source"]).name if r["source"] else "?"
        preview = r["preview"][:80].replace("\n", " ")
        print(f"    {r['rank']}. [{name}] (L2={r['l2']:.4f})", flush=True)
        print(f"       {preview!r}", flush=True)

    # Stable machine-readable summary line for scripts/CI.
    print(f"\n  Query returned {len(results)} result(s).", flush=True)
    return 0


def main() -> None:
    """Main entry point for the kbcraft CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        sys.exit(_cmd_collect(args))
    elif args.command == "presets":
        sys.exit(_cmd_presets())
    elif args.command == "index":
        sys.exit(_cmd_index(args))
    elif args.command == "query":
        sys.exit(_cmd_query(args))
    elif args.command == "config":
        if getattr(args, "config_command", None) == "apply":
            sys.exit(_cmd_config_apply(args))
        parser.parse_args([args.command, "--help"])
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
