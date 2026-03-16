#!/usr/bin/env python3
"""
Build a FAISS index from the kbcraft source base using Qwen3 embeddings.

Runs the full kbcraft pipeline:
  FileSelector → Chunker → Qwen3Embedder → FAISS index

The Qwen3Embedder uses the official Qwen3 transformers tokenizer for exact
token counting and automatically splits any chunk that exceeds the model's
context window.  Batches are sized dynamically by token budget, so no manual
batch_size tuning is required.

Config is loaded from configs/embedding.yaml and configs/vector_store.yaml.
The active model must be one of the qwen3-embedding variants; override with
the MODEL env var (e.g. MODEL=qwen3-embedding:4b).

Connection settings come from the openai_compatible backend section:
  OPENAI_COMPATIBLE_BASE_URL  — default: http://localhost:11434/v1
  OPENAI_COMPATIBLE_TOKEN     — default: (empty, for local servers)

Run via Ollama (after `ollama pull qwen3-embedding:0.6b && ollama serve`):
    python scripts/build_faiss_index_qwen.py

Run with a custom endpoint:
    OPENAI_COMPATIBLE_BASE_URL=http://gpu-box:11434/v1 \\
    MODEL=qwen3-embedding:8b \\
    python scripts/build_faiss_index_qwen.py

Output (written to faiss.output_dir from vector_store.yaml):
    index.faiss   — FAISS flat-L2 index, ready for similarity search
    chunks.json   — chunk texts + metadata (source, index, token_count)
    meta.json     — build metadata (model, dim, total chunks, file list)
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Helpers ───────────────────────────────────────────────────────────────────

INCLUDE_PRESETS = ["python", "markdown", "yaml", "toml"]
EXCLUDE_PATTERNS = [
    "__pycache__/**",
    ".git/**",
    ".github/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
    ".venv/**",
    "venv/**",
    "htmlcov/**",
    "dist/**",
    "build/**",
    "faiss_index/**",
    "poetry.lock",
]

_QWEN3_VARIANT_MAP = {
    "qwen3-embedding:0.6b": "0.6b",
    "qwen3-embedding:4b":   "4b",
    "qwen3-embedding:8b":   "8b",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def log_section(title: str) -> None:
    print(f"\n{'─' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print("─" * 60, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import faiss
    import numpy as np

    from kbcraft.chunker import Chunker
    from kbcraft.config import ConfigFactory
    from kbcraft.embedders.qwen import Qwen3Embedder
    from kbcraft.selector import FileFilter

    factory = ConfigFactory.from_project_root(PROJECT_ROOT)
    emb = factory.load_embedding()
    vs = factory.load_vector_store()

    model = emb.model
    oac = emb.openai_compatible

    if model.name not in _QWEN3_VARIANT_MAP:
        log(
            f"ERROR: active model {model.name!r} is not a Qwen3 embedding variant.\n"
            f"Set MODEL to one of: {list(_QWEN3_VARIANT_MAP)}"
        )
        sys.exit(1)

    variant = _QWEN3_VARIANT_MAP[model.name]

    faiss_cfg = vs.faiss
    output_dir = faiss_cfg.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    log("\n" + "=" * 60)
    log("  kbcraft — build FAISS index  (Qwen3 embeddings)")
    log("=" * 60)
    log(f"  Project root : {PROJECT_ROOT}")
    log(f"  Endpoint     : {oac.base_url}")
    log(f"  Model        : {model.name}  (variant={variant})")
    log(f"  Tokenizer    : transformers / {model.hf_repo}")
    log(f"  Max tokens   : {model.chunking.max_tokens}")
    log(f"  Overlap      : {model.chunking.overlap}")
    log(f"  Output dir   : {output_dir}")

    # ── 1. Select files ───────────────────────────────────────────────────────
    log_section("1. Selecting files")

    file_filter = FileFilter.from_presets(
        INCLUDE_PRESETS,
        exclude_patterns=EXCLUDE_PATTERNS,
    )
    files = file_filter.collect_files(PROJECT_ROOT)

    if not files:
        log("  No files matched — check INCLUDE_PRESETS and EXCLUDE_PATTERNS.")
        sys.exit(1)

    log(f"  Found {len(files)} files:")
    for f in files:
        log(f"    {f.relative_to(PROJECT_ROOT)}")

    # ── 2. Initialise embedder (loads tokenizer) ──────────────────────────────
    log_section("2. Loading Qwen3 tokenizer")

    embedder = Qwen3Embedder(
        variant=variant,
        base_url=oac.base_url,
        token=oac.token,
    )
    # Force tokenizer load now so the timing below reflects only chunking.
    _ = embedder.tokenizer
    log(f"  Loaded tokenizer from {model.hf_repo}")
    log(f"  Embedding dim: {embedder.embedding_dim}")

    # ── 3. Chunk files ────────────────────────────────────────────────────────
    log_section("3. Chunking files")

    chunker = Chunker(
        max_chunk_tokens=model.chunking.max_tokens,
        chunk_overlap=model.chunking.overlap,
        prepend_source=model.chunking.prepend_source,
        # Use the exact Qwen3 tokenizer so chunk boundaries are accurate.
        tokenize=lambda text: embedder.tokenizer.tokenize(text),
    )
    t0 = time.time()
    chunks = chunker.chunk_files(files, base_dir=PROJECT_ROOT)
    log(f"  {len(chunks)} chunks from {len(files)} files  ({time.time() - t0:.1f}s)")

    texts = [c.text for c in chunks]

    # ── 4. Embed chunks ───────────────────────────────────────────────────────
    log_section("4. Embedding chunks")

    log(f"  Embedding {len(texts)} chunks  (dynamic token-budget batching) …")

    t0 = time.time()
    try:
        vectors = embedder.encode(texts)
    except Exception as exc:
        log(f"  ERROR during embedding: {exc}")
        sys.exit(1)

    elapsed = time.time() - t0
    log(f"  Done  ({elapsed:.1f}s, {len(texts) / elapsed:.1f} chunks/s)")

    if len(vectors) != len(chunks):
        # encode() may return more vectors than chunks when the embedder
        # sub-chunked an oversized text.  Truncate metadata to match.
        log(
            f"  Note: embedder produced {len(vectors)} vectors for {len(chunks)} chunks "
            "(some chunks were further split by the tokenizer)."
        )
        chunks = chunks[: len(vectors)]

    # ── 5. Build FAISS index ──────────────────────────────────────────────────
    log_section("5. Building FAISS index")

    dim = embedder.embedding_dim
    matrix = np.array(vectors, dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)
    log(f"  IndexFlatL2  dim={dim}  vectors={index.ntotal}")

    # ── 6. Save to disk ───────────────────────────────────────────────────────
    log_section("6. Saving to disk")

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    log(f"  Wrote {index_path}")

    chunks_path = output_dir / "chunks.json"
    chunks_data = [
        {
            "text": c.text,
            "source": c.source,
            "index": c.index,
            "token_count": c.token_count,
        }
        for c in chunks
    ]
    chunks_path.write_text(
        json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"  Wrote {chunks_path}")

    meta_path = output_dir / "meta.json"
    meta = {
        "model": model.name,
        "variant": variant,
        "hf_repo": model.hf_repo,
        "endpoint": oac.base_url,
        "embedding_dim": dim,
        "total_chunks": len(chunks),
        "total_files": len(files),
        "tokenizer_backend": "transformers",
        "max_chunk_tokens": model.chunking.max_tokens,
        "chunk_overlap": model.chunking.overlap,
        "files": [str(f.relative_to(PROJECT_ROOT)) for f in files],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  Wrote {meta_path}")

    # ── 7. Smoke-test: query the index ────────────────────────────────────────
    log_section("7. Smoke-test query")

    query = "how does the chunker split markdown files"
    log(f"  Query: '{query}'")

    q_vec = np.array([embedder.encode_query(query)], dtype=np.float32)
    distances, ids = index.search(q_vec, k=3)

    log("  Top-3 results:")
    for rank, (dist, idx) in enumerate(zip(distances[0], ids[0]), start=1):
        chunk = chunks[idx]
        rel_source = Path(chunk.source).relative_to(PROJECT_ROOT) if chunk.source else "?"
        preview = chunk.text[:80].replace("\n", " ")
        log(f"    {rank}. [{rel_source}] (L2={dist:.4f})")
        log(f"       {preview!r}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log(f"  Index built   ({len(chunks)} chunks, dim={dim})")
    log(f"  Output: {output_dir}/")
    log("=" * 60 + "\n")


if __name__ == "__main__":
    main()
