#!/usr/bin/env python3
"""
Build a FAISS index from the kbcraft source base.

Runs the full kbcraft pipeline:
  FileSelector → Chunker → OllamaEmbedder → FAISS index

Run via Docker Compose (after `docker compose up -d`):
    docker compose run --rm kbcraft python scripts/build_faiss_index.py

Run locally (requires Ollama running):
    python scripts/build_faiss_index.py

Output (written to <project_root>/faiss_index/):
    index.faiss   — FAISS flat-L2 index, ready for similarity search
    chunks.json   — chunk texts + metadata (source, index, token_count)
    meta.json     — build metadata (model, dim, total chunks, file list)

Environment variables:
    OLLAMA_HOST   — Ollama base URL (default: http://localhost:11434)
    MODEL         — embedding model name   (default: all-minilm)
    MAX_TOKENS    — max tokens per chunk   (default: 200, counted by HF tokenizer when available)
    CHUNK_OVERLAP — overlap tokens         (default: 20)
    OUTPUT_DIR    — output directory path  (default: <project_root>/faiss_index)
"""

import json
import os
import sys
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("MODEL", "all-minilm")
# 100 words ≈ 120-150 subword tokens — well within all-minilm's 256-token
# context window even after the File/Path header is prepended.
# Override via MAX_TOKENS env var for larger-context models.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "10"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(PROJECT_ROOT / "faiss_index")))

# File types to index from the source base
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def log_section(title: str) -> None:
    print(f"\n{'─' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print("─" * 60, flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _embed_one(chunk, text, embedder, tok, good_chunks, vectors, max_retries=5):
    """Embed a single chunk with retries and automatic truncation.

    Returns True if embedded (or truncated and embedded), False if skipped.

    - Transient connection failures are retried with exponential back-off.
    - HTTP 400 "input length exceeds" errors shrink the text by 20% each
      retry, working around tokenizer count vs. Ollama count mismatches.
    """
    embed_text = text
    for attempt in range(max_retries):
        try:
            vec = embedder.encode_documents([embed_text])[0]
            good_chunks.append(chunk)
            vectors.append(vec)
            return True
        except RuntimeError as exc:
            if "input length exceeds" in str(exc):
                # Truncate by chars (not tokens) to avoid WordPiece garbling
                embed_text = embed_text[: int(len(embed_text) * 0.8)]
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                log(f"  SKIP chunk {chunk.index} from {Path(chunk.source).name}"
                    f" ({len(text.split())} words): {exc}")
                return False
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                log(f"  SKIP chunk {chunk.index} from {Path(chunk.source).name}"
                    f" ({len(text.split())} words): {exc}")
                return False
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import faiss
    import numpy as np

    from kbcraft.chunker import Chunker
    from kbcraft.embedders.ollama import OllamaEmbedder
    from kbcraft.selector import FileFilter
    from kbcraft.tokenizer import get_tokenizer

    log("\n" + "=" * 60)
    log("  kbcraft — build FAISS index from source base")
    log("=" * 60)
    tok = get_tokenizer(MODEL, ollama_host=OLLAMA_HOST)

    log(f"  Project root : {PROJECT_ROOT}")
    log(f"  Ollama host  : {OLLAMA_HOST}")
    log(f"  Model        : {MODEL}")
    log(f"  Tokenizer    : {tok.backend} ({tok.model_name})")
    log(f"  Max tokens   : {MAX_TOKENS}")
    log(f"  Overlap      : {CHUNK_OVERLAP}")
    log(f"  Output dir   : {OUTPUT_DIR}")

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
        rel = f.relative_to(PROJECT_ROOT)
        log(f"    {rel}")

    # ── 2. Chunk files ────────────────────────────────────────────────────────
    log_section("2. Chunking files")

    chunker = Chunker(
        max_chunk_tokens=MAX_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        prepend_source=True,
        tokenize=tok.tokenize,
    )
    t0 = time.time()
    chunks = chunker.chunk_files(files, base_dir=PROJECT_ROOT)
    log(f"  {len(chunks)} chunks from {len(files)} files  ({time.time() - t0:.1f}s)")

    texts = [c.text for c in chunks]

    # ── 3. Embed chunks ───────────────────────────────────────────────────────
    log_section("3. Embedding chunks")

    embedder = OllamaEmbedder(model=MODEL, host=OLLAMA_HOST)
    dim = embedder.embedding_dim
    log(f"  Model dim    : {dim}")
    log(f"  Embedding {len(texts)} chunks …")

    t0 = time.time()
    good_chunks = []
    vectors: list = []
    skipped = 0
    batch_size = embedder._batch_size

    for i in range(0, len(texts), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [c.text for c in batch_chunks]
        try:
            batch_vecs = embedder.encode_documents(batch_texts)
            good_chunks.extend(batch_chunks)
            vectors.extend(batch_vecs)
        except Exception:
            # Retry one-by-one to salvage the good chunks in this batch
            time.sleep(2)
            for chunk, text in zip(batch_chunks, batch_texts):
                if not _embed_one(chunk, text, embedder, tok, good_chunks, vectors):
                    skipped += 1
        if (i // batch_size) % 4 == 0:
            log(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks …")

    elapsed = time.time() - t0
    if skipped:
        log(f"  Done  ({elapsed:.1f}s) — {len(good_chunks)} embedded, {skipped} skipped")
    else:
        log(f"  Done  ({elapsed:.1f}s, {len(texts) / elapsed:.1f} chunks/s)")

    chunks = good_chunks

    # ── 4. Build FAISS index ──────────────────────────────────────────────────
    log_section("4. Building FAISS index")

    matrix = np.array(vectors, dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)
    log(f"  IndexFlatL2  dim={dim}  vectors={index.ntotal}")

    # ── 5. Save to disk ───────────────────────────────────────────────────────
    log_section("5. Saving to disk")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_path = OUTPUT_DIR / "index.faiss"
    faiss.write_index(index, str(index_path))
    log(f"  Wrote {index_path}")

    chunks_path = OUTPUT_DIR / "chunks.json"
    chunks_data = [
        {
            "text": c.text,
            "source": c.source,
            "index": c.index,
            "token_count": c.token_count,
        }
        for c in chunks
    ]
    chunks_path.write_text(json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  Wrote {chunks_path}")

    meta_path = OUTPUT_DIR / "meta.json"
    meta = {
        "model": MODEL,
        "ollama_host": OLLAMA_HOST,
        "embedding_dim": dim,
        "total_chunks": len(chunks),
        "total_files": len(files),
        "tokenizer_backend": tok.backend,
        "max_chunk_tokens": MAX_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
        "files": [str(f.relative_to(PROJECT_ROOT)) for f in files],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  Wrote {meta_path}")

    # ── 6. Smoke-test: query the index ────────────────────────────────────────
    log_section("6. Smoke-test query")

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
    log(f"  Index built ✓  ({len(chunks)} chunks, dim={dim})")
    log(f"  Output: {OUTPUT_DIR}/")
    log("=" * 60 + "\n")


if __name__ == "__main__":
    main()
