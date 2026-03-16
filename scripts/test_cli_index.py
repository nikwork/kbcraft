#!/usr/bin/env python3
"""
End-to-end CLI smoke test for `kbcraft index`.

Runs the full pipeline via the CLI binary (not Python imports):
  kbcraft index ./kb --embedder ollama --model all-minilm --output <tmp>

Checks:
  1. CLI exits with code 0
  2. index.faiss, index_chunks.json, index_meta.json are written
  3. FAISS index loads and has the expected vector count
  4. A smoke-test query returns ranked results

Run via Docker Compose (after `docker compose up -d`):
    docker compose run --rm kbcraft python scripts/test_cli_index.py

Run locally (requires `ollama serve` with all-minilm pulled):
    python scripts/test_cli_index.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("TEST_MODEL", "all-minilm")
SOURCE_DIR = PROJECT_ROOT / "kb"

PASS = "  \033[32m✓\033[0m"
FAIL = "  \033[31m✗\033[0m"


def log(msg: str) -> None:
    print(msg, flush=True)


def log_section(title: str) -> None:
    print(f"\n{'─' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print("─" * 60, flush=True)


def check(label: str, condition: bool) -> bool:
    print(f"{PASS if condition else FAIL}  {label}")
    return condition


def main() -> int:
    log("\n" + "=" * 60)
    log("  kbcraft — CLI index smoke test")
    log("=" * 60)
    log(f"  Source dir   : {SOURCE_DIR}")
    log(f"  Ollama host  : {OLLAMA_HOST}")
    log(f"  Model        : {MODEL}")

    failures = 0
    output_dir = Path(tempfile.mkdtemp(prefix="kbcraft_test_"))
    log(f"  Output dir   : {output_dir}")

    try:
        # ── 1. Run the CLI ─────────────────────────────────────────────────────
        log_section("1. Running kbcraft index")

        cmd = [
            "kbcraft", "index", str(SOURCE_DIR),
            "--embedder", "ollama",
            "--model", MODEL,
            "--host", OLLAMA_HOST,
            "--lang", "markdown",
            "--output", str(output_dir),
            "--name", "test_index",
            "--chunk-size", "200",
            "--chunk-overlap", "20",
        ]
        log(f"  $ {' '.join(cmd)}\n")

        result = subprocess.run(cmd, capture_output=False, text=True)

        log("")
        ok = check("exit code is 0", result.returncode == 0)
        if not ok:
            failures += 1
            log(f"\n  Command failed — aborting remaining checks.")
            return 1

        # ── 2. Output files ────────────────────────────────────────────────────
        log_section("2. Output files")

        index_file  = output_dir / "test_index.faiss"
        chunks_file = output_dir / "test_index_chunks.json"
        meta_file   = output_dir / "test_index_meta.json"

        for path in (index_file, chunks_file, meta_file):
            if not check(f"{path.name} exists", path.exists()):
                failures += 1

        # ── 3. Validate chunks.json ────────────────────────────────────────────
        log_section("3. chunks.json")

        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        failures += 0 if check(f"contains {len(chunks)} chunks (> 0)", len(chunks) > 0) else 1

        first = chunks[0]
        for field in ("text", "source", "index", "token_count"):
            failures += 0 if check(f"chunk has field '{field}'", field in first) else 1

        # ── 4. Validate meta.json ──────────────────────────────────────────────
        log_section("4. meta.json")

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        failures += 0 if check(f"model == '{MODEL}'", meta.get("model") == MODEL) else 1
        failures += 0 if check("embedding_dim > 0", meta.get("embedding_dim", 0) > 0) else 1
        failures += 0 if check(f"total_chunks == {len(chunks)}", meta.get("total_chunks") == len(chunks)) else 1

        dim = meta["embedding_dim"]

        # ── 5. Load FAISS index ────────────────────────────────────────────────
        log_section("5. FAISS index")

        try:
            import faiss
            import numpy as np

            index = faiss.read_index(str(index_file))
            failures += 0 if check(f"ntotal == {len(chunks)}", index.ntotal == len(chunks)) else 1
            failures += 0 if check(f"d == {dim}", index.d == dim) else 1

            # ── 6. Smoke-test query ────────────────────────────────────────────
            log_section("6. Smoke-test query")

            from kbcraft.embedders.ollama import OllamaEmbedder

            embedder = OllamaEmbedder(model=MODEL, host=OLLAMA_HOST)
            query = "how does the chunker split markdown files"
            log(f"  Query: '{query}'")

            q_vec = np.array([embedder.encode_query(query)], dtype=np.float32)
            distances, ids = index.search(q_vec, k=3)

            failures += 0 if check("search returns k=3 results", len(ids[0]) == 3) else 1
            failures += 0 if check("all result ids are valid", all(0 <= i < len(chunks) for i in ids[0])) else 1

            log("\n  Top-3 results:")
            for rank, (dist, idx) in enumerate(zip(distances[0], ids[0]), start=1):
                chunk = chunks[idx]
                source = Path(chunk["source"]).name if chunk.get("source") else "?"
                preview = chunk["text"][:80].replace("\n", " ")
                log(f"    {rank}. [{source}] (L2={dist:.4f})")
                log(f"       {preview!r}")

        except ImportError:
            log("  (faiss not available — skipping index and query checks)")

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    if failures == 0:
        log("  All checks passed ✓")
    else:
        log(f"  {failures} check(s) FAILED ✗")
    log("=" * 60 + "\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
