#!/usr/bin/env python3
"""
Smoke test for OllamaEmbedder against a live Ollama server.

Run locally:
    python scripts/test_ollama.py

Run via Docker Compose (after `docker compose up -d`):
    docker compose run kbcraft python scripts/test_ollama.py

Requires:
    - Ollama server running and reachable
    - all-minilm model pulled:  ollama pull all-minilm
"""

import os
import sys

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def check(label: str, passed: bool, detail: str = "") -> None:
    mark = "✓" if passed else "✗"
    line = f"  {mark}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not passed:
        sys.exit(1)


def main() -> None:
    from kbcraft.embedders.ollama import OllamaEmbedder

    print("\n" + "=" * 60)
    print("  kbcraft — OllamaEmbedder smoke test")
    print("=" * 60)
    print(f"  Ollama host : {OLLAMA_HOST}")
    print("  Model       : all-minilm")

    embedder = OllamaEmbedder(model="all-minilm", host=OLLAMA_HOST)

    # ── 1. embedding_dim ──────────────────────────────────────────────
    section("1. Embedding dimension")
    dim = embedder.embedding_dim
    check("embedding_dim == 384", dim == 384, f"got {dim}")

    # ── 2. encode — plain batch ───────────────────────────────────────
    section("2. encode() — plain batch")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    ]
    vecs = embedder.encode(texts)
    check("returns 3 vectors", len(vecs) == 3)
    check("each vector has 384 dims", all(len(v) == 384 for v in vecs))
    check("all values are floats", all(isinstance(x, float) for v in vecs for x in v))

    # ── 3. encode_query ───────────────────────────────────────────────
    section("3. encode_query()")
    q_vec = embedder.encode_query("how to handle errors in Python")
    check("returns a single vector", isinstance(q_vec, list))
    check("has 384 dims", len(q_vec) == 384)

    # ── 4. encode_documents ───────────────────────────────────────────
    section("4. encode_documents()")
    docs = [
        "try:\\n    risky_call()\\nexcept Exception as e:\\n    handle(e)",
        "Use try/except to catch exceptions in Python.",
    ]
    doc_vecs = embedder.encode_documents(docs)
    check("returns 2 vectors", len(doc_vecs) == 2)
    check("each has 384 dims", all(len(v) == 384 for v in doc_vecs))

    # ── 5. semantic similarity (query vs. related doc) ────────────────
    section("5. Semantic similarity")

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    q = embedder.encode_query("error handling Python")
    related = embedder.encode_documents(["try: ... except Exception: ..."])[0]
    unrelated = embedder.encode_documents(["The Eiffel Tower is in Paris."])[0]

    sim_related = cosine(q, related)
    sim_unrelated = cosine(q, unrelated)
    check(
        "related pair scores higher than unrelated",
        sim_related > sim_unrelated,
        f"related={sim_related:.3f}  unrelated={sim_unrelated:.3f}",
    )

    # ── 6. Batching ───────────────────────────────────────────────────
    section("6. Auto-batching (batch_size=2)")
    small_batcher = OllamaEmbedder(model="all-minilm", host=OLLAMA_HOST, batch_size=2)
    batch_vecs = small_batcher.encode(["a", "b", "c", "d", "e"])
    check("5 texts → 5 vectors", len(batch_vecs) == 5)

    # ── 7. ChromaDB adapter ───────────────────────────────────────────
    section("7. ChromaDB adapter (as_chroma_ef)")
    ef = embedder.as_chroma_ef()
    ef_result = ef(["hello chromadb"])
    check("returns list of vectors", isinstance(ef_result, list))
    check("vector has 384 dims", len(ef_result[0]) == 384)

    # ── 8. FAISS adapter ──────────────────────────────────────────────
    section("8. FAISS adapter (as_faiss_matrix)")
    try:
        import numpy as np

        matrix = embedder.as_faiss_matrix(["hello", "world"])
        check("shape is (2, 384)", matrix.shape == (2, 384), str(matrix.shape))
        check("dtype is float32", matrix.dtype == np.float32, str(matrix.dtype))
    except ImportError:
        print("  ⊘  numpy not installed — skipping FAISS test")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  All checks passed ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
