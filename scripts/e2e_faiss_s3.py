#!/usr/bin/env python3
"""
End-to-end test for `kbcraft index` + S3 (MinIO), using OpenAI embeddings.

Pipeline:
  1. Ensure target bucket exists on MinIO (boto3)
  2. Seed sample Markdown docs under ./.e2e/docs
  3. Run `kbcraft index ... --embedder openai --model text-embedding-3-small`
  4. Validate output files (.faiss, _chunks.json, _meta.json)
  5. Validate chunks.json schema
  6. Validate meta.json (model, embedding_dim, total_chunks)
  7. Load the FAISS index and assert ntotal / d
  8. Smoke-test query — encode with OpenAIEmbedder, search top-3
  9. Upload artifacts to s3://<BUCKET>/ via boto3
 10. List the bucket and assert all three artifacts are present

Designed to run **inside the dev container** (kbcraft-dev), where:
  - MinIO is reachable as http://minio:9000 (sibling compose service)
  - .env is auto-loaded into the environment by docker-compose.dev.yml
  - The project venv is on PATH (kbcraft binary callable directly)

Run inside the container:
    python scripts/e2e_faiss_s3.py

From the host (delegates into the container):
    docker exec kbcraft-dev python /app/scripts/e2e_faiss_s3.py

Persist output instead of using a temp dir:
    SAVE_VECTORDB=1 python scripts/e2e_faiss_s3.py
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SOURCE_DIR = PROJECT_ROOT

# ── config ────────────────────────────────────────────────────────────────────
MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
MAX_TOKENS = int(os.environ.get("OPENAI_EMBEDDING_MAX_TOKENS", "8191"))
EXPECTED_DIM = int(os.environ.get("KBCRAFT_E2E_EMBEDDING_DIM", "1536"))

BUCKET = os.environ.get("KBCRAFT_E2E_BUCKET", "kbcraft-e2e")
INDEX_NAME = os.environ.get("KBCRAFT_E2E_INDEX_NAME", "test_index_e2e")

DOCS_DIR = PROJECT_ROOT / ".e2e" / "docs"

# When set, the index output is written under vectordb/ and not cleaned up.
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
SAVE_OUTPUT = os.environ.get("SAVE_VECTORDB", "").lower() in ("1", "true", "yes")

# MinIO defaults match docker-compose.dev.yml. Inside the dev container the
# sibling host is "minio"; from the host it's "localhost". Whichever value is
# already set in the environment wins.
AWS_ACCESS_KEY_ID = os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_DEFAULT_REGION = os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
S3_ENDPOINT = os.environ.setdefault("STORAGE_S3_ENDPOINT_URL", "http://minio:9000")


PASS = "  \033[32m✓\033[0m"
FAIL = "  \033[31m✗\033[0m"


INTRO_MD = """# Introduction

kbcraft is a CLI that turns Markdown notes into a RAG-ready vector index.
The pipeline: select files, chunk them with a tokenizer, embed the chunks,
and write a FAISS index plus chunk + metadata sidecars.
"""

USAGE_MD = """# Usage

Run `kbcraft index ./docs --output ./out --embedder openai
--model text-embedding-3-small` to build a FAISS flat-L2 index using the
OpenAI Embeddings API. The output directory will contain `index.faiss`,
`index_chunks.json`, and `index_meta.json`.
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def log_section(title: str) -> None:
    print(f"\n{'─' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print("─" * 60, flush=True)


def check(label: str, condition: bool) -> bool:
    print(f"{PASS if condition else FAIL}  {label}")
    return condition


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )


def wait_for_minio(timeout_s: int = 30) -> bool:
    url = f"{S3_ENDPOINT.rstrip('/')}/minio/health/ready"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError):
            pass
        time.sleep(1)
    return False


def ensure_bucket(s3, bucket: str) -> bool:
    try:
        s3.head_bucket(Bucket=bucket)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            s3.create_bucket(Bucket=bucket)
            return True
        return False


def seed_docs(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "intro.md").write_text(INTRO_MD)
    (docs_dir / "usage.md").write_text(USAGE_MD)


def log_token_distribution(chunks: list, buckets: int = 8) -> None:
    """Print min/median/mean/max plus a fixed-width text histogram of token_count."""
    counts = [c.get("token_count", 0) for c in chunks]
    if not counts:
        log("      (no chunks to summarize)")
        return

    lo, hi = min(counts), max(counts)
    mean = statistics.fmean(counts)
    median = statistics.median(counts)
    total = sum(counts)
    log(f"      total chunks  : {len(chunks)}")
    log(f"      total tokens  : {total}")
    log(f"      token_count   : min={lo}  median={median:g}  mean={mean:.1f}  max={hi}")

    char_counts = [len(c.get("text", "")) for c in chunks]
    log(
        "      text length   : "
        f"min={min(char_counts)}  median={statistics.median(char_counts):g}  "
        f"mean={statistics.fmean(char_counts):.1f}  max={max(char_counts)}  (chars)"
    )

    if lo == hi:
        log(f"      histogram     : all {len(chunks)} chunk(s) at {lo} tokens")
        return

    # Build a fixed-bucket histogram across [lo, hi].
    width = (hi - lo) / buckets
    edges = [lo + i * width for i in range(buckets + 1)]
    hist = [0] * buckets
    for n in counts:
        # Last bucket is inclusive on the right edge.
        idx = min(int((n - lo) / width), buckets - 1) if width > 0 else 0
        hist[idx] += 1
    peak = max(hist) or 1
    bar_w = 30
    log("      histogram     :")
    for i, h in enumerate(hist):
        bar = "█" * max(1, int(h * bar_w / peak)) if h else ""
        log(f"        [{edges[i]:>6.0f} – {edges[i + 1]:>6.0f}]  {h:>4}  {bar}")


def _normalize_source(raw: str) -> str:
    """Normalize a chunk source to match meta['files'] entries.

    meta['files'] stores paths relative to the index source dir, while
    chunks[*].source stores absolute paths. Strip the PROJECT_ROOT prefix
    when present, falling back to the basename otherwise.
    """
    if not raw:
        return ""
    p = Path(raw)
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.name


def log_input_files(meta: dict, chunks: list) -> None:
    """List embedded source files; annotate with chunk count when known."""
    files = meta.get("files") or []
    per_file: Counter = Counter(_normalize_source(c.get("source", "")) for c in chunks)
    log(f"      total_files in meta : {len(files)}")
    log(f"      sources in chunks   : {len(per_file)}")
    if not files and not per_file:
        log("      (no file list available)")
        return
    keys = files if files else sorted(per_file.keys())
    for fname in keys:
        n = per_file.get(fname, 0)
        log(f"        - {fname}   ({n} chunk{'s' if n != 1 else ''})")
    # Any source that appears in chunks but not in meta['files'].
    extras = sorted(set(per_file) - set(files))
    for fname in extras:
        log(f"        - {fname}   ({per_file[fname]} chunks, not in meta.files)")


def main() -> int:
    log("\n" + "=" * 60)
    log("  kbcraft — CLI index + S3 e2e (OpenAI embeddings)")
    log("=" * 60)
    log(f"  Source dir   : {DOCS_DIR}")
    log(f"  Model        : {MODEL}")
    log(f"  Max tokens   : {MAX_TOKENS}")
    log(f"  S3 endpoint  : {S3_ENDPOINT}")
    log(f"  Bucket       : {BUCKET}")

    if not os.environ.get("OPENAI_API_KEY"):
        log("\n  error: OPENAI_API_KEY is not set — required for OpenAI embeddings")
        return 1

    failures = 0
    if SAVE_OUTPUT:
        VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = VECTORDB_DIR
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="kbcraft_e2e_test_"))
    log(f"  Output dir   : {output_dir}")
    log(f"  Persist output: {SAVE_OUTPUT}")

    s3 = s3_client()

    try:
        # ── 1. Ensure MinIO bucket ──────────────────────────────────────────────
        log_section(f"1. Ensuring bucket s3://{BUCKET}")

        if not check(f"minio reachable at {S3_ENDPOINT}", wait_for_minio()):
            failures += 1
            log("\n  MinIO not ready — aborting remaining checks.")
            return 1

        if not check(f"bucket '{BUCKET}' exists or created", ensure_bucket(s3, BUCKET)):
            failures += 1
            log("\n  Bucket setup failed — aborting remaining checks.")
            return 1

        # ── 2. Seed sample docs ─────────────────────────────────────────────────
        log_section("2. Seeding sample docs")

        seed_docs(DOCS_DIR)
        seeded = sorted(DOCS_DIR.glob("*.md"))
        failures += 0 if check(f"{len(seeded)} markdown file(s) seeded", len(seeded) >= 2) else 1
        for path in seeded:
            log(f"      {path.relative_to(PROJECT_ROOT)}")

        # ── 3. Run the CLI ──────────────────────────────────────────────────────
        log_section("3. Running kbcraft index")

        cmd = [
            "kbcraft",
            "index",
            str(SOURCE_DIR),
            "--embedder",
            "openai",
            "--model",
            MODEL,
            "--lang",
            "markdown",
            "--lang",
            "python",
            "--exclude",
            ".*/**",  # skip every top-level dotted dir (.venv, .git, .e2e, …)
            "--output",
            str(output_dir),
            "--name",
            INDEX_NAME,
            "--chunk-size",
            str(MAX_TOKENS),
            "--chunk-overlap",
            str(MAX_TOKENS // 8),
        ]
        log(f"  $ {' '.join(cmd)}\n")

        result = subprocess.run(cmd, capture_output=False, text=True, cwd=PROJECT_ROOT)

        log("")
        ok = check("exit code is 0", result.returncode == 0)
        if not ok:
            failures += 1
            log("\n  Command failed — aborting remaining checks.")
            return 1

        # ── 4. Output files ─────────────────────────────────────────────────────
        log_section("4. Output files")

        index_file = output_dir / f"{INDEX_NAME}.faiss"
        chunks_file = output_dir / f"{INDEX_NAME}_chunks.json"
        meta_file = output_dir / f"{INDEX_NAME}_meta.json"

        for path in (index_file, chunks_file, meta_file):
            if not check(f"{path.name} exists", path.exists()):
                failures += 1

        # ── 5. Validate chunks.json ─────────────────────────────────────────────
        log_section("5. chunks.json")

        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        failures += 0 if check(f"contains {len(chunks)} chunks (> 0)", len(chunks) > 0) else 1

        first = chunks[0]
        for field in ("text", "source", "index", "token_count"):
            failures += 0 if check(f"chunk has field '{field}'", field in first) else 1

        log("\n  chunk length distribution:")
        log_token_distribution(chunks)

        # ── 6. Validate meta.json ───────────────────────────────────────────────
        log_section("6. meta.json")

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        failures += 0 if check(f"model == '{MODEL}'", meta.get("model") == MODEL) else 1
        failures += (
            0
            if check(f"embedding_dim == {EXPECTED_DIM}", meta.get("embedding_dim") == EXPECTED_DIM)
            else 1
        )
        failures += (
            0
            if check(f"total_chunks == {len(chunks)}", meta.get("total_chunks") == len(chunks))
            else 1
        )

        dim = meta.get("embedding_dim", EXPECTED_DIM)

        log("\n  input files for embedding:")
        log_input_files(meta, chunks)

        # ── 7. Load FAISS index ─────────────────────────────────────────────────
        log_section("7. FAISS index")

        try:
            import faiss
            import numpy as np

            index = faiss.read_index(str(index_file))
            failures += 0 if check(f"ntotal == {len(chunks)}", index.ntotal == len(chunks)) else 1
            failures += 0 if check(f"d == {dim}", index.d == dim) else 1

            # ── 8. Smoke-test query ──────────────────────────────────────────────
            log_section("8. Smoke-test query")

            from kbcraft.embedders.openai import OpenAIEmbedder

            embedder = OpenAIEmbedder(model=MODEL)
            query = "how does kbcraft build a FAISS index from markdown"
            log(f"  Query: '{query}'")

            k = min(3, len(chunks))
            q_vec = np.array(embedder.encode([query]), dtype=np.float32)
            distances, ids = index.search(q_vec, k=k)

            failures += 0 if check(f"search returns k={k} results", len(ids[0]) == k) else 1
            failures += (
                0
                if check("all result ids are valid", all(0 <= i < len(chunks) for i in ids[0]))
                else 1
            )

            log(f"\n  Top-{k} results:")
            for rank, (dist, idx) in enumerate(zip(distances[0], ids[0]), start=1):
                chunk = chunks[idx]
                source = Path(chunk["source"]).name if chunk.get("source") else "?"
                preview = chunk["text"][:80].replace("\n", " ")
                log(f"    {rank}. [{source}] (L2={dist:.4f})")
                log(f"       {preview!r}")

        except ImportError:
            log("  (faiss not available — skipping index and query checks)")

        # ── 9. Upload to S3 ─────────────────────────────────────────────────────
        log_section(f"9. Uploading to s3://{BUCKET}/")

        artifacts = [index_file, chunks_file, meta_file]
        for src in artifacts:
            try:
                s3.upload_file(str(src), BUCKET, src.name)
                check(f"uploaded s3://{BUCKET}/{src.name}", True)
            except Exception as exc:  # noqa: BLE001
                check(f"upload failed for {src.name}: {exc}", False)
                failures += 1

        # ── 10. Verify bucket contents ──────────────────────────────────────────
        log_section(f"10. Listing s3://{BUCKET}/")

        resp = s3.list_objects_v2(Bucket=BUCKET)
        keys = {obj["Key"]: obj["Size"] for obj in resp.get("Contents", [])}
        for src in artifacts:
            present = src.name in keys
            size_ok = present and keys[src.name] > 0
            failures += 0 if check(f"s3 object '{src.name}' present + non-empty", size_ok) else 1

        log("")
        for key, size in sorted(keys.items()):
            log(f"      {size:>10}  {key}")

    finally:
        if not SAVE_OUTPUT:
            shutil.rmtree(output_dir, ignore_errors=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    if failures == 0:
        log("  All checks passed ✓")
    else:
        log(f"  {failures} check(s) FAILED ✗")
    log(f"  MinIO console: http://localhost:9001  (minioadmin / minioadmin)")
    log("=" * 60 + "\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
