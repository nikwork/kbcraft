#!/usr/bin/env python3
"""
End-to-end CLI smoke test for `kbcraft index` using OpenAI embeddings.

Runs the full pipeline via the `kbcraft` CLI binary (not Python imports), with
all pipeline parameters resolved from the yaml config files through
`kbcraft.config.resolve_params` — the same source of truth the shell driver
(`test_cli_index_openai.sh`) uses, so the two stay in lockstep.

  kbcraft index ./ --embedder openai --model <model> --output <tmp> ...
  kbcraft query <tmp> ... -q "..." --json

Checks:
  1. `kbcraft index` exits with code 0
  2. <name>.faiss, <name>_chunks.json, <name>_meta.json are written
  3. chunks.json has > 0 chunks, each with the expected fields
  4. meta.json reports the right model / embedding_dim / total_chunks
  5. FAISS index loads with the expected vector count / dimension
  6. `kbcraft query --json` returns ranked, in-range, relevant results

Run locally (requires OPENAI_API_KEY set or present in .env):
    python scripts/test_cli_index_openai.py

Config resolution (each overrides the yaml; see kbcraft/config.py):
    OPENAI_EMBEDDING_MODEL       -> active embedding model  (default: text-embedding-3-small)
    OPENAI_EMBEDDING_MAX_TOKENS  -> chunk size in tokens    (default: from yaml)
    KBCRAFT_TEST_SOURCE          -> corpus to index         (default: whole repo)
    SAVE_VECTORDB=1              -> persist to the configured output dir
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Map this test's knobs onto the namespaced env vars config.py understands, then
# let config.py resolve everything else from the yaml files.
os.environ.setdefault(
    "KBCRAFT_MODEL", os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)
if os.environ.get("OPENAI_EMBEDDING_MAX_TOKENS"):
    os.environ["KBCRAFT_MAX_TOKENS"] = os.environ["OPENAI_EMBEDDING_MAX_TOKENS"]

from kbcraft.config import resolve_params  # noqa: E402  (after env is prepared)

_params = resolve_params(PROJECT_ROOT / "configs")
MODEL = _params["KBCRAFT_MODEL"]
EMBEDDER = _params["KBCRAFT_EMBEDDER"]
BASE_URL = _params["KBCRAFT_BASE_URL"]
EXPECTED_DIM = int(_params["KBCRAFT_EMBEDDING_DIM"])
CHUNK_SIZE = int(_params["KBCRAFT_MAX_TOKENS"])
CHUNK_OVERLAP = int(_params["KBCRAFT_CHUNK_OVERLAP"])
CONFIG_OUTPUT_DIR = _params["KBCRAFT_OUTPUT_DIR"]

# S3 export params come straight from the S3_* env vars (.env) — the single
# source of truth. Secrets stay in env/config; the boto3 exporter reads them.
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "")
S3_HAS_CREDENTIALS = bool(
    (os.environ.get("S3_ACCESS_KEY_ID") and os.environ.get("S3_SECRET_ACCESS_KEY"))
    or os.environ.get("S3_PROFILE")
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Corpus. Defaults to the whole repo; point at a small fixture dir for cheap CI.
SOURCE_DIR = Path(os.environ.get("KBCRAFT_TEST_SOURCE", PROJECT_ROOT))
INDEX_NAME = "test_index_openai"
QUERY = "how does the chunker split markdown files"

SAVE_OUTPUT = os.environ.get("SAVE_VECTORDB", "").lower() in ("1", "true", "yes")

_BASE_URL_ARGS = ["--base-url", BASE_URL] if BASE_URL else []

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
    log("  kbcraft — CLI index smoke test (OpenAI Embeddings)")
    log("=" * 60)
    log(f"  Source dir    : {SOURCE_DIR}")
    log(f"  Embedder      : {EMBEDDER}")
    log(f"  Model         : {MODEL}")
    log(f"  Embedding dim : {EXPECTED_DIM}")
    log(f"  Chunk size    : {CHUNK_SIZE}")
    log(f"  Chunk overlap : {CHUNK_OVERLAP}")
    log(f"  S3 bucket     : {S3_BUCKET or '<none configured>'}")
    log(f"  S3 endpoint   : {S3_ENDPOINT or '<aws default>'}")

    if EMBEDDER != "openai":
        log(
            f"\nerror: resolved embedder is {EMBEDDER!r}, expected 'openai'. "
            "Set OPENAI_EMBEDDING_MODEL to an OpenAI model."
        )
        return 1

    failures = 0
    if SAVE_OUTPUT:
        output_dir = Path(CONFIG_OUTPUT_DIR)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="kbcraft_openai_test_"))
    log(f"  Output dir    : {output_dir}")
    log(f"  Persist output: {SAVE_OUTPUT}")

    try:
        # ── 1. Run the CLI ─────────────────────────────────────────────────────
        log_section("1. Running kbcraft index")

        cmd = [
            "kbcraft",
            "index",
            str(SOURCE_DIR),
            "--embedder",
            EMBEDDER,
            "--model",
            MODEL,
            *_BASE_URL_ARGS,
            "--lang",
            "markdown",
            "--lang",
            "python",
            "--output",
            str(output_dir),
            "--name",
            INDEX_NAME,
            "--exclude",
            ".venv/**",
            "--chunk-size",
            str(CHUNK_SIZE),
            "--chunk-overlap",
            str(CHUNK_OVERLAP),
        ]
        log(f"  $ {' '.join(cmd)}\n")

        result = subprocess.run(cmd, capture_output=False, text=True)

        log("")
        ok = check("exit code is 0", result.returncode == 0)
        if not ok:
            failures += 1
            log("\n  Command failed — aborting remaining checks.")
            return 1

        # ── 2. Output files ────────────────────────────────────────────────────
        log_section("2. Output files")

        index_file = output_dir / f"{INDEX_NAME}.faiss"
        chunks_file = output_dir / f"{INDEX_NAME}_chunks.json"
        meta_file = output_dir / f"{INDEX_NAME}_meta.json"

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

        # ── 5. Load FAISS index ────────────────────────────────────────────────
        log_section("5. FAISS index")

        try:
            import faiss

            index = faiss.read_index(str(index_file))
            failures += 0 if check(f"ntotal == {len(chunks)}", index.ntotal == len(chunks)) else 1
            failures += 0 if check(f"d == {dim}", index.d == dim) else 1
        except ImportError:
            log("  (faiss not available — skipping index checks)")

        # ── 6. Smoke-test query (via the CLI, JSON output) ─────────────────────
        log_section("6. Smoke-test query")
        log(f"  Query: '{QUERY}'")

        query_cmd = [
            "kbcraft",
            "query",
            str(output_dir),
            "--name",
            INDEX_NAME,
            "--embedder",
            EMBEDDER,
            "--model",
            MODEL,
            *_BASE_URL_ARGS,
            "-q",
            QUERY,
            "-k",
            "3",
            "--json",
        ]
        q_result = subprocess.run(query_cmd, capture_output=True, text=True)
        if q_result.stderr:
            print(q_result.stderr, end="", file=sys.stderr)

        failures += 0 if check("query exits 0", q_result.returncode == 0) else 1

        payload = json.loads(q_result.stdout) if q_result.stdout.strip() else {}
        results = payload.get("results", [])

        failures += 0 if check("query returns 3 results", payload.get("count") == 3) else 1
        failures += (
            0
            if check(
                "all result ids are valid",
                bool(results) and all(0 <= r["id"] < len(chunks) for r in results),
            )
            else 1
        )

        # Relevance is meaningful only for the full-repo corpus.
        if SOURCE_DIR == PROJECT_ROOT:
            sources = " ".join(r.get("source", "").lower() for r in results)
            failures += (
                0
                if check(
                    "top results include a relevant source (chunker/README)",
                    ("chunker" in sources) or ("readme" in sources),
                )
                else 1
            )

        log("\n  Top-3 results:")
        for r in results:
            source = Path(r["source"]).name if r.get("source") else "?"
            preview = r["preview"][:80].replace("\n", " ")
            log(f"    {r['rank']}. [{source}] (L2={r['l2']:.4f})")
            log(f"       {preview!r}")

        # ── 7. Export FAISS index to S3 (config + boto3) ───────────────────────
        log_section("7. Export FAISS index to S3")

        if not S3_BUCKET or not S3_HAS_CREDENTIALS:
            log("  (no S3 bucket/credentials in .env — skipping real S3 export)")
            log("  enable it by setting S3_BUCKET + S3_ACCESS_KEY_ID/")
            log("  S3_SECRET_ACCESS_KEY (or S3_PROFILE) in .env, or configs/storage.yaml")
        else:
            log(f"  Bucket   : {S3_BUCKET}")
            log(f"  Endpoint : {S3_ENDPOINT or '<aws default>'}\n")

            exporter = [sys.executable, str(SCRIPT_DIR / "s3_export.py")]
            configs_args = ["--configs-dir", str(CONFIGS_DIR)]

            ensure_rc = subprocess.run([*exporter, "ensure-bucket", *configs_args]).returncode
            failures += 0 if check(f"bucket '{S3_BUCKET}' exists or created", ensure_rc == 0) else 1

            upload_rc = subprocess.run(
                [*exporter, "upload", *configs_args, "--dir", str(output_dir), "--name", INDEX_NAME]
            ).returncode
            failures += 0 if check("uploaded index artifacts to S3", upload_rc == 0) else 1

            # ── 8. Verify S3 contents (config + boto3) ─────────────────────────
            log_section("8. Verify S3 contents")
            verify_rc = subprocess.run(
                [*exporter, "verify", *configs_args, "--name", INDEX_NAME]
            ).returncode
            failures += (
                0
                if check(f"all 3 artifacts present + non-empty in s3://{S3_BUCKET}", verify_rc == 0)
                else 1
            )

    finally:
        if not SAVE_OUTPUT:
            log(f"\nClear destination folder: {output_dir}")
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
