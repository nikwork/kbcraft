#!/usr/bin/env python3
"""
kbcraft — end-to-end FAISS + OpenAI + S3 (MinIO), Python port.

Mirrors `scripts/e2e_faiss_s3.sh`, but is designed to run **inside the dev
container** (`kbcraft-dev`), where:
  - MinIO is reachable as http://minio:9000 (sibling compose service)
  - .env is auto-loaded into the environment by docker-compose.dev.yml
  - The project venv is on PATH (kbcraft binary callable directly)

Re-running is idempotent: bucket reused, artifacts overwritten.

Usage (inside the dev container):
    python scripts/e2e_faiss_s3.py

From the host (delegates into the container):
    docker exec kbcraft-dev python scripts/e2e_faiss_s3.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# ── config ────────────────────────────────────────────────────────────────────
DOCS_DIR = PROJECT_ROOT / ".e2e" / "docs"
OUTPUT_DIR = PROJECT_ROOT / ".e2e" / "index"
BUCKET = os.environ.get("KBCRAFT_E2E_BUCKET", "kbcraft-e2e")
INDEX_NAME = os.environ.get("KBCRAFT_E2E_INDEX_NAME", "index")
MODEL = os.environ.get("KBCRAFT_E2E_MODEL", "text-embedding-3-small")

# MinIO defaults match docker-compose.dev.yml. Inside the dev container the
# sibling host is "minio"; from the host it's "localhost". Whichever value is
# already set in the environment wins.
AWS_ACCESS_KEY_ID = os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_DEFAULT_REGION = os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
S3_ENDPOINT = os.environ.setdefault("STORAGE_S3_ENDPOINT_URL", "http://minio:9000")


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


def fail(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )


def wait_for_minio(timeout_s: int = 30) -> None:
    url = f"{S3_ENDPOINT.rstrip('/')}/minio/health/ready"
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError) as exc:
            last_err = exc
        time.sleep(1)
    fail(f"minio not ready at {url} within {timeout_s}s ({last_err})")


def ensure_bucket(s3, bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"     bucket exists: {bucket}")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            s3.create_bucket(Bucket=bucket)
            print(f"     created bucket: {bucket}")
        else:
            raise


def seed_docs(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "intro.md").write_text(INTRO_MD)
    (docs_dir / "usage.md").write_text(USAGE_MD)


def build_index(docs_dir: Path, output_dir: Path, name: str, model: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kbcraft = shutil.which("kbcraft")
    cmd: list[str]
    if kbcraft:
        cmd = [kbcraft]
    else:
        cmd = [sys.executable, "-m", "kbcraft.cli"]
    cmd += [
        "index",
        str(docs_dir),
        "--embedder",
        "openai",
        "--model",
        model,
        "--output",
        str(output_dir),
        "--name",
        name,
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def artifacts_for(name: str) -> Iterable[str]:
    return (f"{name}.faiss", f"{name}_chunks.json", f"{name}_meta.json")


def upload_artifacts(s3, bucket: str, output_dir: Path, name: str) -> None:
    for fname in artifacts_for(name):
        src = output_dir / fname
        if not src.is_file():
            fail(f"missing artifact {src}")
        s3.upload_file(str(src), bucket, fname)
        print(f"     uploaded s3://{bucket}/{fname}")


def list_bucket(s3, bucket: str) -> None:
    resp = s3.list_objects_v2(Bucket=bucket)
    for obj in resp.get("Contents", []):
        print(f"     {obj['Size']:>10}  {obj['Key']}")


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        fail("OPENAI_API_KEY is not set — required for OpenAI embeddings")

    print(f"── 1. waiting for minio at {S3_ENDPOINT}")
    wait_for_minio()

    print(f"── 2. ensuring bucket s3://{BUCKET} via boto3")
    s3 = s3_client()
    ensure_bucket(s3, BUCKET)

    print(f"── 3. seeding sample docs in {DOCS_DIR}")
    seed_docs(DOCS_DIR)

    print(f"── 4. building FAISS index with OpenAI embeddings ({MODEL})")
    build_index(DOCS_DIR, OUTPUT_DIR, INDEX_NAME, MODEL)

    print(f"── 5. uploading to s3://{BUCKET}/ via boto3")
    upload_artifacts(s3, BUCKET, OUTPUT_DIR, INDEX_NAME)

    print(f"── 6. listing s3://{BUCKET}/ via boto3")
    list_bucket(s3, BUCKET)

    print()
    print("done. minio console: http://localhost:9001  (minioadmin / minioadmin)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
