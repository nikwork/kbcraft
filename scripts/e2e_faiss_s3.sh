#!/usr/bin/env bash
# kbcraft — local end-to-end: build a FAISS index with real OpenAI embeddings
# and push it to MinIO (the S3 service in docker-compose.dev.yml) using boto3.
#
# Prereqs on host:
#   - docker compose
#   - poetry, with deps installed: poetry install --extras s3 --extras faiss
#     (or: poetry install --all-extras)
#   - OPENAI_API_KEY available (set in <repo>/.env or exported in shell)
#
# Usage:
#   ./scripts/e2e_faiss_s3.sh
#
# Re-run is idempotent: bucket reused, files overwritten.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# Load <repo>/.env if present so OPENAI_API_KEY (and any AWS_* overrides) are
# available without the caller having to export them manually.
if [[ -f "${ROOT}/.env" ]]; then
  echo "── loading ${ROOT}/.env"
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

DOCS_DIR="${ROOT}/.e2e/docs"
OUTPUT_DIR="${ROOT}/.e2e/index"
BUCKET="${KBCRAFT_E2E_BUCKET:-kbcraft-e2e}"
INDEX_NAME="${KBCRAFT_E2E_INDEX_NAME:-index}"
MODEL="${KBCRAFT_E2E_MODEL:-text-embedding-3-small}"

# MinIO defaults match the env block in docker-compose.dev.yml.
export S3_ACCESS_KEY_ID="${S3_ACCESS_KEY_ID:-minioadmin}"
export S3_SECRET_ACCESS_KEY="${S3_SECRET_ACCESS_KEY:-minioadmin}"
export S3_REGION="${S3_REGION:-us-east-1}"
export S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-http://localhost:9000}"

# ── checks ────────────────────────────────────────────────────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY is not set — required for OpenAI embeddings" >&2
  exit 1
fi
for bin in docker poetry curl; do
  command -v "$bin" >/dev/null || { echo "error: '$bin' not on PATH" >&2; exit 1; }
done

# Verify boto3 is importable in the project's venv (extras may not be installed).
if ! (cd "${ROOT}" && poetry run python -c "import boto3" 2>/dev/null); then
  echo "error: boto3 not installed in poetry venv. Run: poetry install --extras s3" >&2
  exit 1
fi

# ── 1. start MinIO (S3) ───────────────────────────────────────────────────────
echo "── 1. starting minio (docker compose -f docker-compose.dev.yml)"
docker compose -f "${ROOT}/docker-compose.dev.yml" up -d minio

echo "     waiting for minio at ${S3_ENDPOINT_URL} …"
ready=0
for _ in $(seq 1 30); do
  if curl -fs "${S3_ENDPOINT_URL}/minio/health/ready" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "error: minio did not become ready in 30s" >&2
  exit 1
fi

# ── 2. ensure bucket (boto3) ──────────────────────────────────────────────────
echo "── 2. ensuring bucket s3://${BUCKET} via boto3"
(
  cd "${ROOT}"
  BUCKET="${BUCKET}" poetry run python - <<'PY'
import os, sys
import boto3
from botocore.exceptions import ClientError

bucket = os.environ["BUCKET"]
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    region_name=os.environ["S3_REGION"],
)
try:
    s3.head_bucket(Bucket=bucket)
    print(f"     bucket exists: {bucket}")
except ClientError as exc:
    code = exc.response.get("Error", {}).get("Code", "")
    if code in ("404", "NoSuchBucket", "NotFound"):
        s3.create_bucket(Bucket=bucket)
        print(f"     created bucket: {bucket}")
    else:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
PY
)

# ── 3. seed sample docs ───────────────────────────────────────────────────────
echo "── 3. seeding sample docs in ${DOCS_DIR}"
mkdir -p "${DOCS_DIR}"
cat > "${DOCS_DIR}/intro.md" <<'EOF'
# Introduction

kbcraft is a CLI that turns Markdown notes into a RAG-ready vector index.
The pipeline: select files, chunk them with a tokenizer, embed the chunks,
and write a FAISS index plus chunk + metadata sidecars.
EOF
cat > "${DOCS_DIR}/usage.md" <<'EOF'
# Usage

Run `kbcraft index ./docs --output ./out --embedder openai
--model text-embedding-3-small` to build a FAISS flat-L2 index using the
OpenAI Embeddings API. The output directory will contain `index.faiss`,
`index_chunks.json`, and `index_meta.json`.
EOF

# ── 4. build the FAISS index ──────────────────────────────────────────────────
echo "── 4. building FAISS index with OpenAI embeddings (${MODEL})"
mkdir -p "${OUTPUT_DIR}"
(
  cd "${ROOT}"
  poetry run kbcraft index "${DOCS_DIR}" \
    --embedder openai \
    --model "${MODEL}" \
    --output "${OUTPUT_DIR}" \
    --name "${INDEX_NAME}"
)

# ── 5. push to S3 (boto3) ─────────────────────────────────────────────────────
echo "── 5. uploading to s3://${BUCKET}/ via boto3"
(
  cd "${ROOT}"
  BUCKET="${BUCKET}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  INDEX_NAME="${INDEX_NAME}" \
  poetry run python - <<'PY'
import os, sys
import boto3

bucket = os.environ["BUCKET"]
out = os.environ["OUTPUT_DIR"]
name = os.environ["INDEX_NAME"]

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    region_name=os.environ["S3_REGION"],
)
files = [f"{name}.faiss", f"{name}_chunks.json", f"{name}_meta.json"]
for fname in files:
    src = os.path.join(out, fname)
    if not os.path.isfile(src):
        print(f"error: missing artifact {src}", file=sys.stderr)
        sys.exit(1)
    s3.upload_file(src, bucket, fname)
    print(f"     uploaded s3://{bucket}/{fname}")
PY
)

# ── 6. verify (boto3) ─────────────────────────────────────────────────────────
echo "── 6. listing s3://${BUCKET}/ via boto3"
(
  cd "${ROOT}"
  BUCKET="${BUCKET}" poetry run python - <<'PY'
import os
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    region_name=os.environ["S3_REGION"],
)
resp = s3.list_objects_v2(Bucket=os.environ["BUCKET"])
for obj in resp.get("Contents", []):
    print(f"     {obj['Size']:>10}  {obj['Key']}")
PY
)

echo
echo "done. minio console: http://localhost:9001  (minioadmin / minioadmin)"
