#!/usr/bin/env bash
# End-to-end CLI smoke test for `kbcraft index` using OpenAI embeddings.
#
# Bash port of scripts/test_cli_index_openai.py — drives the full pipeline
# through the `kbcraft` CLI binary (not Python imports):
#
#   kbcraft index ./ --embedder openai --model <model> --output <tmp> ...
#   kbcraft query <tmp> ... -q "..."
#
# All pipeline parameters (model, embedder, chunk size/overlap, expected
# embedding dim, output dir) are read from the yaml config files via
# src/kbcraft/config.py — NOT hardcoded here. This script asks config.py to
# resolve configs/embedding.yaml + configs/vector_store.yaml and emit them as
# shell variables:
#
#     eval "$(python -m kbcraft.config env --configs-dir configs)"
#
# Checks:
#   1. `kbcraft index` exits with code 0
#   2. <name>.faiss, <name>_chunks.json, <name>_meta.json are written
#   3. chunks.json has > 0 chunks, each with the expected fields
#   4. meta.json reports the right model / embedding_dim / total_chunks
#   5. FAISS index reports the expected vector count + dimension
#   6. `kbcraft query` returns ranked results
#
# Run locally (requires OPENAI_API_KEY set or present in .env):
#     ./scripts/test_cli_index_openai.sh
#
# Config resolution (each overrides the yaml files; see config.py):
#   OPENAI_EMBEDDING_MODEL       -> active embedding model      (default: text-embedding-3-small)
#   OPENAI_EMBEDDING_MAX_TOKENS  -> chunk size in tokens        (default: from yaml)
#   CHUNK_OVERLAP                -> chunk overlap in tokens     (default: from yaml)
#   OUTPUT_DIR                   -> persisted index dir         (default: from vector_store.yaml)
#   SAVE_VECTORDB=1              -> persist to OUTPUT_DIR instead of a temp dir

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIGS_DIR="${KBCRAFT_CONFIGS_DIR:-${ROOT}/configs}"

# Prefer the in-project venv's binaries (kbcraft, python) without requiring the
# caller to activate it first.
if [[ -d "${ROOT}/.venv/bin" ]]; then
  export PATH="${ROOT}/.venv/bin:${PATH}"
fi

# Load <repo>/.env so OPENAI_API_KEY is available without manual export. A
# shell-exported OPENAI_API_KEY still wins over the .env value.
_SHELL_OPENAI_API_KEY="${OPENAI_API_KEY:-}"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
if [[ -n "${_SHELL_OPENAI_API_KEY}" ]]; then
  export OPENAI_API_KEY="${_SHELL_OPENAI_API_KEY}"
fi

# Map this test's knobs onto the namespaced env vars config.py understands, then
# let config.py resolve everything else from the yaml files. This test targets
# the OpenAI backend, so default the active model to an OpenAI one.
export KBCRAFT_MODEL="${OPENAI_EMBEDDING_MODEL:-text-embedding-3-small}"
if [[ -n "${OPENAI_EMBEDDING_MAX_TOKENS:-}" ]]; then
  export KBCRAFT_MAX_TOKENS="${OPENAI_EMBEDDING_MAX_TOKENS}"
fi

SAVE_OUTPUT=0
case "$(printf '%s' "${SAVE_VECTORDB:-}" | tr '[:upper:]' '[:lower:]')" in
  1 | true | yes) SAVE_OUTPUT=1 ;;
esac

PASS=$'  \033[32m\xe2\x9c\x93\033[0m'
FAIL=$'  \033[31m\xe2\x9c\x97\033[0m'
FAILURES=0

log() { printf '%s\n' "$*"; }

log_section() {
  printf '\n%s\n  %s\n%s\n' \
    "────────────────────────────────────────────────────────────" \
    "$1" \
    "────────────────────────────────────────────────────────────"
}

# check <label> <condition-exit-code>
check() {
  local label="$1" ok="$2"
  if [[ "${ok}" -eq 0 ]]; then
    printf '%s  %s\n' "${PASS}" "${label}"
  else
    printf '%s  %s\n' "${FAIL}" "${label}"
    FAILURES=$(( FAILURES + 1 ))
  fi
}

# json_get <file> <jq-filter> — print the result, empty string on error.
json_get() { jq -er "$2" "$1" 2>/dev/null || printf ''; }

# ── preconditions ───────────────────────────────────────────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  log "error: OPENAI_API_KEY is not set (export it or add it to ${ROOT}/.env)" >&2
  exit 1
fi
for bin in kbcraft jq python; do
  command -v "${bin}" >/dev/null || { log "error: '${bin}' not on PATH" >&2; exit 1; }
done

# ── load pipeline params from the yaml config via config.py ─────────────────
# Emit KBCRAFT_*=value lines to a temp file (so we can check the exit code),
# then `source` them — avoids `eval` on command-substituted output.
CONFIG_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/kbcraft_cfg_XXXXXX.env")"
if ! python -m kbcraft.config env --configs-dir "${CONFIGS_DIR}" > "${CONFIG_ENV_FILE}"; then
  log "error: failed to resolve config from ${CONFIGS_DIR}" >&2
  rm -f "${CONFIG_ENV_FILE}"
  exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG_ENV_FILE}"
rm -f "${CONFIG_ENV_FILE}"

MODEL="${KBCRAFT_MODEL}"
EMBEDDER="${KBCRAFT_EMBEDDER}"
BASE_URL="${KBCRAFT_BASE_URL}"
EXPECTED_DIM="${KBCRAFT_EMBEDDING_DIM}"
CHUNK_SIZE="${KBCRAFT_MAX_TOKENS}"
OVERLAP="${KBCRAFT_CHUNK_OVERLAP}"
CONFIG_OUTPUT_DIR="${KBCRAFT_OUTPUT_DIR}"

# Source corpus. Defaults to the whole repo (a realistic, if costly, smoke
# test). Set KBCRAFT_TEST_SOURCE to a small fixture dir to keep CI cheap/fast.
SOURCE_DIR="${KBCRAFT_TEST_SOURCE:-${ROOT}}"
INDEX_NAME="test_index_openai"
QUERY="how does the chunker split markdown files"

if [[ "${EMBEDDER}" != "openai" ]]; then
  log "error: resolved embedder is '${EMBEDDER}', expected 'openai'." >&2
  log "       Set OPENAI_EMBEDDING_MODEL (or configs/embedding.yaml active_model) to an OpenAI model." >&2
  exit 1
fi

log ""
log "============================================================"
log "  kbcraft — CLI index smoke test (OpenAI Embeddings)"
log "============================================================"
log "  Config dir    : ${CONFIGS_DIR}"
log "  Source dir    : ${SOURCE_DIR}"
log "  Embedder      : ${EMBEDDER}"
log "  Model         : ${MODEL}"
log "  Embedding dim : ${EXPECTED_DIM}"
log "  Chunk size    : ${CHUNK_SIZE}"
log "  Chunk overlap : ${OVERLAP}"

if [[ "${SAVE_OUTPUT}" -eq 1 ]]; then
  OUTPUT_DIR="${CONFIG_OUTPUT_DIR}"
  [[ "${OUTPUT_DIR}" = /* ]] || OUTPUT_DIR="${ROOT}/${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"
else
  OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kbcraft_openai_test_XXXXXX")"
fi
log "  Output dir    : ${OUTPUT_DIR}"
log "  Persist output: ${SAVE_OUTPUT}"

cleanup() {
  if [[ "${SAVE_OUTPUT}" -ne 1 && -n "${OUTPUT_DIR:-}" ]]; then
    log "Clear destination folder: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
  fi
}
trap cleanup EXIT

INDEX_FILE="${OUTPUT_DIR}/${INDEX_NAME}.faiss"
CHUNKS_FILE="${OUTPUT_DIR}/${INDEX_NAME}_chunks.json"
META_FILE="${OUTPUT_DIR}/${INDEX_NAME}_meta.json"

# Extra --base-url only when the config resolves one (empty = real OpenAI API).
BASE_URL_ARGS=()
[[ -n "${BASE_URL}" ]] && BASE_URL_ARGS=(--base-url "${BASE_URL}")

# ── 1. Run the CLI ──────────────────────────────────────────────────────────
log_section "1. Running kbcraft index"

set -- kbcraft index "${SOURCE_DIR}" \
  --embedder "${EMBEDDER}" \
  --model "${MODEL}" \
  "${BASE_URL_ARGS[@]}" \
  --lang markdown \
  --lang python \
  --output "${OUTPUT_DIR}" \
  --name "${INDEX_NAME}" \
  --exclude ".venv/**" \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-overlap "${OVERLAP}"

log "  \$ $*"
log ""
"$@"
INDEX_RC=$?

log ""
check "exit code is 0" "$(( INDEX_RC == 0 ? 0 : 1 ))"
if [[ "${INDEX_RC}" -ne 0 ]]; then
  log ""
  log "  Command failed — aborting remaining checks."
  exit 1
fi

# ── 2. Output files ─────────────────────────────────────────────────────────
log_section "2. Output files"
for path in "${INDEX_FILE}" "${CHUNKS_FILE}" "${META_FILE}"; do
  check "$(basename "${path}") exists" "$([[ -f "${path}" ]]; echo $?)"
done

# ── 3. Validate chunks.json ─────────────────────────────────────────────────
log_section "3. chunks.json"
CHUNK_COUNT="$(json_get "${CHUNKS_FILE}" 'length')"
CHUNK_COUNT="${CHUNK_COUNT:-0}"
check "contains ${CHUNK_COUNT} chunks (> 0)" "$([[ "${CHUNK_COUNT}" -gt 0 ]]; echo $?)"

for field in text source index token_count; do
  has_field="$(json_get "${CHUNKS_FILE}" ".[0] | has(\"${field}\")")"
  check "chunk has field '${field}'" "$([[ "${has_field}" == "true" ]]; echo $?)"
done

# ── 4. Validate meta.json ───────────────────────────────────────────────────
log_section "4. meta.json"
META_MODEL="$(json_get "${META_FILE}" '.model')"
check "model == '${MODEL}'" "$([[ "${META_MODEL}" == "${MODEL}" ]]; echo $?)"

META_DIM="$(json_get "${META_FILE}" '.embedding_dim')"
check "embedding_dim == ${EXPECTED_DIM}" "$([[ "${META_DIM}" == "${EXPECTED_DIM}" ]]; echo $?)"

META_TOTAL="$(json_get "${META_FILE}" '.total_chunks')"
check "total_chunks == ${CHUNK_COUNT}" "$([[ "${META_TOTAL}" == "${CHUNK_COUNT}" ]]; echo $?)"

DIM="${META_DIM:-${EXPECTED_DIM}}"

# ── 5. Inspect FAISS index ──────────────────────────────────────────────────
log_section "5. FAISS index"
if python -c "import faiss" >/dev/null 2>&1; then
  read -r NTOTAL IDX_DIM < <(
    python - "${INDEX_FILE}" <<'PY'
import sys
import faiss

index = faiss.read_index(sys.argv[1])
print(index.ntotal, index.d)
PY
  )
  check "ntotal == ${CHUNK_COUNT}" "$([[ "${NTOTAL}" == "${CHUNK_COUNT}" ]]; echo $?)"
  check "d == ${DIM}" "$([[ "${IDX_DIM}" == "${DIM}" ]]; echo $?)"
else
  log "  (faiss not available — skipping index inspection)"
fi

# ── 6. Smoke-test query (via the CLI) ───────────────────────────────────────
log_section "6. Smoke-test query"
log "  Query: '${QUERY}'"
log ""

# --json puts the structured payload on stdout and decorative logging on stderr,
# so we capture stdout for parsing while the human-readable header still shows.
QUERY_JSON="$(kbcraft query "${OUTPUT_DIR}" \
  --name "${INDEX_NAME}" \
  --embedder "${EMBEDDER}" \
  --model "${MODEL}" \
  "${BASE_URL_ARGS[@]}" \
  -q "${QUERY}" \
  -k 3 \
  --json)"
QUERY_RC=$?

check "query exits 0" "$(( QUERY_RC == 0 ? 0 : 1 ))"

# Render the ranked hits for humans from the JSON payload.
printf '%s' "${QUERY_JSON}" \
  | jq -r '.results[] | "    \(.rank). [\(.source)] (L2=\(.l2 | .*10000 | round / 10000))"' 2>/dev/null \
  || true

RESULT_COUNT="$(printf '%s' "${QUERY_JSON}" | jq -er '.count' 2>/dev/null || printf '0')"
check "query returns 3 results" "$([[ "${RESULT_COUNT}" == "3" ]]; echo $?)"

# Relevance: for the full-repo corpus the top hits for a chunker query should
# include the chunker or README source. Skip when a custom fixture dir is used.
if [[ "${SOURCE_DIR}" == "${ROOT}" ]]; then
  TOP_SOURCES="$(printf '%s' "${QUERY_JSON}" | jq -r '.results[].source' 2>/dev/null)"
  RELEVANT=1
  printf '%s\n' "${TOP_SOURCES}" | grep -qiE 'chunker|readme' && RELEVANT=0
  check "top results include a relevant source (chunker/README)" "${RELEVANT}"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
log ""
log "============================================================"
if [[ "${FAILURES}" -eq 0 ]]; then
  log "  All checks passed ✓"
else
  log "  ${FAILURES} check(s) FAILED ✗"
fi
log "============================================================"
log ""

[[ "${FAILURES}" -eq 0 ]]
