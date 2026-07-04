# Flow Analysis & Improvement Report — `test_cli_index_openai` (shell + Python)

**Reviewer perspective:** senior Python / ML engineer
**Date:** 2026-07-04
**Scope:** the OpenAI CLI index smoke test and *every* function it transitively exercises.

> **STATUS — all 10 recommendations applied (2026-07-04).** See the
> [Resolution log](#resolution-log) at the end for the change made per finding
> and the verification performed. The findings below are retained as the
> original analysis.

Files in the call graph that were read for this review:

| Layer | File |
| --- | --- |
| Test driver (shell) | `scripts/test_cli_index_openai.sh` |
| Test driver (Python) | `scripts/test_cli_index_openai.py` |
| Config bridge | `src/kbcraft/config.py` (`ConfigFactory`, `resolve_params`, `_emit_env`) |
| Config data | `configs/embedding.yaml`, `configs/vector_store.yaml` |
| CLI | `src/kbcraft/cli.py` (`_build_parser`, `_build_embedder`, `_cmd_index`, `_cmd_query`) |
| Selection | `src/kbcraft/selector.py` (`FileFilter`) |
| Chunking | `src/kbcraft/chunker.py` (`Chunker`) |
| Embedding | `src/kbcraft/embedder.py` (`TokenChunkingEmbedder`, `OpenAICompatibleEmbedder`), `src/kbcraft/embedders/openai.py` (`OpenAIEmbedder`) |
| Tokenizing | `src/kbcraft/tokenizer.py` (`WhitespaceTokenizer`) |

---

## 1. End-to-end flow

```
test_cli_index_openai.sh
│
├─ 0. bootstrap: prepend .venv/bin to PATH, load .env (shell key wins),
│       map OPENAI_EMBEDDING_MODEL→MODEL, OPENAI_EMBEDDING_MAX_TOKENS→MAX_TOKENS
│
├─ 1. resolve params from YAML  ──►  python -m kbcraft.config env --configs-dir configs
│       ConfigFactory.load_embedding() + load_vector_store()
│       → prints KBCRAFT_MODEL / EMBEDDER / EMBEDDING_DIM / MAX_TOKENS /
│         CHUNK_OVERLAP / BASE_URL / OUTPUT_DIR   (shlex-quoted)
│       shell `eval`s them
│
├─ 2. kbcraft index <repo> --embedder openai --model <m> --chunk-size <M> ...
│       _cmd_index:
│         _build_embedder()  → OpenAIEmbedder + WhitespaceTokenizer  ← ⚠ see F1
│         FileFilter.from_kbignore(...).collect_files()              (selector)
│         Chunker(max_chunk_tokens=M, tokenize=whitespace).chunk_files()
│         embedder.encode(texts)   ← plain encode(), no doc prefix   ← ⚠ see F2
│         faiss.IndexFlatL2(dim).add(matrix)                         ← ⚠ see F1
│         write <name>.faiss / _chunks.json / _meta.json
│
├─ 3-5. assertions via jq + a small inline `faiss.read_index` python snippet
│
└─ 6. kbcraft query <dir> --embedder openai --model <m> -q "..."
        _cmd_query: read index+chunks, embedder.encode([query]), index.search(k)
        grep "Query returned N result(s)."  ← ⚠ brittle, see F9
```

The Python driver (`.py`) follows the same shape **except** it does not use
`config.py` at all — it hardcodes the model list, `overlap = MAX_TOKENS // 8`,
and performs the step-6 query *in-process* (`import faiss` + `OpenAIEmbedder`)
rather than through `kbcraft query`.

---

## 2. Findings (ranked by severity)

| # | Severity | Area | One-line summary |
| --- | --- | --- | --- |
| F1 | **High** | Correctness / ML | Chunk↔vector misalignment when any chunk exceeds the model token limit; the guard is a silent no-op. |
| F2 | **High** | Correctness / ML | CLI index/query call plain `encode()`, so configured `query_prefix`/`document_prefix` are silently dropped (breaks asymmetric models). |
| F3 | Medium | ML retrieval quality | Default chunk size = **8191 tokens** (model *hard limit*) is an anti-pattern for RAG recall. |
| F4 | Medium | Performance | Request batching budget = `max_tokens`, so the corpus is sent as many tiny sequential requests. |
| F5 | Medium | Correctness | Whitespace tokenizer sizes chunks for a tiktoken model → wrong `token_count` metadata and wrong chunk boundaries. |
| F6 | Medium | Robustness | Independent `MAX_TOKENS` override without adjusting overlap can violate `overlap < max_chunk_tokens` → hard crash. |
| F7 | Medium | Design | `config.py` env keys (`MODEL`, `MAX_TOKENS`, `OUTPUT_DIR`) are un-namespaced globals → collision risk; shell `eval` of emitted vars. |
| F8 | Low | Maintainability | `.sh` and `.py` drivers have diverged (config-driven vs hardcoded; CLI query vs in-process). |
| F9 | Low | Test design | Result count scraped from log text; smoke test asserts *count*, never *relevance*; embeds the whole repo (cost/flake). |
| F10 | Low | Reliability | Single `encode()` over the whole corpus, no retry/backoff, no partial progress. |

---

## 3. Detailed findings

### F1 — Chunk↔vector misalignment (High, correctness + data integrity)

**Evidence** — `src/kbcraft/cli.py` `_cmd_index`:

```python
vectors = embedder.encode(texts)
...
if len(vectors) != len(chunks):
    log("Note: ... some chunks were further split ...")
    chunks = chunks[: len(vectors)]        # ← no-op when vectors > chunks
```

`OpenAIEmbedder` inherits `TokenChunkingEmbedder.encode`, which *expands* any
input exceeding `max_tokens` into extra sub-chunks via `split_chunks`
(`embedder.py:362`). Therefore `len(vectors) >= len(chunks)` — it can only grow.
When it grows, `chunks[:len(vectors)]` returns the **full** list unchanged
(verified: `[1,2,3][:5] == [1,2,3]`). Net effect:

* `index.ntotal` (= `len(vectors)`) **exceeds** `len(chunks)`;
* the extra vectors are the tiktoken sub-splits of *earlier* documents, so
  from the first oversize text onward **`vector[i]` no longer corresponds to
  `chunks[i]`** — provenance is silently wrong;
* `kbcraft query` can return an id `>= len(chunks)` (masked today only because
  `_cmd_query` skips out-of-range ids, silently dropping results);
* the Python driver's check `all(0 <= i < len(chunks))` (`.py:197`) would fail.

**Why it didn't fire in the smoke run:** repo files are small and Markdown
header-splitting keeps sections well under the limit, so no chunk exceeded 8191
tiktoken tokens. It **will** fire on real corpora — which is exactly what a
`--chunk-size 8191` configuration invites (see F3).

**Fix (choose one, ideally both):**
1. Make chunk-token accounting match the embedder: on the OpenAI path, inject a
   *tiktoken* length function into the `Chunker` (F5) so the chunker never
   emits a chunk larger than the model limit → `encode()` never sub-splits.
2. Make alignment correct-by-construction: have the pipeline embed
   chunk-by-chunk (or return `(chunk, vector)` pairs from a single API) and
   drop the fragile post-hoc `len()` reconciliation entirely.

---

### F2 — Configured prefixes are dropped on the CLI path (High, ML correctness)

**Evidence** — `src/kbcraft/cli.py`:

```python
vectors = embedder.encode(texts)                       # index  (line 663)
q_vec = np.array(embedder.encode([args.query]), ...)   # query  (line 783)
```

Both call the *symmetric* `encode()`. The asymmetric API
(`encode_documents` / `encode_query`) that injects instruction prefixes lives in
`OllamaEmbedder` (`embedders/ollama.py:165-179`) and is honored by
`scripts/build_faiss_index.py:184,256` — but **the CLI bypasses it**.
`ModelConfig.query_prefix` / `document_prefix` are parsed
(`config.py:424-425`) and then consumed by *nobody* in the CLI path.

**Impact:** For models that require prefixes (nomic-embed-text
`search_query:`/`search_document:`, mxbai/arctic query instruction, E5-style),
indexing and querying through `kbcraft index`/`kbcraft query` produce
**incorrectly-prefixed embeddings and degraded retrieval**. OpenAI models are
unaffected (no prefix needed), so this smoke test passes — but the smoke test is
therefore *not* representative of the Ollama path it shares code with.

**Fix:** In `_cmd_index` use `encode_documents(texts)`; in `_cmd_query` use
`encode_query(text)`. These already fall back to plain `encode()` on
`BaseEmbedder`, so the OpenAI path is unchanged. Thread the configured prefixes
through `_build_embedder`.

---

### F3 — 8191-token chunks are a RAG anti-pattern (Medium, retrieval quality)

`configs/embedding.yaml` sets the OpenAI models' `chunking.max_tokens: 8191`,
i.e. the model's **hard input ceiling**, and the shell/Python drivers pass that
as `--chunk-size`. Embedding near-maximal chunks:

* dilutes the semantic signal (one 8k-token vector averages many topics) →
  lower top-k precision/recall;
* produces coarse retrieval granularity (you retrieve a whole file section, not
  the relevant paragraph);
* directly triggers F1/F5.

Well-tuned RAG chunking for `text-embedding-3-*` is typically **256–800 tokens**
with 10–20 % overlap. Recommend `max_tokens: 512, overlap: 64` as the default
and treat 8191 as the *validation ceiling*, not the target.

---

### F4 — Request batching budget conflates two different limits (Medium, perf)

**Evidence** — `embedder.py:376` `budget = self.max_tokens` (8191 for OpenAI).
`_batches` packs chunks until the running total would exceed **8191 tokens**,
then flushes. So each HTTP request carries ≈ one full-size chunk's worth of
tokens → for a large corpus this is *many* sequential round-trips.

`max_tokens` is the **per-input context window**; it should not also be the
**per-request batch budget**. OpenAI accepts up to 2048 inputs and a large total
token count per request. Introduce a separate `request_token_budget`
(e.g. 100k–300k) and a `max_inputs_per_request` cap (2048); batch to the
*larger* of those, not to the single-input limit. Expect a large wall-clock win
on real corpora.

---

### F5 — Whitespace tokenizer drives a tiktoken model (Medium, correctness)

`_build_embedder` (OpenAI branch, `cli.py`) sets
`tokenize_fn = WhitespaceTokenizer(...).tokenize` while the embedder itself uses
tiktoken `cl100k_base`. Consequences:

* `--chunk-size N` means *N whitespace words*, but the model counts tiktoken
  tokens (~1.3–1.5×/word prose, ~1.5–2.5×/word code). Chunks are under-counted
  → they overflow the true limit → F1.
* `chunks.json.token_count` is a **word count**, not a token count — misleading
  to any downstream consumer that trusts it.

`OpenAIEmbedder` already exposes an exact tiktoken tokenizer
(`embedders/openai.py:131`). Use `embedder.count_tokens` / a tiktoken-based
tokenize function for the chunker on the OpenAI path.

---

### F6 — Overlap can exceed chunk size after an env override (Medium, robustness)

The shell driver lets `OPENAI_EMBEDDING_MAX_TOKENS` override `MAX_TOKENS`
independently, but `CHUNK_OVERLAP` still comes from YAML (1023). Set
`OPENAI_EMBEDDING_MAX_TOKENS=500` and the Chunker constructor raises:

```
ValueError: chunk_overlap (1023) must be less than max_chunk_tokens (500)
```

(`chunker.py:103`). The original Python driver avoided this by *deriving*
`overlap = MAX_TOKENS // 8`; the config-driven refactor decoupled them and
reintroduced the foot-gun. Fix: clamp `overlap = min(overlap, max_tokens // 8)`
in `resolve_params` (or in the CLI), or derive overlap as a ratio.

---

### F7 — Un-namespaced config env keys + `eval` bridge (Medium, design)

* `config.py` reads generic names: `_env("MODEL")`, `_env("MAX_TOKENS")`,
  `_env("OUTPUT_DIR")`, `_env("CHUNK_OVERLAP")`. `MODEL`/`OUTPUT_DIR` are common
  in unrelated tooling → silent cross-contamination. Prefer `KBCRAFT_MODEL`,
  `KBCRAFT_MAX_TOKENS`, … (keep old names as deprecated aliases).
* The shell does `eval "$(python -m kbcraft.config env)"`. Values are
  `shlex.quote`d (injection-safe today), but `eval` is fragile: any stray stdout
  line from an imported module becomes shell code. Prefer
  `source <(python -m kbcraft.config env)` or read `KEY=VALUE` lines into an
  associative array; and have `_emit_env` guarantee stdout carries *only*
  assignments.
* `resolve_params` always reads `vs.faiss.output_dir` regardless of
  `active_backend`, yet advertises a generic `KBCRAFT_VECTOR_BACKEND`. Read
  `vs.backend`'s dir for the active backend.

---

### F8 — The two drivers have diverged (Low, maintainability)

`.sh` is config-driven and queries through the CLI; `.py` hardcodes the model
map (`_KNOWN_DIMS`), derives overlap as `//8`, and queries in-process. They now
test *different* code paths and will drift. Pick one of:
* make `.py` the thin reference and `.sh` the CI entry (or vice-versa), or
* have `.py` also call `resolve_params` and `kbcraft query`, deleting the
  duplicated constants.

---

### F9 — Weak, expensive smoke assertions (Low, test design)

* Step 6 scrapes `Query returned N result(s).` with `sed` — brittle coupling to
  log wording. Add `kbcraft query --json` and assert on structured output.
* The test asserts *3 results returned*, never that result #1 is *relevant*.
  Add a cheap relevance assertion (e.g. top-1 source ∈ {chunker, README}).
* Both drivers embed the **entire repository** on every run → real \$ cost,
  network dependence, CI flakiness. Offer a `--source`/fixture-dir mode that
  indexes a tiny `tests/fixtures/` corpus for CI, reserving the full-repo run
  for an opt-in nightly.

---

### F10 — No resilience around the embedding call (Low, reliability)

`vectors = embedder.encode(texts)` is one all-or-nothing pass. A transient 429/
5xx aborts the whole build with no retry/backoff and no partial checkpoint.
Add bounded exponential-backoff retry (the `openai` SDK has `max_retries`) and
consider streaming vectors to disk so a failure near the end isn't total loss.

---

## 4. ML / retrieval-quality notes (beyond bugs)

* **Distance metric.** `IndexFlatL2` on OpenAI v3 embeddings (unit-normalized)
  ranks identically to cosine, so results are correct. But if anyone uses the
  `dimensions` parameter (truncated embeddings are *not* re-normalized) L2 and
  cosine diverge. `vector_store.yaml` already documents `flat_ip` for cosine —
  prefer `flat_ip` + explicit normalization to make the metric intent obvious
  and dimension-truncation-safe.
* **Provenance header in the embedded text.** `Chunker(prepend_source=True)`
  embeds `File:/Path:` into the chunk *content* (`chunker.py:255`). This nudges
  every vector toward filename tokens and inflates `token_count`. Fine for small
  gains in provenance, but consider storing provenance as metadata only and
  embedding clean text, then re-attaching the header at display time.
* **`token_count` semantics** (see F5) should reflect the *embedding* tokenizer,
  otherwise any budget/window logic built on it downstream is wrong.

---

## 5. Prioritized action list

**Do now (correctness):**
1. F1 — eliminate chunk↔vector misalignment (tiktoken chunk sizing + drop the
   no-op guard, or embed pairwise).
2. F2 — use `encode_documents`/`encode_query` in the CLI and thread prefixes.
3. F5 — inject the model's real tokenizer into the `Chunker` on the OpenAI path.

**Next (quality & robustness):**
4. F3 — default chunk size 512/overlap 64; keep 8191 as the ceiling.
5. F6 — clamp/derive overlap vs max_tokens.
6. F4 — separate request batch budget from context limit.

**Then (hygiene):**
7. F7 — namespace `KBCRAFT_*` env keys; replace `eval` with `source <(...)`.
8. F9 — `kbcraft query --json`, fixture-dir mode for CI, a relevance assertion.
9. F8 — converge the two drivers.
10. F10 — retry/backoff around embedding.

---

## Appendix A — config resolution precedence (as implemented)

For every value: **environment variable → YAML file → dataclass default**
(`config.py` `_env`). Active-model chunking additionally honors `MAX_TOKENS` /
`CHUNK_OVERLAP` overrides (`config.py:264-267`). The shell driver injects
`MODEL` (default `text-embedding-3-small`) and, when set,
`MAX_TOKENS`, before calling the emitter.

## Appendix B — verification performed for this report

* `chunks[:len(vectors)]` no-op when `vectors > chunks` — reproduced in a REPL.
* CLI index/query call plain `encode()` (lines 663, 783); no CLI caller of
  `encode_documents`/`encode_query` — confirmed by grep.
* Batch budget `= self.max_tokens` — confirmed at `embedder.py:376`.
* Prefixes parsed in `config.py` but consumed only by `OllamaEmbedder` and
  `build_faiss_index.py`, never by `cli.py` — confirmed by grep.

---

## Resolution log

All findings were fixed on 2026-07-04. Summary of the change per finding:

| # | Fix applied |
| --- | --- |
| F1 | `_cmd_index` now embeds via `encode_documents`, drops the no-op `chunks[:len(vectors)]` truncation, and **errors loudly** if `len(vectors) != len(chunks)` (misalignment can no longer be written silently). |
| F2 | `_cmd_index` uses `embedder.encode_documents(texts)`; `_cmd_query` uses `embedder.encode_query(text)`. Both fall back to `encode()` for OpenAI, so asymmetric backends now get their configured prefixes. |
| F3 | `configs/embedding.yaml`: OpenAI models' `chunking.max_tokens` 8191→**512**, `overlap` 1023→**64**; model-level `max_tokens: 8191` retained as the ceiling. |
| F4 | `TokenChunkingEmbedder` gains `REQUEST_TOKEN_BUDGET=300_000` + `MAX_INPUTS_PER_REQUEST=2048`; `_batches` packs to the request budget / input cap instead of the per-input `max_tokens`. |
| F5 | `_build_embedder` OpenAI branch now feeds the model's **tiktoken** encoder to the `Chunker` (`tokenize=lambda t: enc.encode(t)`), so chunk sizing is exact and `token_count` is real. |
| F6 | `config.load_embedding` clamps `overlap` to `max_tokens // 8` when an override would make `overlap >= max_tokens`, preventing the `Chunker` `ValueError`. |
| F7 | `config._env_ns` resolves `KBCRAFT_<NAME>` first (legacy bare names kept as aliases) for `MODEL` / `MAX_TOKENS` / `CHUNK_OVERLAP` / `OUTPUT_DIR`. The shell driver exports `KBCRAFT_*` and replaced `eval "$(…)"` with a checked temp-file `source`. |
| F8 | `test_cli_index_openai.py` rewritten to resolve params via `kbcraft.config.resolve_params` and run step 6 through `kbcraft query --json` — same source of truth and code path as the shell driver. Removed `_KNOWN_DIMS` and the `//8` overlap heuristic. |
| F9 | New `kbcraft query --json` (structured stdout, decorative logs to stderr). Both drivers parse JSON (no log scraping), assert result ids are in range, add a relevance assertion, and support a `KBCRAFT_TEST_SOURCE` fixture-dir override for cheap CI. |
| F10 | `OpenAICompatibleEmbedder` clients now pass `max_retries` (default 5, `OPENAI_MAX_RETRIES` override) for exponential backoff on 429/5xx. |

**Verification performed:**

* `ruff` clean; `black` formatted; **170 unit tests pass**.
* Config emitter shows `MAX_TOKENS=512 / CHUNK_OVERLAP=64`; F6 clamp verified
  (`MAX_TOKENS=50` → `CHUNK_OVERLAP=6`); legacy `MODEL=` alias still resolves.
* Full live end-to-end (real OpenAI API), **both** drivers:
  `ntotal == chunks == 479` (F1 holds — one vector per chunk), all checks pass,
  exit `0`. Retrieval improved with 512-token chunks: top-1 is now
  `src/kbcraft/chunker.py` (L2≈0.705) vs the previous README-first result.
