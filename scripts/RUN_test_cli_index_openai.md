# Running `scripts/test_cli_index_openai.py`

End-to-end CLI smoke test for `kbcraft index` using **OpenAI embeddings**.

It runs the real pipeline through the `kbcraft` CLI binary (not Python imports):

```
kbcraft index ./  --embedder openai --model text-embedding-3-small --output <tmp>
```

and checks that:

1. the CLI exits with code `0`,
2. `*.faiss`, `*_chunks.json`, `*_meta.json` are written,
3. the FAISS index loads and has the expected vector count / dimension,
4. a live smoke-test query returns ranked results.

> ⚠️ This makes **real OpenAI API calls** (embeds the whole repo — a few hundred
> chunks). It needs network access and a valid API key, and costs a small amount.

## Prerequisites

- The project virtualenv must be installed (see repo root `CLAUDE.md`). This repo
  uses an in-project venv at `.venv/`:

  ```bash
  poetry install --all-extras
  ```

- An OpenAI API key, provided **either** as an environment variable **or** in a
  `.env` file at the project root:

  ```bash
  # option A: export in your shell
  export OPENAI_API_KEY=sk-...

  # option B: put it in .env at the repo root (auto-loaded via python-dotenv)
  echo 'OPENAI_API_KEY=sk-...' >> .env
  ```

  A shell environment variable takes precedence over `.env`.

## Run it

The script shells out to the `kbcraft` binary, so that binary must be on `PATH`.
The simplest way is to activate the venv first:

```bash
cd /root/dev/kbcraft
source .venv/bin/activate
python scripts/test_cli_index_openai.py
```

Or, without activating, run the venv's Python and put its `bin/` on `PATH` so the
subprocess can find `kbcraft`:

```bash
cd /root/dev/kbcraft
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/test_cli_index_openai.py
```

## Configuration (environment variables)

| Variable                       | Default                  | Effect                                                        |
| ------------------------------ | ------------------------ | ------------------------------------------------------------- |
| `OPENAI_API_KEY`               | — (required)             | OpenAI credentials.                                           |
| `OPENAI_EMBEDDING_MODEL`       | `text-embedding-3-small` | Embedding model to use.                                       |
| `OPENAI_EMBEDDING_MAX_TOKENS`  | `8191`                   | Chunk size in tokens (overlap is `MAX_TOKENS // 8`).          |
| `SAVE_VECTORDB`                | unset                    | If `1`/`true`/`yes`, write output to `vectordb/` and keep it. |

By default the index is written to a temporary directory and **deleted** after
the run. To persist it:

```bash
SAVE_VECTORDB=1 python scripts/test_cli_index_openai.py
# -> output kept in ./vectordb/ (test_index_openai.faiss, *_chunks.json, *_meta.json)
```

Example with a larger model:

```bash
OPENAI_EMBEDDING_MODEL=text-embedding-3-large python scripts/test_cli_index_openai.py
```

## Expected output

On success the run ends with:

```
============================================================
  All checks passed ✓
============================================================
```

The process exits `0` when all checks pass and `1` otherwise, so it can be used
directly in CI.

## Troubleshooting

- **`kbcraft: command not found`** — the venv isn't active / not on `PATH`.
  Activate `.venv` or use the `PATH="$PWD/.venv/bin:$PATH"` form above.
- **Auth / `401` errors** — `OPENAI_API_KEY` is missing or invalid.
- **`ModuleNotFoundError` (faiss, openai, dotenv, kbcraft)** — dependencies not
  installed; run `poetry install --all-extras`.
