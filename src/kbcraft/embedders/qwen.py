"""
Qwen3 embedding implementation via an OpenAI-compatible endpoint.

Supports all three Qwen3-Embedding size variants (0.6b / 4b / 8b).
Zero Python ML dependencies for inference — all inference happens on the
remote server.

The official Qwen3 tokenizer (loaded via ``transformers.AutoTokenizer``) is
used to:

- Count exact input tokens before each request.
- Automatically split texts that exceed the model's context window into
  non-overlapping chunks.  ``encode()`` returns one vector per chunk, so the
  output list may be longer than the input list when long texts are present.

Quickstart::

    # Serve via Ollama (exposes OpenAI-compatible /v1/embeddings)
    ollama pull qwen3-embedding:0.6b
    ollama serve

    from kbcraft.embedders import Qwen3Embedder

    embedder = Qwen3Embedder()                              # 0.6b, localhost Ollama
    embedder = Qwen3Embedder(variant="4b")                  # 4b variant
    embedder = Qwen3Embedder(base_url="http://gpu:11434/v1", token="sk-...")
"""

from typing import Dict, Iterator, List

from kbcraft.embedder import OpenAICompatibleEmbedder

# ---------------------------------------------------------------------------
# Known model metadata
# ---------------------------------------------------------------------------

#: Ollama model tag for each variant.
QWEN3_MODEL_TAG: Dict[str, str] = {
    "0.6b": "qwen3-embedding:0.6b",
    "4b": "qwen3-embedding:4b",
    "8b": "qwen3-embedding:8b",
}

#: Official HuggingFace repo for the tokenizer of each variant.
QWEN3_HF_REPO: Dict[str, str] = {
    "0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "4b": "Qwen/Qwen3-Embedding-4B",
    "8b": "Qwen/Qwen3-Embedding-8B",
}

#: Embedding dimensions per variant.
QWEN3_DIM: Dict[str, int] = {
    "0.6b": 1024,
    "4b": 2560,
    "8b": 4096,
}

#: Maximum input tokens per variant (all share the same 32k context).
QWEN3_MAX_TOKENS: Dict[str, int] = {
    "0.6b": 32768,
    "4b": 32768,
    "8b": 32768,
}

_VARIANTS = tuple(QWEN3_MODEL_TAG)


class Qwen3Embedder(OpenAICompatibleEmbedder):
    """Embed text using a Qwen3-Embedding model via an OpenAI-compatible server.

    All three size variants are supported via the *variant* parameter.
    Defaults to the lightweight ``0.6b`` model served by a local Ollama instance.

    Long texts are automatically split into non-overlapping chunks using the
    official Qwen3 ``transformers`` tokenizer before being sent to the server.
    ``encode()`` returns **one vector per chunk**, so the output list can be
    longer than the input list when any text exceeds :attr:`max_tokens`.

    The tokenizer is loaded lazily on first use and cached for the lifetime
    of the embedder instance.

    Args:
        variant:    Model size — ``"0.6b"``, ``"4b"``, or ``"8b"``.
                    Default: ``"0.6b"``.
        base_url:   Base URL of the OpenAI-compatible server.
                    Default: ``"http://localhost:11434/v1"`` (Ollama).
        token:      Bearer token for the ``Authorization`` header.
                    Pass ``None`` or ``""`` for unauthenticated local servers.
        Batches are sized dynamically: chunks are packed into each request
        until the next chunk would push the total token count over
        :attr:`max_tokens`.  This maximises throughput without overloading
        the server.

    Example::

        embedder = Qwen3Embedder()

        # Short texts — one vector per text
        vecs = embedder.encode(["hello world", "foo bar"])
        assert len(vecs) == 2

        # Long text — split into chunks, one vector per chunk
        long_doc = open("book.txt").read()
        chunk_vecs = embedder.encode([long_doc])   # len >= 1

        # Async
        chunk_vecs = await embedder.encode_async(["hello world"])

        # ChromaDB adapter
        collection = client.get_or_create_collection(
            "docs", embedding_function=embedder.as_chroma_ef()
        )

        # FAISS adapter
        import faiss
        index = faiss.IndexFlatL2(embedder.embedding_dim)
        index.add(embedder.as_faiss_matrix(documents))
    """

    def __init__(
        self,
        variant: str = "0.6b",
        base_url: str = "http://localhost:11434/v1",
        token: str = "",
    ) -> None:
        if variant not in QWEN3_MODEL_TAG:
            raise ValueError(f"Unknown Qwen3 variant {variant!r}. Choose one of: {_VARIANTS}")
        super().__init__(
            model=QWEN3_MODEL_TAG[variant],
            token=token,
            base_url=base_url,
        )
        self._variant = variant
        self._dim = QWEN3_DIM[variant]
        self._tok = None  # AutoTokenizer, loaded lazily

    # ------------------------------------------------------------------
    # BaseEmbedder overrides
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"qwen3/{QWEN3_MODEL_TAG[self._variant]}"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def max_tokens(self) -> int:
        return QWEN3_MAX_TOKENS[self._variant]

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Chunk long texts then encode, splitting into batches as needed.

        Returns one vector per chunk.  Texts within the context window produce
        exactly one chunk each; longer texts produce two or more.
        """
        chunks = self._expand_chunks(texts)
        results: List[List[float]] = []
        for batch in self._batches(chunks):
            results.extend(super().encode(batch))
        return results

    async def encode_async(self, texts: List[str]) -> List[List[float]]:
        """Chunk long texts then encode asynchronously, splitting into batches.

        Returns one vector per chunk (see :py:meth:`encode`).
        """
        chunks = self._expand_chunks(texts)
        results: List[List[float]] = []
        for batch in self._batches(chunks):
            results.extend(await super().encode_async(batch))
        return results

    # ------------------------------------------------------------------
    # Tokenizer (transformers.AutoTokenizer)
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """Lazily loaded ``AutoTokenizer`` for the active Qwen3 variant."""
        if self._tok is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for Qwen3 token counting and chunking. "
                    "Install it with: pip install transformers"
                ) from exc
            self._tok = AutoTokenizer.from_pretrained(QWEN3_HF_REPO[self._variant])
        return self._tok

    def count_tokens(self, text: str) -> int:
        """Return the exact token count for *text* using the Qwen3 tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_chunks(self, text: str) -> List[str]:
        """Split *text* into non-overlapping chunks that each fit within :attr:`max_tokens`.

        Uses ``AutoTokenizer`` with ``return_overflowing_tokens=True`` so the
        split boundaries are exact.  A single text that already fits returns a
        one-element list.
        """
        tok = self.tokenizer
        enc = tok(
            text,
            truncation=True,
            max_length=self.max_tokens,
            return_overflowing_tokens=True,
            stride=0,
            return_tensors=None,
            add_special_tokens=False,
        )
        # When return_overflowing_tokens=True the tokenizer returns a list of
        # encodings even for a single input, so input_ids is List[List[int]].
        return [tok.decode(ids, skip_special_tokens=True) for ids in enc["input_ids"]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_chunks(self, texts: List[str]) -> List[str]:
        """Expand *texts* by splitting any text that exceeds :attr:`max_tokens`."""
        chunks: List[str] = []
        for text in texts:
            chunks.extend(self.split_chunks(text))
        return chunks

    def _batches(self, chunks: List[str]) -> Iterator[List[str]]:
        """Yield batches of *chunks* sized by token budget.

        Chunks are packed greedily into each batch until the next chunk would
        push the running token total over :attr:`max_tokens`.  This maximises
        the number of texts per request without exceeding the model limit.
        """
        budget = self.max_tokens
        batch: List[str] = []
        batch_tokens = 0
        for chunk in chunks:
            n = self.count_tokens(chunk)
            if batch and batch_tokens + n > budget:
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(chunk)
            batch_tokens += n
        if batch:
            yield batch
