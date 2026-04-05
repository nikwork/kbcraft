"""
OpenAI embedding implementation.

Thin wrapper around :class:`~kbcraft.embedder.TokenChunkingEmbedder` that
pre-fills OpenAI defaults, reads credentials from the environment, and
auto-splits long texts into non-overlapping chunks using ``tiktoken``.

Supported models (non-exhaustive):

* ``text-embedding-3-small``  — 1536 dims, 8191 token limit, cheapest
* ``text-embedding-3-large``  — 3072 dims, 8191 token limit, highest quality
* ``text-embedding-ada-002``  — 1536 dims, 8191 token limit, legacy

Quickstart::

    from kbcraft.embedders import OpenAIEmbedder

    embedder = OpenAIEmbedder()                              # reads OPENAI_API_KEY
    embedder = OpenAIEmbedder(model="text-embedding-3-large")
    embedder = OpenAIEmbedder(token="sk-...")                # explicit token

    vectors = embedder.encode(["Hello world"])
    vectors = await embedder.encode_async(["Hello world"])
"""

import os
from typing import List, Optional

from kbcraft.embedder import TokenChunkingEmbedder

#: Embedding dimensions for known OpenAI models.
OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

#: Maximum input tokens per request for known OpenAI embedding models.
OPENAI_MAX_TOKENS = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}

#: tiktoken encoding name for each model (falls back to cl100k_base).
OPENAI_ENCODING = {
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
}

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_MAX_TOKENS = 8191
_DEFAULT_ENCODING = "cl100k_base"


class OpenAIEmbedder(TokenChunkingEmbedder):
    """Embed text using the OpenAI Embeddings API.

    Long texts are automatically split into non-overlapping chunks using
    ``tiktoken`` before being sent to the API.  ``encode()`` returns **one
    vector per chunk**, so the output list can be longer than the input list
    when any text exceeds :attr:`max_tokens`.

    The tokenizer is loaded lazily on first use and cached for the lifetime
    of the embedder instance.

    Args:
        model:  Model name. Default: ``"text-embedding-3-small"``.
        token:  OpenAI API key. Falls back to the ``OPENAI_API_KEY``
                environment variable when not provided.

    Example::

        embedder = OpenAIEmbedder()

        # Short texts — one vector per text
        vecs = embedder.encode(["Hello world", "Foo bar"])
        assert len(vecs) == 2

        # Long text — split into chunks, one vector per chunk
        long_doc = open("book.txt").read()
        chunk_vecs = embedder.encode([long_doc])   # len >= 1

        # Async
        vecs = await embedder.encode_async(["Hello world"])

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
        model: str = _DEFAULT_MODEL,
        token: str = "",
    ) -> None:
        resolved_token = token or os.environ.get("OPENAI_API_KEY", "")
        super().__init__(model=model, token=resolved_token)
        self._dim: Optional[int] = OPENAI_DIMS.get(model)
        self._enc = None  # tiktoken.Encoding, loaded lazily

    # ------------------------------------------------------------------
    # BaseEmbedder overrides
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.encode(["probe"])[0])
        return self._dim

    @property
    def max_tokens(self) -> int:
        return OPENAI_MAX_TOKENS.get(self._model, _DEFAULT_MAX_TOKENS)

    # ------------------------------------------------------------------
    # TokenChunkingEmbedder interface — tiktoken tokenizer
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """Lazily loaded ``tiktoken.Encoding`` for the active model."""
        if self._enc is None:
            try:
                import tiktoken
            except ImportError as exc:
                raise ImportError(
                    "tiktoken is required for token counting and chunking. "
                    "Install it with: pip install tiktoken"
                ) from exc
            encoding_name = OPENAI_ENCODING.get(self._model, _DEFAULT_ENCODING)
            self._enc = tiktoken.get_encoding(encoding_name)
        return self._enc

    def count_tokens(self, text: str) -> int:
        """Return the exact token count for *text* using tiktoken."""
        return len(self.tokenizer.encode(text))

    def split_chunks(self, text: str) -> List[str]:
        """Split *text* into non-overlapping chunks that each fit within :attr:`max_tokens`.

        A single text that already fits returns a one-element list.
        """
        token_ids = self.tokenizer.encode(text)
        limit = self.max_tokens
        if len(token_ids) <= limit:
            return [text]
        chunks = []
        for i in range(0, len(token_ids), limit):
            chunks.append(self.tokenizer.decode(token_ids[i : i + limit]))
        return chunks
