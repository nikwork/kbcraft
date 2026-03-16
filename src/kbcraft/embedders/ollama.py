"""
Ollama embedding implementation.

Wraps a locally running Ollama server via its REST API.
Zero Python ML dependencies — all inference happens inside Ollama.

Quickstart::

    ollama pull nomic-embed-text
    # Ollama server starts automatically or: ollama serve

    from kbcraft.embedders import OllamaEmbedder

    embedder = OllamaEmbedder()                          # nomic-embed-text
    embedder = OllamaEmbedder(model="bge-m3")            # BGE-M3
    embedder = OllamaEmbedder(host="http://gpu-box:11434")
"""

import json
import urllib.error
import urllib.request
from typing import Dict, Iterator, List, Optional

from kbcraft.embedder import BaseEmbedder

# ---------------------------------------------------------------------------
# Known model metadata
# ---------------------------------------------------------------------------

#: Embedding dimensions for models available via Ollama.
OLLAMA_DIM_MAP: Dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
    "snowflake-arctic-embed": 1024,
    "all-minilm": 384,
}

#: Maximum input tokens for models available via Ollama.
OLLAMA_MAX_TOKENS: Dict[str, int] = {
    "nomic-embed-text": 8192,
    "mxbai-embed-large": 512,
    "bge-m3": 8192,
    "snowflake-arctic-embed": 512,
    "all-minilm": 256,
}

#: Instruction prefix injected by encode_query() per model.
#: Empty string means no prefix is needed (e.g. BGE-M3).
OLLAMA_QUERY_PREFIX: Dict[str, str] = {
    "nomic-embed-text": "search_query: ",
    "mxbai-embed-large": "Represent this sentence for searching relevant passages: ",
    "bge-m3": "",
    "snowflake-arctic-embed": "Represent this sentence for searching relevant passages: ",
    "all-minilm": "",
}

#: Instruction prefix injected by encode_documents() per model.
OLLAMA_DOCUMENT_PREFIX: Dict[str, str] = {
    "nomic-embed-text": "search_document: ",
    "mxbai-embed-large": "",
    "bge-m3": "",
    "snowflake-arctic-embed": "",
    "all-minilm": "",
}


class OllamaEmbedder(BaseEmbedder):
    """Embed text using a locally running Ollama server.

    Default model is ``nomic-embed-text`` — 768 dimensions, 8192-token
    context, good EN + RU + Code quality, Apache 2.0 license.

    The embedder uses the Ollama ``/api/embed`` REST endpoint directly
    via stdlib ``urllib``, so **no extra Python packages are required**.

    Args:
        model:    Ollama model name (must be pulled first with
                  ``ollama pull <model>``). Default: ``"nomic-embed-text"``.
        host:     Base URL of the Ollama server.
                  Default: ``"http://localhost:11434"``.
        timeout:  HTTP request timeout in seconds. Default: ``60``.
        batch_size: Maximum texts per HTTP request. Large batches are
                  split automatically. Default: ``32``.
        query_prefix: Override the instruction prefix used by
                  :py:meth:`encode_query`. ``None`` uses the model default.
        document_prefix: Override the instruction prefix used by
                  :py:meth:`encode_documents`. ``None`` uses the model
                  default.

    Example::

        embedder = OllamaEmbedder()

        # Semantic search — different paths for query vs. indexed docs
        query_vec = embedder.encode_query("how to handle errors in Python")
        doc_vecs  = embedder.encode_documents(["try:\\n    ...\\nexcept Exception: ..."])

        # Plain encoding (no prefix)
        vecs = embedder.encode(["hello world"])

        # ChromaDB
        collection = client.get_or_create_collection(
            "docs", embedding_function=embedder.as_chroma_ef()
        )

        # FAISS
        import faiss
        index = faiss.IndexFlatL2(embedder.embedding_dim)
        index.add(embedder.as_faiss_matrix(documents))
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
        batch_size: int = 32,
        query_prefix: Optional[str] = None,
        document_prefix: Optional[str] = None,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._batch_size = batch_size
        self._query_prefix = (
            query_prefix if query_prefix is not None else OLLAMA_QUERY_PREFIX.get(model, "")
        )
        self._document_prefix = (
            document_prefix
            if document_prefix is not None
            else OLLAMA_DOCUMENT_PREFIX.get(model, "")
        )
        # Cached after first call if model is unknown
        self._dim: Optional[int] = OLLAMA_DIM_MAP.get(model)

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            # Probe the server with a dummy text to discover dimension
            self._dim = len(self._post(["_probe_"])[0])
        return self._dim

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    @property
    def max_tokens(self) -> int:
        return OLLAMA_MAX_TOKENS.get(self._model, 512)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode *texts* without any instruction prefix.

        Use this for symmetric tasks (clustering, deduplication).
        For asymmetric retrieval (query vs. document), prefer
        :py:meth:`encode_query` and :py:meth:`encode_documents`.
        """
        return self._encode_with_prefix(texts, prefix="")

    def encode_query(self, text: str) -> List[float]:
        """Encode a single search query with the model's query prefix.

        For ``nomic-embed-text`` this prepends ``"search_query: "``,
        which is required for best retrieval quality.
        """
        return self._encode_with_prefix([text], prefix=self._query_prefix)[0]

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """Encode documents for indexing with the model's document prefix.

        For ``nomic-embed-text`` this prepends ``"search_document: "``
        to each text.
        """
        return self._encode_with_prefix(texts, prefix=self._document_prefix)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_with_prefix(self, texts: List[str], prefix: str) -> List[List[float]]:
        """Apply *prefix* to every text then batch-encode."""
        prefixed = [prefix + t for t in texts] if prefix else texts
        results: List[List[float]] = []
        for batch in self._batches(prefixed):
            results.extend(self._post(batch))
        return results

    def _batches(self, texts: List[str]) -> Iterator[List[str]]:
        """Yield successive slices of *texts* of size ``self._batch_size``."""
        for i in range(0, len(texts), self._batch_size):
            yield texts[i : i + self._batch_size]

    def _post(self, texts: List[str]) -> List[List[float]]:
        """POST *texts* to ``/api/embed`` and return the embeddings."""
        payload = json.dumps({"model": self._model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{self._host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code} for model {self._model!r}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._host}. " "Make sure Ollama is running: ollama serve"
            ) from exc

        return data["embeddings"]
