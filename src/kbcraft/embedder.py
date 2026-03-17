"""
Embedding generation for document chunks.

Defines BaseEmbedder — the abstract interface for all embedding models —
plus HybridOutput for models that support both dense and sparse vectors
(e.g. BGE-M3), and lightweight adapters for ChromaDB and FAISS.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HybridOutput:
    """Output from a hybrid embedding model (dense + sparse vectors).

    Used by models that produce both dense semantic vectors and sparse
    lexical weights in a single pass (e.g. BGE-M3 via FlagEmbedding).

    Attributes:
        dense:  Dense float vectors, shape ``(N, embedding_dim)``.
                Used for ANN (approximate nearest-neighbour) search.
        sparse: Sparse lexical weights per text. Each entry maps
                ``token_id -> weight``, equivalent to BM25-style keyword
                importance. Used for exact-token matching in code search.
        texts:  Original input texts (preserved for reference).
    """

    dense: List[List[float]]
    sparse: List[Dict[int, float]]
    texts: List[str]


class BaseEmbedder(ABC):
    """Abstract base class for all embedding models.

    Subclass this to add a new embedding backend. Two abstract members
    must be implemented:

    - :py:meth:`encode` — batch-encode strings into float vectors
    - :py:attr:`embedding_dim` — output vector dimensionality
    - :py:attr:`model_name` — human-readable model identifier

    All other methods have defaults that delegate to ``encode()`` and can
    be overridden when a model needs query/document asymmetry (E5, Jina,
    GTE-Qwen2) or hybrid output (BGE-M3).

    ChromaDB adapter::

        collection = client.get_or_create_collection(
            "my_docs",
            embedding_function=embedder.as_chroma_ef(),
        )

    FAISS adapter::

        import faiss
        index = faiss.IndexFlatL2(embedder.embedding_dim)
        index.add(embedder.as_faiss_matrix(documents))

    Minimal concrete implementation::

        class MyEmbedder(BaseEmbedder):
            @property
            def embedding_dim(self) -> int:
                return 1024

            @property
            def model_name(self) -> str:
                return "my-model-name"

            def encode(self, texts: List[str]) -> List[List[float]]:
                return my_model.embed(texts)
    """

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every subclass
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output float vectors."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. ``"BAAI/bge-m3"``)."""
        ...

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode *texts* for symmetric tasks (clustering, similarity).

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, one per input text.
            Each vector has exactly :py:attr:`embedding_dim` elements.
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides — sensible defaults, override when needed
    # ------------------------------------------------------------------

    @property
    def max_tokens(self) -> int:
        """Maximum input tokens accepted by this model. Default: 512."""
        return 512

    def encode_query(self, text: str) -> List[float]:
        """Encode a single search query.

        Override to inject query-specific instruction prefixes required
        by some models (e.g. ``"search_query: "`` for E5 / Nomic).
        The default delegates to :py:meth:`encode`.
        """
        return self.encode([text])[0]

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """Encode documents for indexing.

        Override to inject document-specific prefixes required by some
        models (e.g. ``"search_document: "`` for E5 / Nomic).
        The default delegates to :py:meth:`encode`.
        """
        return self.encode(texts)

    def encode_hybrid(self, texts: List[str]) -> HybridOutput:
        """Return both dense and sparse vectors in a single pass.

        Only models with built-in sparse retrieval support should override
        this (currently BGE-M3 via FlagEmbedding). The default raises
        :py:exc:`NotImplementedError`.

        Raises:
            NotImplementedError: For models that do not support hybrid output.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support hybrid (dense+sparse) encoding. "
            "Use BGEM3Embedder for hybrid output."
        )

    # ------------------------------------------------------------------
    # Vector store adapters
    # ------------------------------------------------------------------

    def as_chroma_ef(self) -> "ChromaEmbeddingFunction":
        """Return a ChromaDB-compatible ``EmbeddingFunction`` wrapper.

        The returned object can be passed directly as the
        ``embedding_function`` argument when creating or getting a
        ChromaDB collection::

            collection = client.get_or_create_collection(
                "my_docs",
                embedding_function=embedder.as_chroma_ef(),
            )

        ChromaDB calls the function with ``List[str]`` for both indexed
        documents and queries. This wrapper routes all calls through
        :py:meth:`encode`.
        """
        return ChromaEmbeddingFunction(self)

    def as_faiss_matrix(self, texts: List[str]):
        """Encode *texts* and return a ``float32`` numpy array for FAISS.

        The returned array has shape ``(len(texts), embedding_dim)`` and
        dtype ``numpy.float32``, which is required by
        ``faiss.Index.add()``::

            import faiss
            index = faiss.IndexFlatL2(embedder.embedding_dim)
            index.add(embedder.as_faiss_matrix(documents))

            # Query
            import numpy as np
            q = np.array([embedder.encode_query(query)], dtype=np.float32)
            distances, indices = index.search(q, k=5)

        Args:
            texts: Documents or queries to encode.

        Returns:
            ``numpy.ndarray`` of shape ``(len(texts), embedding_dim)``,
            dtype ``float32``.

        Raises:
            ImportError: If numpy is not installed.
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "numpy is required for FAISS encoding. " "Install it with: pip install numpy"
            ) from exc

        return np.array(self.encode(texts), dtype=np.float32)


class OpenAICompatibleEmbedder(BaseEmbedder):
    """Embedder backed by any OpenAI-compatible ``/v1/embeddings`` endpoint.

    Works with OpenAI, Azure OpenAI, Ollama, LocalAI, vLLM, LM Studio,
    and any other server that speaks the OpenAI embeddings API.

    Args:
        base_url: Base URL of the API server, e.g. ``"https://api.openai.com/v1"``
                  or ``"http://localhost:11434/v1"`` for Ollama.
        model:    Model name passed in the request body, e.g.
                  ``"text-embedding-3-small"`` or ``"nomic-embed-text"``.
        token:    Bearer token for the ``Authorization`` header.
                  Pass ``None`` or ``""`` to skip auth (local servers).

    Example::

        embedder = OpenAICompatibleEmbedder(
            model="text-embedding-3-small",
            token="sk-...",
        )
        embedder = OpenAICompatibleEmbedder(
            model="nomic-embed-text",
            token="",
            base_url="http://localhost:11434/v1",   # custom server
        )
        vectors = embedder.encode(["Hello world"])
        vectors = await embedder.encode_async(["Hello world"])
    """

    def __init__(self, model: str, token: str, base_url: str = "") -> None:
        self._base_url = base_url or ""
        self._model = model
        self._token = token or ""
        self._dim: int = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.encode(["probe"])[0])
        return self._dim

    def _client(self):
        from openai import OpenAI

        kwargs = {"api_key": self._token or "nokeyneeded"}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

    def _async_client(self):
        from openai import AsyncOpenAI

        kwargs = {"api_key": self._token or "nokeyneeded"}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return AsyncOpenAI(**kwargs)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Synchronously embed *texts*, sending all inputs in one request."""
        response = self._client().embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    async def encode_async(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed *texts*, sending all inputs in one request."""
        response = await self._async_client().embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


class ChromaEmbeddingFunction:
    """Wraps a :class:`BaseEmbedder` as a ChromaDB ``EmbeddingFunction``.

    ChromaDB expects a callable that accepts ``List[str]`` and returns
    ``List[List[float]]``. This wrapper satisfies that protocol without
    requiring ``chromadb`` to be installed at import time.

    Instantiate via :py:meth:`BaseEmbedder.as_chroma_ef` rather than
    directly::

        ef = embedder.as_chroma_ef()
        collection = client.get_or_create_collection(
            "docs", embedding_function=ef
        )
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embedder.encode(input)

    @property
    def embedder(self) -> BaseEmbedder:
        """The underlying :class:`BaseEmbedder` instance."""
        return self._embedder
