"""
Token counting utilities for embedding models.

Three backends are provided, from most to least accurate:

1. :class:`OllamaTokenizer` — exact counts via ``/api/tokenize`` (requires a
   running Ollama server).
2. :class:`HFTokenizer` — exact counts via a local HuggingFace ``tokenizers``
   vocabulary (requires the ``tokenizers`` package, already a transitive dep).
3. :class:`WhitespaceTokenizer` — word-split approximation, zero extra deps.

The :func:`get_tokenizer` factory returns the best available backend for a
given Ollama model name.  It can be plugged directly into :class:`Chunker`::

    from kbcraft.tokenizer import get_tokenizer
    from kbcraft.chunker import Chunker

    tok = get_tokenizer("all-minilm")
    chunker = Chunker(max_chunk_tokens=200, tokenize=tok.tokenize)

    # Standalone usage
    print(tok.count("hello world"))          # → 2
    print(tok.count_batch(["a b", "c"]))     # → [2, 1]
    print(tok.truncate("a b c d", 2))        # → "a b"
"""

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Model → HuggingFace tokenizer vocab mapping
# Vocabularies are downloaded on first use and cached by the tokenizers lib.
# ---------------------------------------------------------------------------

_HF_VOCAB: Dict[str, str] = {
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "bge-m3": "BAAI/bge-m3",
    "snowflake-arctic-embed": "Snowflake/snowflake-arctic-embed-m",
    "qwen3-embedding:0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embedding:4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding:8b": "Qwen/Qwen3-Embedding-8B",
}

# Fallback HF vocab used when the model isn't in _HF_VOCAB
_HF_DEFAULT = "bert-base-uncased"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseTokenizer(ABC):
    """Common interface for all tokenizer backends.

    Every backend exposes the same three methods so they are interchangeable
    and can be injected into :class:`~kbcraft.chunker.Chunker`.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model this tokenizer was built for."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Short label identifying the backend (``'ollama'``, ``'hf'``, ``'whitespace'``)."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Split *text* into tokens and return them as strings.

        This signature matches the ``tokenize`` parameter of
        :class:`~kbcraft.chunker.Chunker`.
        """

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self.tokenize(text))

    def count_batch(self, texts: List[str]) -> List[int]:
        """Return token counts for every string in *texts*."""
        return [self.count(t) for t in texts]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Return *text* truncated to at most *max_tokens* tokens.

        Reconstruction joins tokens with a single space, which is exact for
        whitespace-tokenized text and a close approximation for subword
        tokenizers.  If you need byte-perfect reconstruction, use the
        backend's decode method directly.
        """
        tokens = self.tokenize(text)
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"


# ---------------------------------------------------------------------------
# Backend 1 — Ollama /api/tokenize
# ---------------------------------------------------------------------------


class OllamaTokenizer(BaseTokenizer):
    """Exact token counts via Ollama's ``/api/tokenize`` endpoint.

    Requires a running Ollama server.  Token IDs are returned as integers;
    :meth:`tokenize` converts each ID to its string representation so the
    interface matches the other backends.

    Args:
        model:   Ollama model name (must be pulled).
        host:    Ollama base URL. Default: ``http://localhost:11434``.
        timeout: HTTP request timeout in seconds. Default: ``10``.
    """

    def __init__(
        self,
        model: str = "all-minilm",
        host: str = "http://localhost:11434",
        timeout: float = 10.0,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def backend(self) -> str:
        return "ollama"

    def tokenize(self, text: str) -> List[str]:
        """Call ``/api/tokenize`` and return token IDs as strings."""
        payload = json.dumps({"model": self._model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self._host}/api/tokenize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._host}. "
                "Make sure Ollama is running: ollama serve"
            ) from exc
        # Response: {"tokens": [1234, 5678, ...]}
        return [str(t) for t in data.get("tokens", [])]


# ---------------------------------------------------------------------------
# Backend 2 — HuggingFace tokenizers (local vocab)
# ---------------------------------------------------------------------------


class HFTokenizer(BaseTokenizer):
    """Exact token counts using a local HuggingFace tokenizer vocabulary.

    The ``tokenizers`` package is a transitive dependency of ``chromadb``
    so it is always available.  The vocabulary is downloaded from the HF Hub
    on first use and then cached on disk.

    Args:
        model:   Ollama model name.  Used to look up the corresponding HF
                 repo in :data:`_HF_VOCAB`.  Falls back to ``bert-base-uncased``.
        hf_repo: Override the HF repo directly (e.g. ``"bert-base-uncased"``).
    """

    def __init__(self, model: str = "all-minilm", hf_repo: Optional[str] = None) -> None:
        self._model = model
        repo = hf_repo or _HF_VOCAB.get(model, _HF_DEFAULT)
        try:
            from tokenizers import Tokenizer as _HFTok

            self._tok = _HFTok.from_pretrained(repo)
            # Disable padding and truncation so count reflects actual tokens,
            # not a fixed-length padded/truncated sequence.
            self._tok.no_padding()
            self._tok.no_truncation()
        except Exception as exc:
            raise ImportError(
                f"Could not load HF tokenizer for {repo!r}. "
                "Install 'tokenizers' or use WhitespaceTokenizer."
            ) from exc

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def backend(self) -> str:
        return "hf"

    def tokenize(self, text: str) -> List[str]:
        """Encode *text* and return the token strings."""
        encoding = self._tok.encode(text)
        return encoding.tokens


# ---------------------------------------------------------------------------
# Backend 3 — whitespace (zero deps)
# ---------------------------------------------------------------------------


class WhitespaceTokenizer(BaseTokenizer):
    """Approximate token counts via whitespace splitting.

    One "token" = one whitespace-delimited word.  No extra dependencies.
    Overestimates for punctuation-heavy or code-heavy text (typical ratio is
    ~1.2–1.5 subword tokens per word for English prose and ~1.5–2.5 for
    source code).

    This is the same approximation used as the :class:`~kbcraft.chunker.Chunker`
    default.
    """

    def __init__(self, model: str = "unknown") -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def backend(self) -> str:
        return "whitespace"

    def tokenize(self, text: str) -> List[str]:
        return text.split()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PREFER_HF = True  # set False to skip HF and use whitespace by default


def get_tokenizer(
    model: str = "all-minilm",
    ollama_host: Optional[str] = None,
    prefer_ollama: bool = False,
) -> BaseTokenizer:
    """Return the best available tokenizer for *model*.

    Resolution order:

    1. :class:`OllamaTokenizer` — only when *prefer_ollama* is ``True`` **and**
       *ollama_host* is provided.  Exact but requires a live server.
    2. :class:`HFTokenizer` — if the ``tokenizers`` package is importable.
       Downloads the vocab once and caches it.
    3. :class:`WhitespaceTokenizer` — always available fallback.

    Args:
        model:         Ollama model name (e.g. ``"all-minilm"``).
        ollama_host:   Ollama base URL.  Required to enable the Ollama backend.
        prefer_ollama: Try the Ollama backend first (requires *ollama_host*).

    Returns:
        A :class:`BaseTokenizer` instance.

    Example::

        tok = get_tokenizer("all-minilm")
        print(tok.backend)          # "hf" if tokenizers is installed
        print(tok.count("hello world"))  # 3  (incl. [CLS] / [SEP] for BERT)
    """
    if prefer_ollama and ollama_host:
        return OllamaTokenizer(model=model, host=ollama_host)

    if _PREFER_HF:
        try:
            return HFTokenizer(model=model)
        except Exception:
            pass

    return WhitespaceTokenizer(model=model)
