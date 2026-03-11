"""
Tests for OllamaEmbedder.

All HTTP calls are mocked — no Ollama server required.
"""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from kbcraft.embedder import ChromaEmbeddingFunction
from kbcraft.embedders.ollama import (
    OLLAMA_DIM_MAP,
    OLLAMA_DOCUMENT_PREFIX,
    OLLAMA_MAX_TOKENS,
    OLLAMA_QUERY_PREFIX,
    OllamaEmbedder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 768  # nomic-embed-text default


def fake_response(embeddings):
    """Return a mock urlopen context-manager that yields *embeddings*."""
    body = json.dumps({"embeddings": embeddings}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def make_vecs(n, dim=DIM):
    """Create *n* deterministic float vectors of length *dim*."""
    return [[float(i) / 100] * dim for i in range(n)]


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_default_model(self):
        e = OllamaEmbedder()
        assert "nomic-embed-text" in e.model_name

    def test_custom_model(self):
        e = OllamaEmbedder(model="bge-m3")
        assert "bge-m3" in e.model_name

    def test_known_embedding_dim(self):
        for model, dim in OLLAMA_DIM_MAP.items():
            assert OllamaEmbedder(model=model).embedding_dim == dim

    def test_unknown_model_probes_server(self):
        e = OllamaEmbedder(model="my-custom-model")
        probe_vecs = [[0.1] * 42]
        with patch("urllib.request.urlopen", return_value=fake_response(probe_vecs)):
            assert e.embedding_dim == 42

    def test_max_tokens_known(self):
        for model, tokens in OLLAMA_MAX_TOKENS.items():
            assert OllamaEmbedder(model=model).max_tokens == tokens

    def test_max_tokens_unknown_defaults_to_512(self):
        assert OllamaEmbedder(model="unknown-model").max_tokens == 512

    def test_host_trailing_slash_stripped(self):
        e = OllamaEmbedder(host="http://localhost:11434/")
        assert not e._host.endswith("/")


# ---------------------------------------------------------------------------
# encode — raw, no prefix
# ---------------------------------------------------------------------------


class TestEncode:
    def setup_method(self):
        self.embedder = OllamaEmbedder()

    def _patch(self, n):
        return patch(
            "urllib.request.urlopen",
            return_value=fake_response(make_vecs(n)),
        )

    def test_returns_list_of_vectors(self):
        with self._patch(2):
            result = self.embedder.encode(["hello", "world"])
        assert len(result) == 2

    def test_vector_length_equals_dim(self):
        with self._patch(1):
            result = self.embedder.encode(["hello"])
        assert len(result[0]) == DIM

    def test_no_prefix_in_request(self):
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(1))) as mock:
            self.embedder.encode(["hello"])
            payload = json.loads(mock.call_args[0][0].data)
        assert payload["input"] == ["hello"]

    def test_model_in_request(self):
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(1))) as mock:
            self.embedder.encode(["hello"])
            payload = json.loads(mock.call_args[0][0].data)
        assert payload["model"] == "nomic-embed-text"


# ---------------------------------------------------------------------------
# encode_query — injects query prefix
# ---------------------------------------------------------------------------


class TestEncodeQuery:
    def setup_method(self):
        self.embedder = OllamaEmbedder()
        self.expected_prefix = OLLAMA_QUERY_PREFIX["nomic-embed-text"]

    def test_returns_single_vector(self):
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(1))):
            vec = self.embedder.encode_query("find errors")
        assert len(vec) == DIM

    def test_prefix_prepended(self):
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(1))) as mock:
            self.embedder.encode_query("find errors")
            payload = json.loads(mock.call_args[0][0].data)
        assert payload["input"] == [self.expected_prefix + "find errors"]

    def test_custom_query_prefix(self):
        e = OllamaEmbedder(query_prefix="query: ")
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(1))) as mock:
            e.encode_query("test")
            payload = json.loads(mock.call_args[0][0].data)
        assert payload["input"] == ["query: test"]

    def test_no_prefix_model(self):
        e = OllamaEmbedder(model="bge-m3")
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(1))) as mock:
            e.encode_query("test")
            payload = json.loads(mock.call_args[0][0].data)
        assert payload["input"] == ["test"]  # BGE-M3 needs no prefix


# ---------------------------------------------------------------------------
# encode_documents — injects document prefix
# ---------------------------------------------------------------------------


class TestEncodeDocuments:
    def setup_method(self):
        self.embedder = OllamaEmbedder()
        self.expected_prefix = OLLAMA_DOCUMENT_PREFIX["nomic-embed-text"]

    def test_returns_list_of_vectors(self):
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(2))):
            result = self.embedder.encode_documents(["doc1", "doc2"])
        assert len(result) == 2

    def test_prefix_prepended_to_each(self):
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(2))) as mock:
            self.embedder.encode_documents(["doc1", "doc2"])
            payload = json.loads(mock.call_args[0][0].data)
        assert payload["input"] == [
            self.expected_prefix + "doc1",
            self.expected_prefix + "doc2",
        ]

    def test_custom_document_prefix(self):
        e = OllamaEmbedder(document_prefix="passage: ")
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(1))) as mock:
            e.encode_documents(["text"])
            payload = json.loads(mock.call_args[0][0].data)
        assert payload["input"] == ["passage: text"]


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    def test_large_input_split_into_batches(self):
        e = OllamaEmbedder(batch_size=2)
        calls = []

        def mock_urlopen(req, timeout=None):
            payload = json.loads(req.data)
            n = len(payload["input"])
            calls.append(n)
            return fake_response(make_vecs(n))

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = e.encode(["a", "b", "c", "d", "e"])

        assert len(result) == 5
        assert calls == [2, 2, 1]  # 3 batches

    def test_single_batch_when_small(self):
        e = OllamaEmbedder(batch_size=32)
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(3))) as mock:
            e.encode(["a", "b", "c"])
        assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_connection_error_when_server_down(self):
        e = OllamaEmbedder()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(ConnectionError, match="ollama serve"):
                e.encode(["hello"])

    def test_error_message_includes_host(self):
        e = OllamaEmbedder(host="http://remote-host:11434")
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            with pytest.raises(ConnectionError, match="remote-host"):
                e.encode(["hello"])


# ---------------------------------------------------------------------------
# ChromaDB and FAISS adapters (inherited from BaseEmbedder)
# ---------------------------------------------------------------------------


class TestAdapters:
    def setup_method(self):
        self.embedder = OllamaEmbedder()

    def test_as_chroma_ef(self):
        ef = self.embedder.as_chroma_ef()
        assert isinstance(ef, ChromaEmbeddingFunction)

    def test_chroma_ef_calls_encode(self):
        ef = self.embedder.as_chroma_ef()
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(2))):
            result = ef(["a", "b"])
        assert len(result) == 2

    def test_as_faiss_matrix(self):
        np = pytest.importorskip("numpy")
        with patch("urllib.request.urlopen", return_value=fake_response(make_vecs(3))):
            matrix = self.embedder.as_faiss_matrix(["a", "b", "c"])
        assert matrix.shape == (3, DIM)
        assert matrix.dtype == np.float32
