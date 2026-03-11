"""
Tests for BaseEmbedder, HybridOutput, and ChromaEmbeddingFunction.
"""

import pytest

from kbcraft.embedder import BaseEmbedder, ChromaEmbeddingFunction, HybridOutput


# ---------------------------------------------------------------------------
# Minimal concrete implementation used across all tests
# ---------------------------------------------------------------------------


class DummyEmbedder(BaseEmbedder):
    """Returns deterministic vectors: [text_length / 100.0] * dim."""

    DIM = 4

    @property
    def embedding_dim(self) -> int:
        return self.DIM

    @property
    def model_name(self) -> str:
        return "dummy-embedder"

    def encode(self, texts):
        return [[len(t) / 100.0] * self.DIM for t in texts]


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestBaseEmbedderABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseEmbedder()  # type: ignore[abstract]

    def test_partial_implementation_raises(self):
        class Partial(BaseEmbedder):
            @property
            def embedding_dim(self):
                return 8

            @property
            def model_name(self):
                return "partial"

            # encode() not implemented

        with pytest.raises(TypeError):
            Partial()


# ---------------------------------------------------------------------------
# encode / encode_query / encode_documents
# ---------------------------------------------------------------------------


class TestEncode:
    def setup_method(self):
        self.embedder = DummyEmbedder()

    def test_encode_returns_one_vector_per_text(self):
        result = self.embedder.encode(["hello", "world", "foo"])
        assert len(result) == 3

    def test_encode_vector_length_equals_dim(self):
        result = self.embedder.encode(["hello"])
        assert len(result[0]) == DummyEmbedder.DIM

    def test_encode_query_returns_single_vector(self):
        vec = self.embedder.encode_query("hello")
        assert isinstance(vec, list)
        assert len(vec) == DummyEmbedder.DIM

    def test_encode_query_equals_encode_single(self):
        text = "hello world"
        assert self.embedder.encode_query(text) == self.embedder.encode([text])[0]

    def test_encode_documents_returns_list_of_vectors(self):
        result = self.embedder.encode_documents(["doc1", "doc2"])
        assert len(result) == 2
        assert all(len(v) == DummyEmbedder.DIM for v in result)

    def test_encode_documents_equals_encode_by_default(self):
        texts = ["doc1", "doc2"]
        assert self.embedder.encode_documents(texts) == self.embedder.encode(texts)


# ---------------------------------------------------------------------------
# model properties
# ---------------------------------------------------------------------------


class TestProperties:
    def setup_method(self):
        self.embedder = DummyEmbedder()

    def test_embedding_dim(self):
        assert self.embedder.embedding_dim == DummyEmbedder.DIM

    def test_model_name(self):
        assert self.embedder.model_name == "dummy-embedder"

    def test_max_tokens_default(self):
        assert self.embedder.max_tokens == 512

    def test_max_tokens_override(self):
        class BigContextEmbedder(DummyEmbedder):
            @property
            def max_tokens(self):
                return 8192

        assert BigContextEmbedder().max_tokens == 8192


# ---------------------------------------------------------------------------
# encode_hybrid — default raises NotImplementedError
# ---------------------------------------------------------------------------


class TestEncodeHybrid:
    def test_default_raises(self):
        with pytest.raises(NotImplementedError, match="hybrid"):
            DummyEmbedder().encode_hybrid(["hello"])

    def test_override_works(self):
        class HybridEmbedder(DummyEmbedder):
            def encode_hybrid(self, texts):
                dense = self.encode(texts)
                sparse = [{0: 1.0}] * len(texts)
                return HybridOutput(dense=dense, sparse=sparse, texts=texts)

        result = HybridEmbedder().encode_hybrid(["hello"])
        assert isinstance(result, HybridOutput)
        assert len(result.dense) == 1
        assert len(result.sparse) == 1
        assert result.texts == ["hello"]


# ---------------------------------------------------------------------------
# ChromaDB adapter
# ---------------------------------------------------------------------------


class TestChromaEmbeddingFunction:
    def setup_method(self):
        self.embedder = DummyEmbedder()
        self.ef = self.embedder.as_chroma_ef()

    def test_returns_chroma_ef_instance(self):
        assert isinstance(self.ef, ChromaEmbeddingFunction)

    def test_callable(self):
        result = self.ef(["hello", "world"])
        assert len(result) == 2

    def test_output_shape(self):
        result = self.ef(["abc"])
        assert len(result[0]) == DummyEmbedder.DIM

    def test_delegates_to_embedder_encode(self):
        texts = ["foo", "bar"]
        assert self.ef(texts) == self.embedder.encode(texts)

    def test_embedder_property(self):
        assert self.ef.embedder is self.embedder


# ---------------------------------------------------------------------------
# FAISS adapter
# ---------------------------------------------------------------------------


class TestFaissMatrix:
    def setup_method(self):
        self.embedder = DummyEmbedder()

    def test_returns_numpy_array(self):
        np = pytest.importorskip("numpy")
        matrix = self.embedder.as_faiss_matrix(["hello", "world"])
        assert isinstance(matrix, np.ndarray)

    def test_shape(self):
        pytest.importorskip("numpy")
        matrix = self.embedder.as_faiss_matrix(["a", "bb", "ccc"])
        assert matrix.shape == (3, DummyEmbedder.DIM)

    def test_dtype_float32(self):
        np = pytest.importorskip("numpy")
        matrix = self.embedder.as_faiss_matrix(["hello"])
        assert matrix.dtype == np.float32

    def test_values_match_encode(self):
        np = pytest.importorskip("numpy")
        texts = ["hello", "world"]
        matrix = self.embedder.as_faiss_matrix(texts)
        expected = np.array(self.embedder.encode(texts), dtype=np.float32)
        assert np.allclose(matrix, expected)


# ---------------------------------------------------------------------------
# HybridOutput dataclass
# ---------------------------------------------------------------------------


class TestHybridOutput:
    def test_fields(self):
        h = HybridOutput(
            dense=[[0.1, 0.2]],
            sparse=[{42: 0.9}],
            texts=["hello"],
        )
        assert h.dense == [[0.1, 0.2]]
        assert h.sparse == [{42: 0.9}]
        assert h.texts == ["hello"]
