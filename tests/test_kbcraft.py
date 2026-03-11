"""
Tests for main kbcraft module.
"""

import kbcraft


def test_version_exists():
    """Test that version is defined."""
    assert hasattr(kbcraft, "__version__")
    assert isinstance(kbcraft.__version__, str)


def test_version_format():
    """Test that version follows semantic versioning."""
    version = kbcraft.__version__
    parts = version.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


class TestModuleImports:
    """Test that all submodules can be imported."""

    def test_import_cli(self):
        """Test cli module import."""
        from kbcraft import cli

        assert cli is not None

    def test_import_scaffold(self):
        """Test scaffold module import."""
        from kbcraft import scaffold

        assert scaffold is not None

    def test_import_organize(self):
        """Test selector module import."""
        from kbcraft import selector

        assert selector is not None

    def test_import_chunker(self):
        """Test chunker module import."""
        from kbcraft import chunker

        assert chunker is not None

    def test_import_embedder(self):
        """Test embedder module import."""
        from kbcraft import embedder

        assert embedder is not None

    def test_import_sync(self):
        """Test sync module import."""
        from kbcraft import sync

        assert sync is not None

    def test_import_utils(self):
        """Test utils module import."""
        from kbcraft import utils

        assert utils is not None

    def test_import_vector_stores(self):
        """Test vector_stores package import."""
        from kbcraft import vector_stores

        assert vector_stores is not None

    def test_import_vector_stores_chroma(self):
        """Test chroma vector store import."""
        from kbcraft.vector_stores import chroma

        assert chroma is not None

    def test_import_vector_stores_qdrant(self):
        """Test qdrant vector store import."""
        from kbcraft.vector_stores import qdrant

        assert qdrant is not None

    def test_import_vector_stores_pinecone(self):
        """Test pinecone vector store import."""
        from kbcraft.vector_stores import pinecone

        assert pinecone is not None

    def test_import_vector_stores_faiss(self):
        """Test faiss vector store import."""
        from kbcraft.vector_stores import faiss

        assert faiss is not None
