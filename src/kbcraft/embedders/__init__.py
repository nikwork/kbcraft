"""
Concrete embedding model implementations for kbcraft.

All classes extend BaseEmbedder from kbcraft.embedder and are
drop-in compatible with ChromaDB and FAISS via the adapter methods.
"""

from kbcraft.embedders.ollama import OllamaEmbedder

__all__ = ["OllamaEmbedder"]
