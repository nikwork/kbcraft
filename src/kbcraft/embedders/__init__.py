"""
Concrete embedding model implementations for kbcraft.

All classes extend BaseEmbedder from kbcraft.embedder and are
drop-in compatible with ChromaDB and FAISS via the adapter methods.
"""

from kbcraft.embedder import TokenChunkingEmbedder
from kbcraft.embedders.ollama import OllamaEmbedder
from kbcraft.embedders.openai import OpenAIEmbedder
from kbcraft.embedders.qwen import Qwen3Embedder

__all__ = ["TokenChunkingEmbedder", "OllamaEmbedder", "OpenAIEmbedder", "Qwen3Embedder"]
