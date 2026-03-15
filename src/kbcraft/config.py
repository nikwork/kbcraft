"""
Config data models and factory for kbcraft.

Loads ``configs/embedding.yaml`` and ``configs/vector_store.yaml``,
applies environment variable overrides, and returns typed dataclass instances.

Usage::

    from pathlib import Path
    from kbcraft.config import ConfigFactory

    factory = ConfigFactory.from_project_root(Path(__file__).parent.parent)

    emb = factory.load_embedding()
    print(emb.active_model)          # "nomic-embed-text"
    print(emb.model.embedding_dim)   # 768
    print(emb.ollama.host)           # "http://localhost:11434"

    vs = factory.load_vector_store()
    print(vs.active_backend)         # "faiss"
    print(vs.backend.index_type)     # "flat_l2"  (FaissConfig)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load config files. "
            "Install it with: pip install pyyaml"
        ) from exc
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _env(key: str, default=None):
    """Return the env var value if set and non-empty, else *default*."""
    val = os.environ.get(key, "")
    return val if val else default


# ── Embedding data models ──────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    max_tokens: int
    overlap: int
    prepend_source: bool = True


@dataclass
class ModelConfig:
    name: str
    backend: str
    batch_size: int
    embedding_dim: int
    max_tokens: int
    hf_repo: str
    query_prefix: str
    document_prefix: str
    chunking: ChunkingConfig
    hybrid: bool = False


@dataclass
class OllamaBackendConfig:
    host: str
    timeout: float


@dataclass
class TokenizerConfig:
    prefer_ollama: bool = False
    hf_fallback_repo: str = "bert-base-uncased"


@dataclass
class EmbeddingConfig:
    active_model: str
    ollama: OllamaBackendConfig
    models: Dict[str, ModelConfig]
    tokenizer: TokenizerConfig

    @property
    def model(self) -> ModelConfig:
        """Return the active :class:`ModelConfig`."""
        return self.models[self.active_model]


# ── Vector store data models ───────────────────────────────────────────────────

@dataclass
class FaissConfig:
    index_type: str = "flat_l2"
    output_dir: Path = Path("faiss_index")
    ivf_nlist: int = 100
    ivf_nprobe: int = 10


@dataclass
class ChromaConfig:
    distance_function: str = "cosine"


@dataclass
class QdrantConfig:
    distance: str = "Cosine"


@dataclass
class PineconeConfig:
    environment: str = ""
    metric: str = "cosine"


@dataclass
class VectorStoreConfig:
    active_backend: str
    faiss: FaissConfig
    chroma: ChromaConfig
    qdrant: QdrantConfig
    pinecone: PineconeConfig

    @property
    def backend(self) -> "FaissConfig | ChromaConfig | QdrantConfig | PineconeConfig":
        """Return the config for the active backend."""
        return getattr(self, self.active_backend)


# ── Factory ────────────────────────────────────────────────────────────────────

class ConfigFactory:
    """Load and instantiate kbcraft config objects from the ``configs/`` directory.

    Environment variables override config file values (see yaml files for
    which env var maps to which setting).

    Args:
        configs_dir: Path to the directory containing ``embedding.yaml``
                     and ``vector_store.yaml``.

    Example::

        factory = ConfigFactory.from_project_root(PROJECT_ROOT)
        emb = factory.load_embedding()
        vs  = factory.load_vector_store()
    """

    def __init__(self, configs_dir: Path) -> None:
        self._dir = Path(configs_dir)

    @classmethod
    def from_project_root(cls, project_root: Path) -> "ConfigFactory":
        """Construct from the project root (looks for ``<root>/configs/``)."""
        return cls(Path(project_root) / "configs")

    # ── Public loaders ─────────────────────────────────────────────────────────

    def load_embedding(self) -> EmbeddingConfig:
        """Load ``embedding.yaml`` and return a fully resolved :class:`EmbeddingConfig`.

        Resolution order for every value: env var → yaml → dataclass default.
        """
        raw = _load_yaml(self._dir / "embedding.yaml")

        ollama_raw = raw.get("backends", {}).get("ollama", {})
        ollama = OllamaBackendConfig(
            host=_env("OLLAMA_HOST", ollama_raw.get("host", "http://localhost:11434")),
            timeout=float(ollama_raw.get("timeout", 60)),
        )

        tok_raw = raw.get("tokenizer", {})
        tokenizer = TokenizerConfig(
            prefer_ollama=tok_raw.get("prefer_ollama", False),
            hf_fallback_repo=tok_raw.get("hf_fallback_repo", "bert-base-uncased"),
        )

        models = {
            name: self._parse_model(name, cfg)
            for name, cfg in raw.get("models", {}).items()
        }

        active_model = _env("MODEL", raw.get("active_model"))
        if active_model not in models:
            raise KeyError(
                f"Model {active_model!r} not found in embedding.yaml. "
                f"Available: {list(models)}"
            )

        # Apply chunking env var overrides to the active model only
        active = models[active_model]
        if _env("MAX_TOKENS"):
            active.chunking.max_tokens = int(_env("MAX_TOKENS"))
        if _env("CHUNK_OVERLAP"):
            active.chunking.overlap = int(_env("CHUNK_OVERLAP"))

        return EmbeddingConfig(
            active_model=active_model,
            ollama=ollama,
            models=models,
            tokenizer=tokenizer,
        )

    def load_vector_store(self) -> VectorStoreConfig:
        """Load ``vector_store.yaml`` and return a :class:`VectorStoreConfig`."""
        raw = _load_yaml(self._dir / "vector_store.yaml")
        backends = raw.get("backends", {})

        faiss_raw = backends.get("faiss", {})
        output_dir = Path(_env("OUTPUT_DIR", faiss_raw.get("output_dir", "faiss_index")))
        ivf = faiss_raw.get("ivf", {})
        faiss = FaissConfig(
            index_type=faiss_raw.get("index_type", "flat_l2"),
            output_dir=output_dir,
            ivf_nlist=ivf.get("nlist", 100),
            ivf_nprobe=ivf.get("nprobe", 10),
        )

        chroma_raw = backends.get("chroma", {})
        chroma = ChromaConfig(
            distance_function=chroma_raw.get("distance_function", "cosine"),
        )

        qdrant_raw = backends.get("qdrant", {})
        qdrant = QdrantConfig(
            distance=qdrant_raw.get("distance", "Cosine"),
        )

        pinecone_raw = backends.get("pinecone", {})
        pinecone = PineconeConfig(
            environment=pinecone_raw.get("environment", ""),
            metric=pinecone_raw.get("metric", "cosine"),
        )

        active_backend = raw.get("active_backend", "faiss")

        return VectorStoreConfig(
            active_backend=active_backend,
            faiss=faiss,
            chroma=chroma,
            qdrant=qdrant,
            pinecone=pinecone,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_model(name: str, raw: dict) -> ModelConfig:
        chunking_raw = raw.get("chunking", {})
        chunking = ChunkingConfig(
            max_tokens=chunking_raw.get("max_tokens", 512),
            overlap=chunking_raw.get("overlap", 50),
            prepend_source=chunking_raw.get("prepend_source", True),
        )
        return ModelConfig(
            name=name,
            backend=raw.get("backend", "ollama"),
            batch_size=raw.get("batch_size", 32),
            embedding_dim=raw.get("embedding_dim", 0),
            max_tokens=raw.get("max_tokens", 512),
            hf_repo=raw.get("hf_repo", ""),
            query_prefix=raw.get("query_prefix", ""),
            document_prefix=raw.get("document_prefix", ""),
            chunking=chunking,
            hybrid=raw.get("hybrid", False),
        )
