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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load config files. " "Install it with: pip install pyyaml"
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
class OpenAICompatibleBackendConfig:
    base_url: str
    token: str


@dataclass
class TokenizerConfig:
    prefer_ollama: bool = False
    hf_fallback_repo: str = "bert-base-uncased"


@dataclass
class EmbeddingConfig:
    active_model: str
    ollama: OllamaBackendConfig
    openai_compatible: OpenAICompatibleBackendConfig
    models: Dict[str, ModelConfig]
    tokenizer: TokenizerConfig

    @property
    def model(self) -> ModelConfig:
        """Return the active :class:`ModelConfig`."""
        return self.models[self.active_model]


# ── Storage data models ────────────────────────────────────────────────────────


@dataclass
class StoragePaths:
    chunks: str = "chunks"
    embeddings: str = "embeddings"
    indexes: str = "indexes"
    exports: str = "exports"
    reports: str = "reports"


@dataclass
class LocalStorageConfig:
    root: Path
    paths: StoragePaths


@dataclass
class S3TransferConfig:
    multipart_threshold: int = 104857600   # 100 MB
    multipart_chunksize: int = 26214400    # 25 MB
    max_concurrency: int = 10
    max_bandwidth: int = None
    use_threads: bool = True


@dataclass
class S3StorageConfig:
    region_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str
    profile_name: str
    endpoint_url: str
    transfer: S3TransferConfig


@dataclass
class StorageConfig:
    active_backend: str
    local: LocalStorageConfig
    s3: S3StorageConfig

    @property
    def backend(self) -> "LocalStorageConfig | S3StorageConfig":
        """Return the config for the active backend."""
        return getattr(self, self.active_backend)


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

        oac_raw = raw.get("backends", {}).get("openai_compatible", {})
        openai_compatible = OpenAICompatibleBackendConfig(
            base_url=_env(
                "OPENAI_COMPATIBLE_BASE_URL", oac_raw.get("base_url", "http://localhost:11434/v1")
            ),
            token=_env("OPENAI_COMPATIBLE_TOKEN", oac_raw.get("token", "")),
        )

        tok_raw = raw.get("tokenizer", {})
        tokenizer = TokenizerConfig(
            prefer_ollama=tok_raw.get("prefer_ollama", False),
            hf_fallback_repo=tok_raw.get("hf_fallback_repo", "bert-base-uncased"),
        )

        models = {name: self._parse_model(name, cfg) for name, cfg in raw.get("models", {}).items()}

        active_model = _env("MODEL", raw.get("active_model"))
        if active_model not in models:
            raise KeyError(
                f"Model {active_model!r} not found in embedding.yaml. " f"Available: {list(models)}"
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
            openai_compatible=openai_compatible,
            models=models,
            tokenizer=tokenizer,
        )

    def load_storage(self) -> StorageConfig:
        """Load ``storage.yaml`` and return a fully resolved :class:`StorageConfig`.

        Resolution order for every value: env var → yaml → dataclass default.
        """
        raw = _load_yaml(self._dir / "storage.yaml")
        backends = raw.get("backends", {})

        local_raw = backends.get("local", {})
        paths_raw = local_raw.get("paths", {})
        paths = StoragePaths(
            chunks=paths_raw.get("chunks", "chunks"),
            embeddings=paths_raw.get("embeddings", "embeddings"),
            indexes=paths_raw.get("indexes", "indexes"),
            exports=paths_raw.get("exports", "exports"),
            reports=paths_raw.get("reports", "reports"),
        )
        local = LocalStorageConfig(
            root=Path(_env("STORAGE_LOCAL_ROOT", local_raw.get("root", "./artifacts"))),
            paths=paths,
        )

        s3_raw = backends.get("s3", {})
        transfer_raw = s3_raw.get("transfer", {})
        transfer = S3TransferConfig(
            multipart_threshold=transfer_raw.get("multipart_threshold", 104857600),
            multipart_chunksize=transfer_raw.get("multipart_chunksize", 26214400),
            max_concurrency=transfer_raw.get("max_concurrency", 10),
            max_bandwidth=transfer_raw.get("max_bandwidth"),
            use_threads=transfer_raw.get("use_threads", True),
        )
        s3 = S3StorageConfig(
            region_name=_env("AWS_DEFAULT_REGION", s3_raw.get("region_name", "us-east-1")),
            aws_access_key_id=_env("AWS_ACCESS_KEY_ID", s3_raw.get("aws_access_key_id", "")),
            aws_secret_access_key=_env(
                "AWS_SECRET_ACCESS_KEY", s3_raw.get("aws_secret_access_key", "")
            ),
            aws_session_token=_env("AWS_SESSION_TOKEN", s3_raw.get("aws_session_token", "")),
            profile_name=_env("AWS_PROFILE", s3_raw.get("profile_name", "")),
            endpoint_url=_env("STORAGE_S3_ENDPOINT_URL", s3_raw.get("endpoint_url", "")),
            transfer=transfer,
        )

        active_backend = _env("STORAGE_BACKEND", raw.get("active_backend", "local"))

        return StorageConfig(
            active_backend=active_backend,
            local=local,
            s3=s3,
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
