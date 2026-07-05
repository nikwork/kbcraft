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


def _env_ns(name: str, *aliases: str, default=None):
    """Resolve a config value from ``KBCRAFT_<NAME>`` first, then legacy aliases.

    Generic names like ``MODEL`` / ``OUTPUT_DIR`` collide with unrelated tooling,
    so the ``KBCRAFT_``-prefixed form takes precedence. Bare *aliases* (e.g.
    ``"MODEL"``) remain supported for backward compatibility.
    """
    for key in (f"KBCRAFT_{name}", *aliases):
        val = os.environ.get(key, "")
        if val:
            return val
    return default


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
class OpenAIBackendConfig:
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
    openai: OpenAIBackendConfig
    models: Dict[str, ModelConfig]
    tokenizer: TokenizerConfig

    @property
    def model(self) -> ModelConfig:
        """Return the active :class:`ModelConfig`."""
        return self.models[self.active_model]

    #: Model backend name → the CLI ``--embedder`` value it maps to.
    _EMBEDDER_FOR_BACKEND = {
        "ollama": "ollama",
        "openai": "openai",
        "openai_compatible": "openai",
    }

    @property
    def embedder(self) -> str:
        """CLI ``--embedder`` value for the active model's backend."""
        return self._EMBEDDER_FOR_BACKEND.get(self.model.backend, self.model.backend)

    @property
    def base_url(self) -> str:
        """Base URL for the active model's backend (empty for the real OpenAI API)."""
        backend = self.model.backend
        if backend == "openai":
            return self.openai.base_url
        if backend == "openai_compatible":
            return self.openai_compatible.base_url
        if backend == "ollama":
            return self.ollama.host
        return ""


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
    multipart_threshold: int = 104857600  # 100 MB
    multipart_chunksize: int = 26214400  # 25 MB
    max_concurrency: int = 10
    max_bandwidth: int = None
    use_threads: bool = True


@dataclass
class S3StorageConfig:
    region: str
    access_key_id: str
    secret_access_key: str
    session_token: str
    profile: str
    endpoint_url: str
    bucket: str
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

        openai_raw = raw.get("backends", {}).get("openai", {})
        openai = OpenAIBackendConfig(
            base_url=_env("OPENAI_BASE_URL", openai_raw.get("base_url", "")),
            token=_env("OPENAI_API_KEY", openai_raw.get("token", "")),
        )

        tok_raw = raw.get("tokenizer", {})
        tokenizer = TokenizerConfig(
            prefer_ollama=tok_raw.get("prefer_ollama", False),
            hf_fallback_repo=tok_raw.get("hf_fallback_repo", "bert-base-uncased"),
        )

        models = {name: self._parse_model(name, cfg) for name, cfg in raw.get("models", {}).items()}

        active_model = _env_ns("MODEL", "MODEL", default=raw.get("active_model"))
        if active_model not in models:
            raise KeyError(
                f"Model {active_model!r} not found in embedding.yaml. " f"Available: {list(models)}"
            )

        # Apply chunking env var overrides to the active model only
        active = models[active_model]
        max_tokens_override = _env_ns("MAX_TOKENS", "MAX_TOKENS")
        if max_tokens_override:
            active.chunking.max_tokens = int(max_tokens_override)
        overlap_override = _env_ns("CHUNK_OVERLAP", "CHUNK_OVERLAP")
        if overlap_override:
            active.chunking.overlap = int(overlap_override)

        # Keep the Chunker invariant (0 <= overlap < max_tokens). An independent
        # max_tokens override can otherwise leave overlap >= max_tokens, which
        # makes Chunker.__init__ raise. Clamp to a sane fraction rather than crash.
        if active.chunking.overlap >= active.chunking.max_tokens:
            active.chunking.overlap = max(0, active.chunking.max_tokens // 8)

        return EmbeddingConfig(
            active_model=active_model,
            ollama=ollama,
            openai_compatible=openai_compatible,
            openai=openai,
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
            region=_env("S3_REGION", s3_raw.get("region", "us-east-1")),
            access_key_id=_env("S3_ACCESS_KEY_ID", s3_raw.get("access_key_id", "")),
            secret_access_key=_env("S3_SECRET_ACCESS_KEY", s3_raw.get("secret_access_key", "")),
            session_token=_env("S3_SESSION_TOKEN", s3_raw.get("session_token", "")),
            profile=_env("S3_PROFILE", s3_raw.get("profile", "")),
            endpoint_url=_env("S3_ENDPOINT_URL", s3_raw.get("endpoint_url", "")),
            bucket=_env("S3_BUCKET", s3_raw.get("bucket", "")),
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
        output_dir = Path(
            _env_ns("OUTPUT_DIR", "OUTPUT_DIR", default=faiss_raw.get("output_dir", "faiss_index"))
        )
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


# ── Shell-env emitter ────────────────────────────────────────────────────────
#
# Resolve the configs and print ``KBCRAFT_*`` shell assignments so bash scripts
# can source every pipeline parameter from the yaml files instead of hardcoding
# them. Consumed by scripts/test_cli_index_openai.sh via:
#
#     eval "$(python -m kbcraft.config env --configs-dir configs)"


def resolve_params(configs_dir: Path) -> Dict[str, str]:
    """Resolve embedding + vector store configs into a flat KEY→str map.

    Every value already reflects the env → yaml → default resolution done by
    :class:`ConfigFactory`, so callers see the same params the pipeline will use.
    """
    factory = ConfigFactory(configs_dir)
    emb = factory.load_embedding()
    vs = factory.load_vector_store()

    model = emb.model
    output_dir = vs.faiss.output_dir

    # S3 params are intentionally NOT emitted here — they live solely as the
    # ``S3_*`` env vars in .env (read directly by scripts and by
    # ConfigFactory.load_storage / the boto3 exporter). Re-emitting them under
    # ``KBCRAFT_*`` names would just duplicate that single source of truth.
    return {
        "KBCRAFT_MODEL": model.name,
        "KBCRAFT_BACKEND": model.backend,
        "KBCRAFT_EMBEDDER": emb.embedder,
        "KBCRAFT_BASE_URL": emb.base_url,
        "KBCRAFT_EMBEDDING_DIM": str(model.embedding_dim),
        "KBCRAFT_MAX_TOKENS": str(model.chunking.max_tokens),
        "KBCRAFT_CHUNK_OVERLAP": str(model.chunking.overlap),
        "KBCRAFT_VECTOR_BACKEND": vs.active_backend,
        "KBCRAFT_OUTPUT_DIR": str(output_dir),
    }


def _emit_env(configs_dir: Path) -> int:
    import shlex

    params = resolve_params(configs_dir)
    for key, value in params.items():
        print(f"{key}={shlex.quote(value)}")
    return 0


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m kbcraft.config",
        description="Resolve kbcraft yaml configs and emit them for other tools.",
    )
    sub = parser.add_subparsers(dest="command")

    env = sub.add_parser(
        "env",
        help="Print resolved params as KBCRAFT_*=value lines for `eval` in shell.",
    )
    env.add_argument(
        "--configs-dir",
        metavar="DIR",
        default="configs",
        help="Directory holding embedding.yaml / vector_store.yaml. Default: ./configs",
    )

    args = parser.parse_args(argv)
    if args.command == "env":
        return _emit_env(Path(args.configs_dir))
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
