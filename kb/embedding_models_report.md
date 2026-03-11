# Embedding Models Report: English + Russian + Code
### A Technical Reference for Building a Python Wrapper for Vector Databases

> **Scope:** This report focuses on models best suited for **English**, **Russian**, and **source code** retrieval.
> Primary use case: semantic search over mixed-language codebases using vector databases.

---

## Table of Contents
1. [TL;DR — Recommended Stack](#tldr--recommended-stack)
2. [Model Comparison Matrix](#model-comparison-matrix)
3. [Model Deep Dives](#model-deep-dives)
   - [BGE-M3](#1-bge-m3--primary-recommendation)
   - [GTE-Qwen2-7B](#2-gte-qwen2-7b-instruct--best-quality-on-premise)
   - [multilingual-E5-large](#3-multilingual-e5-large--simple-multilingual-baseline)
   - [OpenAI text-embedding-3](#4-openai-text-embedding-3-small--large)
   - [Cohere Embed v3](#5-cohere-embed-v3)
   - [Mistral Embed](#6-mistral-embed)
   - [sentence-transformers / SBERT](#7-sentence-transformers--sbert-english-only)
   - [Nomic Embed via Ollama](#8-nomic-embed-via-ollama--lightweight-local)
   - [Jina Embeddings v3](#9-jina-embeddings-v3)
4. [Russian Language Quality Notes](#russian-language-quality-notes)
5. [Code Retrieval Quality Notes](#code-retrieval-quality-notes)
6. [Vector Database Compatibility](#vector-database-compatibility)
7. [Python Wrapper Architecture](#python-wrapper-architecture)
8. [Wrapper Implementation Guide](#wrapper-implementation-guide)
9. [Model Selection Guide](#model-selection-guide)

---

## TL;DR — Recommended Stack

| Role | Model | License | On-Premise | EN | RU | Code |
|---|---|---|---|---|---|---|
| **Primary** | `BAAI/bge-m3` | MIT | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Best quality** | `Alibaba-NLP/gte-qwen2-7b-instruct` | Apache 2.0 | ✅ (GPU) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Cloud fallback** | `text-embedding-3-large` (OpenAI) | Proprietary | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Lightweight local** | `nomic-embed-text` via Ollama | Apache 2.0 | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Simple multilingual** | `intfloat/multilingual-e5-large` | MIT | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**Practical recommendation:**
- Use **BGE-M3** as your default on-premise model — best balance of quality, context length, multilingual support, and open license
- Use **GTE-Qwen2-7B** when you have GPU budget and need maximum quality
- Use **OpenAI text-embedding-3-large** when cost is acceptable and you want zero infrastructure
- Use **Ollama + nomic-embed-text** for fast local prototyping with no Python ML dependencies

---

## Model Comparison Matrix

| Model | Provider | Dimensions | Max Tokens | License | On-Premise | Russian | Code | MTEB |
|---|---|---|---|---|---|---|---|---|
| **bge-m3** | BAAI | 1024 | **8192** | MIT | ✅ | ✅✅ | ✅✅ | ~63 |
| **gte-qwen2-7b-instruct** | Alibaba | 3584 | **32768** | Apache 2.0 | ✅ (GPU) | ✅✅✅ | ✅✅✅ | ~65 |
| **multilingual-e5-large** | Microsoft | 1024 | 512 | MIT | ✅ | ✅✅ | ✅ | ~61 |
| text-embedding-3-small | OpenAI | 1536 | 8191 | Proprietary | ❌ | ✅✅ | ✅✅ | ~62 |
| text-embedding-3-large | OpenAI | 3072 | 8191 | Proprietary | ❌ | ✅✅ | ✅✅ | ~64 |
| embed-multilingual-v3.0 | Cohere | 1024 | 512 | Proprietary | ❌ | ✅✅ | ✅ | ~62 |
| mistral-embed | Mistral AI | 1024 | 8192 | Proprietary | ❌ | ✅✅ | ✅✅ | ~60 |
| all-mpnet-base-v2 | SBERT | 768 | 384 | Apache 2.0 | ✅ | ❌ | ✅ | ~57 |
| nomic-embed-text-v1.5 | Nomic AI | 768 | 8192 | Apache 2.0 | ✅ | ✅ | ✅✅ | ~62 |
| jina-embeddings-v3 | Jina AI | 1024 | 8192 | CC BY-NC 4.0 | ✅ (non-com) | ✅✅ | ✅✅ | ~63 |
| gte-large-en-v1.5 | Alibaba | 1024 | 8192 | Apache 2.0 | ✅ | ❌ | ✅✅ | ~63 |
| bge-large-en-v1.5 | BAAI | 1024 | 512 | MIT | ✅ | ❌ | ✅✅ | ~64 |

> ✅ = supported, ✅✅ = good quality, ✅✅✅ = excellent quality, ❌ = not supported/poor

---

## Model Deep Dives

---

### 1. BGE-M3 — Primary Recommendation

**Type:** Open-source, on-premise
**Provider:** Beijing Academy of AI (BAAI)
**License:** MIT ✅
**HuggingFace:** https://huggingface.co/BAAI/bge-m3

#### Why BGE-M3 for EN + RU + Code
- Trained on **100+ languages** including strong Russian coverage
- **8192 token context** — fits most source files without chunking
- **Multi-functionality** in a single model:
  - **Dense retrieval** — standard semantic similarity vectors
  - **Sparse retrieval** — lexical/keyword matching (BM25-like), critical for code where exact token matches matter (function names, variable names, error codes)
  - **ColBERT-style** multi-vector — late interaction for highest precision
- MIT license — safe for commercial use
- No instruction prefix required (unlike BGE v1.5)

#### Key Properties
| Property | Value |
|---|---|
| Dimensions | 1024 (dense) |
| Max tokens | 8192 |
| Languages | 100+ |
| License | MIT |
| VRAM fp16 | ~2.5 GB |
| VRAM fp32 | ~5 GB |
| CPU usable | ✅ (slow but works) |

#### Installation
```bash
pip install FlagEmbedding
# or for sentence-transformers interface:
pip install sentence-transformers
```

#### Python Usage — FlagEmbedding (full features)
```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True,   # faster on GPU, minimal quality loss
    device="cuda"    # or "cpu"
)

# Dense only (most vector DBs)
output = model.encode(
    ["def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"],
    batch_size=12,
    max_length=8192,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False
)
dense_vectors = output["dense_vecs"]  # shape: (N, 1024)

# Hybrid: dense + sparse (best for code search)
output = model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
)
dense_vecs  = output["dense_vecs"]       # for semantic similarity
sparse_vecs = output["lexical_weights"]  # dict of token_id -> weight
```

#### Python Usage — sentence-transformers (simple interface)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

# No prefix needed for BGE-M3
embeddings = model.encode(
    ["Hello world", "Привет мир", "def hello(): pass"],
    normalize_embeddings=True,
    batch_size=32
)
# embeddings.shape: (3, 1024)
```

#### Important: No Instruction Prefix for BGE-M3
Unlike BGE v1.5, **BGE-M3 does not require instruction prefixes**. Encode queries and documents the same way.

---

### 2. GTE-Qwen2-7B-Instruct — Best Quality On-Premise

**Type:** Open-source, on-premise (GPU required)
**Provider:** Alibaba DAMO
**License:** Apache 2.0 ✅
**HuggingFace:** https://huggingface.co/Alibaba-NLP/gte-qwen2-7b-instruct

#### Why GTE-Qwen2-7B for EN + RU + Code
- Based on **Qwen2-7B**, a decoder-only LLM trained on massive multilingual + code corpus
- **Top MTEB scores** across multilingual benchmarks
- **32768 token context** — can embed entire source files, READMEs, or large docstrings
- Qwen2 base model has exceptional Russian and Chinese coverage
- Excellent code understanding due to code-heavy pretraining

#### Key Properties
| Property | Value |
|---|---|
| Dimensions | 3584 |
| Max tokens | 32768 |
| Languages | 100+ (especially strong Ru/Zh) |
| License | Apache 2.0 |
| VRAM fp16 | ~16 GB |
| VRAM int4 | ~5 GB (with quantization) |
| CPU usable | ❌ (impractically slow) |

#### Installation
```bash
pip install sentence-transformers transformers torch
```

#### Python Usage
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Alibaba-NLP/gte-qwen2-7b-instruct",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": "float16"}
)

embeddings = model.encode(
    ["def merge_sort(arr): ...", "Сортировка слиянием"],
    batch_size=4,            # small batch due to model size
    normalize_embeddings=True
)
# embeddings.shape: (2, 3584)
```

#### Quantized / Smaller Alternative
```python
# 1.5B version — fits on 4GB VRAM, strong quality/size tradeoff
model = SentenceTransformer(
    "Alibaba-NLP/gte-qwen2-1.5b-instruct",
    trust_remote_code=True
)
# Dimensions: 1536, Context: 32768
```

#### Resource Warning
The 7B model requires a **16GB VRAM GPU** (e.g., RTX 3090, A100). For CPU-only environments, use BGE-M3 instead. The 1.5B variant runs on ~4GB VRAM.

---

### 3. multilingual-E5-large — Simple Multilingual Baseline

**Type:** Open-source, on-premise
**Provider:** Microsoft
**License:** MIT ✅
**HuggingFace:** https://huggingface.co/intfloat/multilingual-e5-large

#### Why multilingual-E5-large
- Simple and reliable — no special library needed, pure sentence-transformers
- Good Russian quality (trained on mC4, CC100, multilingual datasets)
- 1024 dimensions — good density for vector DBs
- Easy to run on CPU for smaller workloads

#### Key Properties
| Property | Value |
|---|---|
| Dimensions | 1024 |
| Max tokens | 512 |
| Languages | 100+ |
| License | MIT |
| VRAM fp16 | ~1.1 GB |
| CPU usable | ✅ (reasonable speed) |

#### Critical: Instruction Prefix Required
E5 models **require task prefixes** — the wrapper must handle this internally.

```python
# Queries — always prefix with "query: "
query = "query: как работает алгоритм слияния"

# Documents / code — always prefix with "passage: "
doc = "passage: def merge(left, right): ..."
```

#### Python Usage
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")

# The wrapper injects prefixes transparently
query_embedding = model.encode(["query: merge sort algorithm"])
doc_embeddings  = model.encode(["passage: def merge_sort(arr): ..."])
```

#### Limitation
**512 token max** is a significant constraint for code files. Use BGE-M3 or GTE-Qwen2 when files are longer. The wrapper should warn or auto-truncate.

---

### 4. OpenAI `text-embedding-3-small` & `text-embedding-3-large`

**Type:** Cloud API only
**Provider:** OpenAI
**License:** Proprietary

#### Why OpenAI for EN + RU + Code
- Trained on massive multilingual corpus with good Russian coverage
- Excellent general-purpose quality across languages and domains including code
- `text-embedding-3-large` supports **Matryoshka dimension reduction** — store at 1024 dims to save space
- Zero infrastructure — ideal for prototyping or cloud-first deployments

#### Key Properties
| Model | Dimensions | Max Tokens | Price / 1M tokens |
|---|---|---|---|
| text-embedding-3-small | 1536 (scalable) | 8191 | $0.020 |
| text-embedding-3-large | 3072 (scalable) | 8191 | $0.130 |

#### Python Usage
```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")

response = client.embeddings.create(
    input=["def fibonacci(n): ...", "функция Фибоначчи"],
    model="text-embedding-3-large",
    dimensions=1024  # optional: reduce from 3072 to save storage
)

vectors = [item.embedding for item in response.data]
```

#### Batching
```python
# Max 2048 inputs per request — chunk large lists:
def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

all_vectors = []
for batch in chunk(texts, 500):
    response = client.embeddings.create(input=batch, model="text-embedding-3-large")
    all_vectors.extend([item.embedding for item in response.data])
```

---

### 5. Cohere Embed v3

**Type:** Cloud API only
**Provider:** Cohere
**License:** Proprietary

#### Key Properties
- 1024 dimensions
- `embed-multilingual-v3.0` — good Russian support
- **Requires `input_type`** parameter: `search_document`, `search_query`, `classification`, `clustering`
- Native int8 and binary quantization
- **512 token limit** — significant constraint for code

#### Python Usage
```python
import cohere

co = cohere.Client("your-api-key")

# Documents
doc_response = co.embed(
    texts=["def merge_sort(arr): ..."],
    model="embed-multilingual-v3.0",
    input_type="search_document"
)

# Queries
query_response = co.embed(
    texts=["алгоритм сортировки"],
    model="embed-multilingual-v3.0",
    input_type="search_query"
)
```

#### Wrapper Note
The `input_type` must be set correctly — `encode_query()` and `encode_documents()` must pass different values automatically.

---

### 6. Mistral Embed

**Type:** Cloud API only
**Provider:** Mistral AI
**License:** Proprietary

#### Key Properties
- 1024 dimensions
- 8192 token context — good for code files
- Strong multilingual capability (European languages including Russian)
- Simpler API than Cohere — no input_type needed

#### Python Usage
```python
from mistralai import Mistral

client = Mistral(api_key="your-api-key")

response = client.embeddings.create(
    model="mistral-embed",
    inputs=["def hello(): pass", "функция приветствия"]
)
vectors = [item.embedding for item in response.data]
```

---

### 7. sentence-transformers / SBERT (English only)

**Type:** Open-source, on-premise
**Provider:** UKPLab / HuggingFace
**License:** Apache 2.0 ✅

#### English-Only Models (no Russian support)
| Model | Dimensions | Context | Notes |
|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 256 | Fastest, lightest |
| all-MiniLM-L12-v2 | 384 | 256 | Slightly better |
| all-mpnet-base-v2 | 768 | 384 | Best English SBERT |

> **Note:** These models have limited or no Russian support. For EN+RU, prefer BGE-M3 or multilingual-E5.

#### Python Usage
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(
    ["Hello world", "def foo(): pass"],
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True
)
```

#### When to Use SBERT
- English-only codebases
- CPU-only servers with limited RAM
- Prototyping / low-cost scenarios
- **Not recommended for mixed EN+RU projects**

---

### 8. Nomic Embed via Ollama — Lightweight Local

**Type:** Open-source, on-premise (via Ollama runtime)
**License:** Apache 2.0 ✅
**Website:** https://ollama.com

#### Why Ollama
- **Zero Python ML dependencies** — runs as a local REST server
- Simple REST API, no CUDA setup
- Can serve multiple embedding models side by side
- `bge-m3` available via Ollama — full EN+RU+Code support without FlagEmbedding complexity

#### Available Models via Ollama
| Ollama Model Name | Base | Dimensions | Context | RU | Code |
|---|---|---|---|---|---|
| `nomic-embed-text` | Nomic v1.5 | 768 | 8192 | ✅ | ✅✅ |
| `mxbai-embed-large` | MixedBread | 1024 | 512 | ✅ | ✅ |
| `bge-m3` | BAAI BGE-M3 | 1024 | 8192 | ✅✅ | ✅✅ |
| `snowflake-arctic-embed` | Snowflake | 1024 | 512 | ✅ | ✅✅ |

#### Setup
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull nomic-embed-text
ollama pull bge-m3
```

#### Python Usage
```python
import ollama

# Sync
response = ollama.embed(
    model="bge-m3",
    input=["def fibonacci(n): ...", "алгоритм Фибоначчи"]
)
vectors = response["embeddings"]  # list[list[float]]

# Async via httpx
import httpx

async def embed_async(texts: list[str], model: str = "bge-m3") -> list[list[float]]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11434/api/embed",
            json={"model": model, "input": texts},
            timeout=60.0
        )
    return resp.json()["embeddings"]
```

#### Nomic Instruction Prefixes (when used directly, not via Ollama)
```python
query   = "search_query: "    + user_query
passage = "search_document: " + document_text
code    = "search_document: " + source_code  # treat code as a document
```

---

### 9. Jina Embeddings v3

**Type:** Open-source (non-commercial) / Cloud API
**Provider:** Jina AI
**License:** CC BY-NC 4.0 (non-commercial), commercial license available
**HuggingFace:** https://huggingface.co/jinaai/jina-embeddings-v3

#### Key Properties for EN + RU + Code
- 1024 dimensions, **8192 token context**
- 89 languages including Russian
- Task-specific LoRA adapters: `retrieval.query`, `retrieval.passage`, `text-matching`, `classification`, `separation`
- Matryoshka dimension reduction

#### Python Usage
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

query_vecs = model.encode(
    ["поиск функции сортировки"],
    task="retrieval.query"
)
doc_vecs = model.encode(
    ["def sort(arr): return sorted(arr)"],
    task="retrieval.passage"
)
```

#### License Warning
**CC BY-NC 4.0** — free for non-commercial / research use only. Obtain a commercial license from Jina AI for production deployments.

---

## Russian Language Quality Notes

Models ranked by Russian quality:

1. **GTE-Qwen2-7B** — Qwen2 base trained on large Russian web corpus (CommonCrawl RU); best quality
2. **BGE-M3** — explicitly trained on multilingual retrieval including Russian; strong performance
3. **multilingual-E5-large** — trained on mC4 + MIRACL which includes Russian; reliable
4. **text-embedding-3-large** — OpenAI's internal multilingual training; excellent Cyrillic handling
5. **embed-multilingual-v3.0** — Cohere 100+ languages; Russian well-covered
6. **Jina v3** — 89 languages, Russian included, good quality
7. **nomic-embed-text** — decent Russian via multilingual pretraining, not optimized
8. **all-mpnet-base-v2** — English only, avoid for Russian

### Russian-Specific Test Pairs for Validation
```python
RU_TEST_PAIRS = [
    ("функция сортировки", "def sort(arr): return sorted(arr)"),
    ("обработка ошибок", "try:\n    ...\nexcept Exception as e:\n    ..."),
    ("подключение к базе данных", "db = connect('postgresql://...')"),
    ("рекурсивный алгоритм", "def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)"),
]
# Cosine similarity should be > 0.5 for related pairs
```

---

## Code Retrieval Quality Notes

### Why Code Retrieval Is Different
- **Exact token matching matters** — function names, class names, error codes should match lexically
- **Long context is essential** — functions can be hundreds of lines
- **Mixed-language content** — docstrings in Russian/English inside Python/JS/Go files
- **Semantic gap** — natural language queries ("sort a list") should match code (`def merge_sort`)

### Model Capabilities for Code

| Model | Long Context | Exact Match | NL→Code | Code→Code |
|---|---|---|---|---|
| bge-m3 (dense+sparse) | ✅ 8K | ✅✅ sparse | ✅✅ | ✅✅ |
| gte-qwen2-7b | ✅ 32K | ✅ | ✅✅✅ | ✅✅✅ |
| text-embedding-3-large | ✅ 8K | ✅ | ✅✅ | ✅✅ |
| multilingual-e5-large | ❌ 512 | ✅ | ✅✅ | ✅ |
| nomic-embed-text | ✅ 8K | ✅ | ✅✅ | ✅✅ |

### Chunking Strategy for Code (wrapper should implement)
```python
# Recommended chunking for code files:
# - Chunk at function/class boundaries, not fixed character counts
# - Include file path and language in chunk metadata
# - For very long functions: sliding window with 20% overlap

CHUNK_METADATA_TEMPLATE = """
# File: {file_path}
# Language: {language}
# Scope: {scope}  # e.g., "class MyClass > method fit"

{code_chunk}
"""
```

### Hybrid Search for Code
For codebases, use **hybrid search** combining dense + sparse vectors:
- Dense vectors: semantic similarity ("sort algorithm" matches `merge_sort`)
- Sparse vectors: exact token match (`merge_sort` matches `merge_sort`)
- BGE-M3 provides both in one model via FlagEmbedding

---

## Vector Database Compatibility

| Vector DB | Python Client | Hybrid Search | Notes |
|---|---|---|---|
| **Qdrant** | `qdrant-client` | ✅ (named vectors) | Best BGE-M3 hybrid support |
| **Weaviate** | `weaviate-client` | ✅ (BM25 + dense) | Good multilingual modules |
| **Chroma** | `chromadb` | ❌ (dense only) | Simple, good for prototyping |
| **Pinecone** | `pinecone` | ✅ (sparse+dense) | Cloud only |
| **Milvus** | `pymilvus` | ✅ | Self-hosted, scalable |
| **pgvector** | `psycopg2`/`asyncpg` | ❌ (dense only) | SQL-native, simple |
| **LanceDB** | `lancedb` | ✅ | Arrow-native, embedded |
| **Faiss** | `faiss-cpu` | ❌ | Library only, no sparse |

**Recommended DB for this stack:** **Qdrant** — best native support for BGE-M3's hybrid dense+sparse output, named vectors, and excellent Python client.

---

## Python Wrapper Architecture

### Design Principles
1. **Unified interface** — one `encode()` method regardless of model
2. **Transparent prefixes** — `encode_query()` / `encode_documents()` inject prefixes internally
3. **Async support** — `encode_async()` for cloud APIs and Ollama
4. **Batching** — chunk large inputs automatically per model limits
5. **Normalization** — always return `list[list[float]]` (float32)
6. **Dimension awareness** — expose `embedding_dim` property
7. **Lazy loading** — local models loaded only on first call
8. **Hybrid output** — BGE-M3 returns both dense + sparse when requested

### Class Hierarchy
```
BaseEmbedder (ABC)
├── encode(texts: list[str]) -> list[list[float]]
├── encode_query(text: str) -> list[float]
├── encode_documents(texts: list[str]) -> list[list[float]]
├── encode_hybrid(texts) -> HybridOutput   # dense + sparse, BGE-M3 only
├── embedding_dim: int
└── model_name: str

Implementations:
├── BGEM3Embedder           <- PRIMARY — dense + sparse, EN+RU+Code, MIT
├── GTEQwen2Embedder        <- BEST QUALITY — EN+RU+Code, GPU required
├── MultilingualE5Embedder  <- SIMPLE — EN+RU, handles query:/passage: prefixes
├── OpenAIEmbedder          <- CLOUD — dimension reduction, batching
├── CohereEmbedder          <- CLOUD — input_type handling
├── MistralEmbedder         <- CLOUD — simple API
├── SBERTEmbedder           <- LOCAL — English only, generic sentence-transformers
├── OllamaEmbedder          <- LOCAL SERVER — REST API, async support
└── JinaEmbedder            <- LOCAL/CLOUD — task-based LoRA adapters

Factory:
└── get_embedder(model_id: str, **kwargs) -> BaseEmbedder
```

### HybridOutput Type
```python
from dataclasses import dataclass

@dataclass
class HybridOutput:
    dense: list[list[float]]          # shape: (N, 1024) — for ANN search
    sparse: list[dict[int, float]]    # token_id -> weight — for BM25-like search
    texts: list[str]                  # original inputs
```

---

## Wrapper Implementation Guide

### 1. Dependencies

```toml
# requirements.txt

# Core ML
torch>=2.0.0
transformers>=4.40.0
sentence-transformers>=3.0.0

# BGE-M3 (primary model)
FlagEmbedding>=1.2.0

# Cloud APIs
openai>=1.0.0
cohere>=5.0.0
mistralai>=1.0.0

# Ollama local server
ollama>=0.2.0
httpx>=0.27.0          # async REST calls

# Utilities
numpy>=1.24.0
tenacity>=8.0.0        # retry logic for API calls
```

### 2. Base Embedder

```python
# embedders/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class HybridOutput:
    dense: list[list[float]]
    sparse: list[dict[int, float]]
    texts: list[str]


class BaseEmbedder(ABC):
    """
    Unified interface for all embedding models.
    All implementations return normalized float32 vectors.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output vector dimensionality."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @property
    def max_tokens(self) -> int:
        """Maximum input tokens. Override in subclass."""
        return 512

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode texts for symmetric tasks (clustering, classification).
        No query/document distinction.
        """
        ...

    def encode_query(self, text: str) -> list[float]:
        """
        Encode a single search query.
        Override to inject query-specific prefixes or task parameters.
        """
        return self.encode([text])[0]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Encode documents for indexing.
        Override to inject document-specific prefixes or task parameters.
        """
        return self.encode(texts)

    def encode_hybrid(self, texts: list[str]) -> HybridOutput:
        """
        Return both dense and sparse vectors.
        Only implemented by BGEM3Embedder. Others raise NotImplementedError.
        """
        raise NotImplementedError(f"{self.model_name} does not support hybrid encoding.")

    def _to_list(self, arr) -> list[list[float]]:
        """Normalize numpy arrays / tensors to list[list[float]]."""
        if isinstance(arr, np.ndarray):
            return arr.astype(np.float32).tolist()
        return [[float(v) for v in row] for row in arr]

    def _chunk(self, lst: list, size: int):
        """Split list into chunks of given size."""
        for i in range(0, len(lst), size):
            yield lst[i : i + size]
```

### 3. BGE-M3 Embedder (Primary)

```python
# embedders/bge_m3.py
from FlagEmbedding import BGEM3FlagModel
from .base import BaseEmbedder, HybridOutput


class BGEM3Embedder(BaseEmbedder):
    """
    BGE-M3: multilingual, long-context, dense + sparse hybrid.
    Best on-premise model for EN + RU + Code.
    No instruction prefixes needed.
    """

    def __init__(
        self,
        device: str = "cuda",
        use_fp16: bool = True,
        batch_size: int = 12,
    ):
        self._model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=use_fp16,
            device=device,
        )
        self._batch_size = batch_size

    @property
    def embedding_dim(self) -> int:
        return 1024

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def model_name(self) -> str:
        return "BAAI/bge-m3"

    def encode(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self.max_tokens,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return self._to_list(output["dense_vecs"])

    # encode_query and encode_documents fall back to encode() — no prefixes needed

    def encode_hybrid(self, texts: list[str]) -> HybridOutput:
        """Return dense + sparse vectors for hybrid search in Qdrant."""
        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self.max_tokens,
            return_dense=True,
            return_sparse=True,
        )
        return HybridOutput(
            dense=self._to_list(output["dense_vecs"]),
            sparse=output["lexical_weights"],   # list of {token_id: weight}
            texts=texts,
        )
```

### 4. multilingual-E5 Embedder

```python
# embedders/e5.py
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder


class MultilingualE5Embedder(BaseEmbedder):
    """
    multilingual-e5-large: EN + RU + 100 languages.
    Requires "query: " and "passage: " prefixes — handled internally.
    """

    QUERY_PREFIX    = "query: "
    DOCUMENT_PREFIX = "passage: "

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self._model = SentenceTransformer(
            "intfloat/multilingual-e5-large",
            device=device
        )
        self._batch_size = batch_size

    @property
    def embedding_dim(self) -> int:
        return 1024

    @property
    def max_tokens(self) -> int:
        return 512  # hard limit — warn user for long code files

    @property
    def model_name(self) -> str:
        return "intfloat/multilingual-e5-large"

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Symmetric encoding — no prefix (for clustering/classification)."""
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True
        )
        return self._to_list(vecs)

    def encode_query(self, text: str) -> list[float]:
        vecs = self._model.encode(
            [self.QUERY_PREFIX + text],
            normalize_embeddings=True
        )
        return self._to_list(vecs)[0]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.DOCUMENT_PREFIX + t for t in texts]
        vecs = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=True
        )
        return self._to_list(vecs)
```

### 5. OpenAI Embedder

```python
# embedders/openai_embedder.py
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI text-embedding-3-small / text-embedding-3-large.
    Supports Matryoshka dimension reduction via `dimensions` param.
    """

    DIMENSIONS_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    BATCH_LIMIT = 500  # conservative (API max: 2048)

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int | None = None,
    ):
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions   # None = use model default

    @property
    def embedding_dim(self) -> int:
        if self._dimensions:
            return self._dimensions
        return self.DIMENSIONS_MAP.get(self._model, 1536)

    @property
    def max_tokens(self) -> int:
        return 8191

    @property
    def model_name(self) -> str:
        return self._model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def encode(self, texts: list[str]) -> list[list[float]]:
        all_vectors = []
        for batch in self._chunk(texts, self.BATCH_LIMIT):
            kwargs = {"input": batch, "model": self._model}
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            response = self._client.embeddings.create(**kwargs)
            all_vectors.extend([item.embedding for item in response.data])
        return all_vectors
```

### 6. Cohere Embedder

```python
# embedders/cohere_embedder.py
import cohere
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    """
    Cohere embed-multilingual-v3.0.
    input_type is injected automatically per encode_query / encode_documents.
    """

    BATCH_LIMIT = 96  # Cohere API limit per request

    def __init__(self, api_key: str, model: str = "embed-multilingual-v3.0"):
        self._client = cohere.Client(api_key)
        self._model = model

    @property
    def embedding_dim(self) -> int:
        return 1024

    @property
    def max_tokens(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return self._model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        all_vectors = []
        for batch in self._chunk(texts, self.BATCH_LIMIT):
            response = self._client.embed(
                texts=batch,
                model=self._model,
                input_type=input_type
            )
            all_vectors.extend(response.embeddings)
        return all_vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type="clustering")

    def encode_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="search_query")[0]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type="search_document")
```

### 7. Ollama Embedder

```python
# embedders/ollama_embedder.py
import httpx
from .base import BaseEmbedder

OLLAMA_DIM_MAP = {
    "nomic-embed-text":        768,
    "mxbai-embed-large":       1024,
    "bge-m3":                  1024,
    "snowflake-arctic-embed":  1024,
    "all-minilm":              384,
}


class OllamaEmbedder(BaseEmbedder):
    """
    Wraps a locally running Ollama server.
    Supports any model pulled via `ollama pull <model>`.
    Zero Python ML dependencies — all inference via REST API.
    """

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    @property
    def embedding_dim(self) -> int:
        return OLLAMA_DIM_MAP.get(self._model, 768)

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    def encode(self, texts: list[str]) -> list[list[float]]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts}
            )
            resp.raise_for_status()
        return resp.json()["embeddings"]

    async def encode_async(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts}
            )
            resp.raise_for_status()
        return resp.json()["embeddings"]
```

### 8. Jina Embedder

```python
# embedders/jina_embedder.py
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder


class JinaEmbedder(BaseEmbedder):
    """
    Jina Embeddings v3 with task-specific LoRA adapters.
    License: CC BY-NC 4.0 — non-commercial only without paid license.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self._model = SentenceTransformer(
            "jinaai/jina-embeddings-v3",
            trust_remote_code=True,
            device=device
        )
        self._batch_size = batch_size

    @property
    def embedding_dim(self) -> int:
        return 1024

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def model_name(self) -> str:
        return "jinaai/jina-embeddings-v3"

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, task="text-matching", batch_size=self._batch_size)
        return self._to_list(vecs)

    def encode_query(self, text: str) -> list[float]:
        vecs = self._model.encode([text], task="retrieval.query")
        return self._to_list(vecs)[0]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, task="retrieval.passage", batch_size=self._batch_size)
        return self._to_list(vecs)
```

### 9. GTE-Qwen2 Embedder

```python
# embedders/gte_qwen2.py
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder

MODEL_MAP = {
    "7b":  ("Alibaba-NLP/gte-qwen2-7b-instruct",  3584),
    "1.5b": ("Alibaba-NLP/gte-qwen2-1.5b-instruct", 1536),
}


class GTEQwen2Embedder(BaseEmbedder):
    """
    GTE-Qwen2: highest quality EN+RU+Code embeddings.
    Requires GPU. No instruction prefix needed.
    """

    def __init__(
        self,
        size: str = "7b",     # "7b" or "1.5b"
        device: str = "cuda",
        batch_size: int = 4,
    ):
        hf_name, self._dim = MODEL_MAP[size]
        self._model = SentenceTransformer(
            hf_name,
            trust_remote_code=True,
            device=device,
            model_kwargs={"torch_dtype": "float16"}
        )
        self._batch_size = batch_size
        self._hf_name = hf_name

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def max_tokens(self) -> int:
        return 32768

    @property
    def model_name(self) -> str:
        return self._hf_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True
        )
        return self._to_list(vecs)
```

### 10. Factory Function

```python
# embedders/factory.py
from .base import BaseEmbedder
from .bge_m3 import BGEM3Embedder
from .e5 import MultilingualE5Embedder
from .openai_embedder import OpenAIEmbedder
from .ollama_embedder import OllamaEmbedder


def get_embedder(model_id: str, **kwargs) -> BaseEmbedder:
    """
    Factory — returns the correct embedder for the given model_id.
    Format: "provider:model_name" or just "provider" for defaults.

    Examples:
        get_embedder("bge-m3")
        get_embedder("bge-m3", device="cpu")
        get_embedder("e5")
        get_embedder("openai:text-embedding-3-large", api_key="sk-...", dimensions=1024)
        get_embedder("openai:text-embedding-3-small", api_key="sk-...")
        get_embedder("ollama:bge-m3")
        get_embedder("ollama:nomic-embed-text")
        get_embedder("cohere:embed-multilingual-v3.0", api_key="...")
        get_embedder("mistral", api_key="...")
        get_embedder("sbert:all-mpnet-base-v2")
        get_embedder("gte-qwen2")
        get_embedder("gte-qwen2", size="1.5b")
        get_embedder("jina")
    """
    provider, _, model = model_id.partition(":")

    match provider:
        case "bge-m3":
            return BGEM3Embedder(**kwargs)

        case "e5" | "multilingual-e5":
            return MultilingualE5Embedder(**kwargs)

        case "openai":
            model = model or "text-embedding-3-large"
            return OpenAIEmbedder(model=model, **kwargs)

        case "ollama":
            model = model or "bge-m3"
            return OllamaEmbedder(model=model, **kwargs)

        case "cohere":
            from .cohere_embedder import CohereEmbedder
            model = model or "embed-multilingual-v3.0"
            return CohereEmbedder(model=model, **kwargs)

        case "mistral":
            from .mistral_embedder import MistralEmbedder
            return MistralEmbedder(**kwargs)

        case "sbert":
            from .sbert_embedder import SBERTEmbedder
            model = model or "all-mpnet-base-v2"
            return SBERTEmbedder(model=model, **kwargs)

        case "gte-qwen2":
            from .gte_qwen2 import GTEQwen2Embedder
            return GTEQwen2Embedder(**kwargs)

        case "jina":
            from .jina_embedder import JinaEmbedder
            return JinaEmbedder(**kwargs)

        case _:
            raise ValueError(
                f"Unknown model_id: '{model_id}'. "
                f"Valid providers: bge-m3, e5, openai, ollama, cohere, mistral, sbert, gte-qwen2, jina"
            )
```

### 11. Qdrant Integration Helper

```python
# vector_db/qdrant_helper.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams,
    SparseVector,
)
from embedders.base import BaseEmbedder, HybridOutput


def create_collection(client: QdrantClient, name: str, embedder: BaseEmbedder):
    """Create a Qdrant collection sized to the embedder's output."""
    client.recreate_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=embedder.embedding_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams())
        }
    )


def upsert_hybrid(
    client: QdrantClient,
    collection: str,
    hybrid_output: HybridOutput,
    ids: list[str],
    payloads: list[dict] | None = None,
):
    """Upsert BGE-M3 hybrid (dense + sparse) vectors into Qdrant."""
    points = []
    for i, (id_, dense, sparse) in enumerate(
        zip(ids, hybrid_output.dense, hybrid_output.sparse)
    ):
        sparse_vec = SparseVector(
            indices=list(sparse.keys()),
            values=list(sparse.values())
        )
        points.append(PointStruct(
            id=id_,
            vector={"dense": dense, "sparse": sparse_vec},
            payload=(payloads[i] if payloads else {"text": hybrid_output.texts[i]})
        ))
    client.upsert(collection_name=collection, points=points)


def search_hybrid(
    client: QdrantClient,
    collection: str,
    embedder: BaseEmbedder,
    query: str,
    top_k: int = 10,
):
    """
    Hybrid search: dense semantic + sparse lexical, fused via Reciprocal Rank Fusion.
    Requires an embedder that supports encode_hybrid() (i.e. BGEM3Embedder).
    """
    hybrid = embedder.encode_hybrid([query])
    dense_vec  = hybrid.dense[0]
    sparse_vec = SparseVector(
        indices=list(hybrid.sparse[0].keys()),
        values=list(hybrid.sparse[0].values())
    )
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            {"query": dense_vec,  "using": "dense",  "limit": top_k * 2},
            {"query": sparse_vec, "using": "sparse", "limit": top_k * 2},
        ],
        query={"fusion": "rrf"},   # Reciprocal Rank Fusion
        limit=top_k,
    )
    return results
```

### 12. Full Pipeline Example

```python
# example_pipeline.py
from embedders.factory import get_embedder
from vector_db.qdrant_helper import create_collection, upsert_hybrid, search_hybrid
from qdrant_client import QdrantClient

# Initialize embedder and DB client
embedder = get_embedder("bge-m3", device="cuda", use_fp16=True)
client   = QdrantClient(host="localhost", port=6333)

# Create collection
create_collection(client, "codebase", embedder)

# Prepare code chunks with metadata prefix
code_chunks = [
    "# File: sort.py | Language: Python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    ...",
    "# File: db.py | Language: Python\nclass DatabaseConnection:\n    def __init__(self, url):\n        self.url = url",
    "# File: sort.py | Language: Python\n# Сортировка слиянием\ndef merge(left, right):\n    result = []\n    ...",
]
ids = ["chunk_001", "chunk_002", "chunk_003"]

# Index with hybrid vectors
hybrid_output = embedder.encode_hybrid(code_chunks)
upsert_hybrid(client, "codebase", hybrid_output, ids)

# Search in Russian — finds both Russian comments and English code
results = search_hybrid(
    client, "codebase", embedder,
    query="алгоритм сортировки массива",
    top_k=5
)
for r in results.points:
    print(f"Score: {r.score:.4f} | {r.payload['text'][:80]}")
```

---

## Model Selection Guide

| Scenario | Recommended Model | Config |
|---|---|---|
| On-premise, GPU available | `bge-m3` | `device="cuda", use_fp16=True` |
| On-premise, CPU only | `bge-m3` | `device="cpu"` or `ollama:bge-m3` |
| Maximum quality + GPU | `gte-qwen2` | `size="7b"` needs ~16GB VRAM |
| Maximum quality + limited GPU | `gte-qwen2` | `size="1.5b"` needs ~4GB VRAM |
| Zero infra / cloud | `openai:text-embedding-3-large` | `dimensions=1024` to reduce storage |
| Local dev, no ML deps | `ollama:bge-m3` | `ollama pull bge-m3` |
| Strict commercial license | `bge-m3` or `e5` | MIT / MIT |
| Non-commercial / research | `jina` | Best task-specific LoRA adapters |
| Hybrid code search | `bge-m3` | Use `encode_hybrid()` + Qdrant RRF |

### Dimension / Storage Reference

| Model | Dimensions | Storage per 1M vectors |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | ~1.5 GB |
| nomic-embed-text | 768 | ~3 GB |
| bge-m3 / e5-large | 1024 | ~4 GB |
| text-embedding-3-large | 3072 | ~12 GB |
| gte-qwen2-7b | 3584 | ~14 GB |

---

*Report scope: English + Russian + Code retrieval for vector database pipelines*
*Recommended primary model: **BAAI/bge-m3** — MIT, on-premise, EN+RU+Code, hybrid dense+sparse search*
*Last updated: March 2026*
