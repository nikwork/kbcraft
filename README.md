# kbcraft

> **Build structured, RAG-ready knowledge bases from Markdown — straight from your terminal.**

`kbcraft` is a Python command-line utility for authoring, organizing, and indexing collections of Markdown files into vector stores for use in Retrieval-Augmented Generation (RAG) pipelines. It bridges the gap between human-readable documentation and machine-queryable knowledge.

---

## Features

- 📝 **Scaffold knowledge bases** — generate structured sets of `.md` files from templates or outlines
- 🗂️ **Organize & validate** — enforce consistent frontmatter, naming conventions, and directory layout
- 🔍 **Chunk & embed** — split documents intelligently and generate vector embeddings
- 🗄️ **Push to vector stores** — built-in support for Chroma, Qdrant, Pinecone, and FAISS
- 🔄 **Incremental sync** — only re-index documents that have changed
- 🖥️ **Pure CLI** — scriptable, CI-friendly, no GUI required
