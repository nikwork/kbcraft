"""
Document chunking functionality for splitting documents intelligently.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_MD_EXTENSIONS = {".md", ".mdx"}

_MD_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
    ("######", "h6"),
]


def _whitespace_tokenize(text: str) -> List[str]:
    """Split text on whitespace. Approximates subword token counts."""
    return text.split()


@dataclass
class Chunk:
    """A single text chunk produced by the Chunker.

    Attributes:
        text: The chunk's text content.
        source: File path or identifier the chunk came from (empty string if unknown).
        index: Zero-based position of this chunk within its source.
        token_count: Number of tokens in this chunk.
    """

    text: str
    source: str
    index: int
    token_count: int


class Chunker:
    """Splits text or files into token-bounded chunks using LangChain splitters.

    For plain text files :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`
    is used with a custom ``length_function`` so chunk sizes are measured in tokens.

    For Markdown files (``.md`` / ``.mdx``) the pipeline is:

    1. :class:`~langchain_text_splitters.MarkdownHeaderTextSplitter` splits the
       document at header boundaries (h1–h6), preserving headers in the text.
    2. Each section is further split by
       :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter` to
       enforce the ``max_chunk_tokens`` limit.

    By default tokens are whitespace-delimited words, which approximates
    GPT-style subword token counts well enough for most prose and code.
    Inject a custom ``tokenize`` function to use a real tokenizer::

        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        chunker = Chunker(
            max_chunk_tokens=512,
            tokenize=lambda text: [str(t) for t in enc.encode(text)],
        )

    When ``prepend_source=True`` (the default) every chunk produced by
    :meth:`chunk_file` or :meth:`chunk_files` is prefixed with a two-line
    header that embeds the file name and path directly in the chunk text::

        File: notes.md
        Path: docs/notes.md

        <chunk content …>

    This gives the LLM immediate provenance context without requiring
    separate metadata look-ups.  The header tokens are included in
    ``Chunk.token_count``.  Pass ``prepend_source=False`` to disable.

    Args:
        max_chunk_tokens: Maximum number of tokens per chunk. Default: 512.
        chunk_overlap: Tokens shared between adjacent chunks. Default: 64.
            Must be less than ``max_chunk_tokens``.
        tokenize: Callable that splits a string into a list of string tokens.
            Defaults to whitespace splitting.
        prepend_source: Prepend ``File:`` / ``Path:`` header to every chunk
            produced by :meth:`chunk_file`. Default: ``True``.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        chunk_overlap: int = 64,
        tokenize: Optional[Callable[[str], List[str]]] = None,
        prepend_source: bool = True,
    ) -> None:
        if max_chunk_tokens <= 0:
            raise ValueError(f"max_chunk_tokens must be positive, got {max_chunk_tokens}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= max_chunk_tokens:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"max_chunk_tokens ({max_chunk_tokens})"
            )
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.prepend_source = prepend_source
        self._tokenize = tokenize or _whitespace_tokenize

        length_fn = lambda text: len(self._tokenize(text))  # noqa: E731

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_tokens,
            chunk_overlap=chunk_overlap,
            length_function=length_fn,
        )
        self._md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MD_HEADERS,
            strip_headers=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, source: str = "") -> List[Chunk]:
        """Split *text* into overlapping chunks bounded by ``max_chunk_tokens``.

        Uses :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`.
        Prefer :meth:`chunk_markdown` for Markdown content, or use
        :meth:`chunk_file` which auto-detects by file extension.

        Args:
            text: The input text to split.
            source: Optional label for the origin of the text (e.g. file path).

        Returns:
            List of :class:`Chunk` objects. Returns a single empty-text chunk
            for empty input so callers always receive a non-empty list.
        """
        if not text.strip():
            return [Chunk(text="", source=source, index=0, token_count=0)]

        docs = self._text_splitter.create_documents([text])
        return [
            Chunk(
                text=doc.page_content,
                source=source,
                index=i,
                token_count=len(self._tokenize(doc.page_content)),
            )
            for i, doc in enumerate(docs)
        ]

    def chunk_markdown(self, text: str, source: str = "") -> List[Chunk]:
        """Split Markdown *text* using structural then token boundaries.

        First splits at header lines (h1–h6) using
        :class:`~langchain_text_splitters.MarkdownHeaderTextSplitter`, then
        applies :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`
        to sections that exceed ``max_chunk_tokens``.

        Args:
            text: Markdown text to chunk.
            source: Optional label for the origin of the text.

        Returns:
            List of :class:`Chunk` objects preserving Markdown structure.
        """
        if not text.strip():
            return [Chunk(text="", source=source, index=0, token_count=0)]

        # 1. Split by headers
        header_docs = self._md_header_splitter.split_text(text)

        # 2. Further split oversized sections by tokens
        final_docs = self._text_splitter.split_documents(header_docs)

        return [
            Chunk(
                text=doc.page_content,
                source=source,
                index=i,
                token_count=len(self._tokenize(doc.page_content)),
            )
            for i, doc in enumerate(final_docs)
        ]

    def chunk_file(self, path: Path, base_dir: Optional[Path] = None) -> List[Chunk]:
        """Read *path* and return chunks with ``source`` set to the file path.

        Automatically uses :meth:`chunk_markdown` for ``.md`` and ``.mdx``
        files, and :meth:`chunk_text` for everything else.

        When ``prepend_source=True`` (the default) each chunk is prefixed with::

            File: <filename>
            Path: <relative-or-absolute path>

        Args:
            path: Path to a UTF-8 text file.
            base_dir: Optional base directory used to compute a relative path
                for the ``Path:`` header. Defaults to the current working
                directory when ``prepend_source=True``.

        Returns:
            List of :class:`Chunk` objects.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        source = str(path)

        if path.suffix.lower() in _MD_EXTENSIONS:
            chunks = self.chunk_markdown(text, source=source)
        else:
            chunks = self.chunk_text(text, source=source)

        if self.prepend_source:
            chunks = self._prepend_file_header(chunks, path, base_dir)

        return chunks

    def chunk_files(self, paths: List[Path], base_dir: Optional[Path] = None) -> List[Chunk]:
        """Chunk every file in *paths* and return all chunks in order.

        Args:
            paths: Iterable of file paths to process.
            base_dir: Passed through to :meth:`chunk_file` for relative-path
                computation in the ``File:`` / ``Path:`` header.

        Returns:
            Flat list of :class:`Chunk` objects across all files.
        """
        result: List[Chunk] = []
        for path in paths:
            result.extend(self.chunk_file(path, base_dir=base_dir))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepend_file_header(
        self, chunks: List[Chunk], path: Path, base_dir: Optional[Path]
    ) -> List[Chunk]:
        """Return new Chunk objects with a ``File:`` / ``Path:`` header prepended."""
        try:
            rel = path.relative_to(base_dir or Path.cwd())
        except ValueError:
            rel = path

        header = f"File: {path.name}\nPath: {rel}\n\n"
        updated: List[Chunk] = []
        for chunk in chunks:
            new_text = header + chunk.text
            updated.append(
                Chunk(
                    text=new_text,
                    source=chunk.source,
                    index=chunk.index,
                    token_count=len(self._tokenize(new_text)),
                )
            )
        return updated
