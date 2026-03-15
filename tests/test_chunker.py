"""
Tests for document chunking functionality.
"""
import pytest

from kbcraft.chunker import Chunk, Chunker


class TestChunkerInit:
    def test_defaults(self):
        c = Chunker()
        assert c.max_chunk_tokens == 512
        assert c.chunk_overlap == 64

    def test_custom_params(self):
        c = Chunker(max_chunk_tokens=128, chunk_overlap=16)
        assert c.max_chunk_tokens == 128
        assert c.chunk_overlap == 16

    def test_zero_overlap_allowed(self):
        c = Chunker(max_chunk_tokens=10, chunk_overlap=0)
        assert c.chunk_overlap == 0

    def test_invalid_max_chunk_tokens(self):
        with pytest.raises(ValueError, match="max_chunk_tokens"):
            Chunker(max_chunk_tokens=0)

    def test_negative_max_chunk_tokens(self):
        with pytest.raises(ValueError, match="max_chunk_tokens"):
            Chunker(max_chunk_tokens=-1)

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            Chunker(max_chunk_tokens=10, chunk_overlap=-1)

    def test_overlap_equal_to_max_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            Chunker(max_chunk_tokens=10, chunk_overlap=10)

    def test_overlap_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            Chunker(max_chunk_tokens=10, chunk_overlap=11)


class TestChunkText:
    def test_empty_text_returns_one_chunk(self):
        chunks = Chunker().chunk_text("")
        assert len(chunks) == 1
        assert chunks[0].text == ""
        assert chunks[0].token_count == 0

    def test_short_text_single_chunk(self):
        chunks = Chunker(max_chunk_tokens=100, chunk_overlap=0).chunk_text("hello world")
        assert len(chunks) == 1
        assert "hello" in chunks[0].text
        assert chunks[0].token_count == 2

    def test_chunk_respects_max_tokens(self):
        words = " ".join(f"word{i}" for i in range(100))
        chunks = Chunker(max_chunk_tokens=10, chunk_overlap=0).chunk_text(words)
        for chunk in chunks:
            assert chunk.token_count <= 10

    def test_long_text_produces_multiple_chunks(self):
        words = " ".join(f"w{i}" for i in range(50))
        chunks = Chunker(max_chunk_tokens=10, chunk_overlap=0).chunk_text(words)
        assert len(chunks) > 1

    def test_chunk_index_is_sequential(self):
        words = " ".join(f"w{i}" for i in range(30))
        chunks = Chunker(max_chunk_tokens=5, chunk_overlap=0).chunk_text(words)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_source_propagated(self):
        words = " ".join(f"w{i}" for i in range(20))
        chunks = Chunker(max_chunk_tokens=5, chunk_overlap=0).chunk_text(words, source="my_doc")
        for chunk in chunks:
            assert chunk.source == "my_doc"

    def test_chunk_is_dataclass(self):
        chunks = Chunker(max_chunk_tokens=100, chunk_overlap=0).chunk_text("hello world")
        assert isinstance(chunks[0], Chunk)

    def test_overlap_produces_shared_content(self):
        # With overlap > 0, adjacent chunks should share some tokens
        words = " ".join(f"w{i}" for i in range(30))
        no_overlap = Chunker(max_chunk_tokens=10, chunk_overlap=0).chunk_text(words)
        with_overlap = Chunker(max_chunk_tokens=10, chunk_overlap=3).chunk_text(words)
        # Overlap means more chunks for the same content
        assert len(with_overlap) >= len(no_overlap)


class TestChunkMarkdown:
    def test_splits_at_headers(self):
        text = "# Section A\n\nParagraph one.\n\n# Section B\n\nParagraph two."
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_markdown(text)
        texts = [c.text for c in chunks]
        assert any("Section A" in t for t in texts)
        assert any("Section B" in t for t in texts)

    def test_empty_text_returns_one_chunk(self):
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_markdown("")
        assert len(chunks) == 1
        assert chunks[0].text == ""

    def test_source_propagated(self):
        text = "# A\n\nhello\n\n# B\n\nworld"
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_markdown(
            text, source="doc.md"
        )
        for chunk in chunks:
            assert chunk.source == "doc.md"

    def test_indexes_sequential(self):
        text = "\n\n".join(f"# H{i}\n\npara {i}" for i in range(5))
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_markdown(text)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_oversized_section_further_split(self):
        # One section with 50 words, max=10 → must produce multiple chunks
        words = " ".join(f"word{i}" for i in range(50))
        text = f"# Big Section\n\n{words}"
        chunks = Chunker(max_chunk_tokens=10, chunk_overlap=0).chunk_markdown(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= 10

    def test_token_count_matches_text(self):
        text = "# Header\n\nsome words here\n\n# Another\n\nmore words"
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_markdown(text)
        for chunk in chunks:
            expected = len(chunk.text.split())
            assert chunk.token_count == expected

    def test_no_headers_treated_as_single_section(self):
        text = "Just plain text without any headers.\n\nAnother paragraph."
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_markdown(text)
        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        assert "plain text" in combined


class TestChunkFile:
    def test_md_file_uses_markdown_splitter(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(
            "# Intro\n\nhello world\n\n# Body\n\nmore text here", encoding="utf-8"
        )
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_file(f)
        texts = " ".join(c.text for c in chunks)
        assert "Intro" in texts
        assert "Body" in texts

    def test_mdx_file_uses_markdown_splitter(self, tmp_path):
        f = tmp_path / "doc.mdx"
        f.write_text("# A\n\nhello\n\n# B\n\nworld", encoding="utf-8")
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_file(f)
        texts = " ".join(c.text for c in chunks)
        assert "hello" in texts

    def test_txt_file_uses_text_splitter(self, tmp_path):
        words = " ".join(f"w{i}" for i in range(50))
        f = tmp_path / "doc.txt"
        f.write_text(words, encoding="utf-8")
        # prepend_source=False so token_count reflects only content
        chunks = Chunker(max_chunk_tokens=10, chunk_overlap=0, prepend_source=False).chunk_file(f)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= 10

    def test_source_set_to_file_path(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# H\n\nhello world", encoding="utf-8")
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_file(f)
        assert all(chunk.source == str(f) for chunk in chunks)


class TestPrependSource:
    def test_header_present_by_default(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Hello\n\nsome content", encoding="utf-8")
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_file(f)
        assert all(c.text.startswith("File: notes.md\nPath:") for c in chunks)

    def test_header_contains_filename(self, tmp_path):
        f = tmp_path / "guide.txt"
        f.write_text("hello world", encoding="utf-8")
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_file(f)
        assert "File: guide.txt" in chunks[0].text

    def test_header_contains_relative_path(self, tmp_path):
        sub = tmp_path / "docs"
        sub.mkdir()
        f = sub / "notes.txt"
        f.write_text("hello world", encoding="utf-8")
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_file(
            f, base_dir=tmp_path
        )
        assert "Path: docs/notes.txt" in chunks[0].text

    def test_token_count_includes_header(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello world", encoding="utf-8")
        chunker = Chunker(max_chunk_tokens=200, chunk_overlap=0)
        chunks = chunker.chunk_file(f)
        expected = len(chunks[0].text.split())
        assert chunks[0].token_count == expected

    def test_prepend_source_false_omits_header(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello world", encoding="utf-8")
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0, prepend_source=False).chunk_file(f)
        assert not chunks[0].text.startswith("File:")

    def test_chunk_files_inherits_prepend(self, tmp_path):
        for name in ("a.txt", "b.txt"):
            (tmp_path / name).write_text("foo bar baz", encoding="utf-8")
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_files(
            list(tmp_path.glob("*.txt"))
        )
        assert all("File:" in c.text for c in chunks)

    def test_base_dir_passed_through_chunk_files(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        f = sub / "code.txt"
        f.write_text("hello world", encoding="utf-8")
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_files(
            [f], base_dir=tmp_path
        )
        assert "Path: src/code.txt" in chunks[0].text


class TestChunkFiles:
    def test_combines_multiple_files(self, tmp_path):
        for i in range(3):
            (tmp_path / f"file{i}.txt").write_text(
                " ".join(f"w{j}" for j in range(20)), encoding="utf-8"
            )
        paths = list(tmp_path.glob("*.txt"))
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_files(paths)
        sources = {c.source for c in chunks}
        assert len(sources) == 3

    def test_empty_list_returns_empty(self):
        assert Chunker().chunk_files([]) == []

    def test_indexes_reset_per_file(self, tmp_path):
        for i in range(2):
            words = " ".join(f"w{j}" for j in range(30))
            (tmp_path / f"f{i}.txt").write_text(words, encoding="utf-8")
        paths = sorted(tmp_path.glob("*.txt"))
        chunks = Chunker(max_chunk_tokens=5, chunk_overlap=0).chunk_files(paths)
        sources: dict = {}
        for chunk in chunks:
            sources.setdefault(chunk.source, []).append(chunk.index)
        for indexes in sources.values():
            assert indexes[0] == 0

    def test_mixed_md_and_txt(self, tmp_path):
        (tmp_path / "a.md").write_text("# H\n\nhello world", encoding="utf-8")
        (tmp_path / "b.txt").write_text("foo bar baz", encoding="utf-8")
        chunks = Chunker(max_chunk_tokens=200, chunk_overlap=0).chunk_files(
            [tmp_path / "a.md", tmp_path / "b.txt"]
        )
        combined = " ".join(c.text for c in chunks)
        assert "hello" in combined
        assert "foo" in combined


class TestCustomTokenizer:
    def test_custom_tokenize_controls_chunk_size(self):
        # Character tokenizer: each char = 1 token → max=5 chars per chunk
        chunker = Chunker(max_chunk_tokens=5, chunk_overlap=0, tokenize=list)
        chunks = chunker.chunk_text("abcdefghij")
        for chunk in chunks:
            assert chunk.token_count <= 5

    def test_custom_tokenize_used_for_length(self):
        calls = []

        def counting_tokenize(text):
            tokens = text.split()
            calls.append(len(tokens))
            return tokens

        Chunker(
            max_chunk_tokens=100, chunk_overlap=0, tokenize=counting_tokenize
        ).chunk_text("hello world foo")
        assert len(calls) > 0
