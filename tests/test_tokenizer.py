"""
Tests for tokenizer backends and factory.
"""
from unittest.mock import MagicMock, patch

import pytest

from kbcraft.tokenizer import (
    HFTokenizer,
    OllamaTokenizer,
    WhitespaceTokenizer,
    get_tokenizer,
)


# ---------------------------------------------------------------------------
# WhitespaceTokenizer
# ---------------------------------------------------------------------------


class TestWhitespaceTokenizer:
    def test_backend_label(self):
        assert WhitespaceTokenizer().backend == "whitespace"

    def test_model_name(self):
        assert WhitespaceTokenizer(model="all-minilm").model_name == "all-minilm"

    def test_tokenize_simple(self):
        tok = WhitespaceTokenizer()
        assert tok.tokenize("hello world") == ["hello", "world"]

    def test_tokenize_empty(self):
        assert WhitespaceTokenizer().tokenize("") == []

    def test_count(self):
        assert WhitespaceTokenizer().count("one two three") == 3

    def test_count_empty(self):
        assert WhitespaceTokenizer().count("") == 0

    def test_count_batch(self):
        tok = WhitespaceTokenizer()
        assert tok.count_batch(["a b", "c d e", ""]) == [2, 3, 0]

    def test_truncate_within_limit(self):
        tok = WhitespaceTokenizer()
        assert tok.truncate("a b c", 10) == "a b c"

    def test_truncate_at_limit(self):
        tok = WhitespaceTokenizer()
        assert tok.truncate("a b c d e", 3) == "a b c"

    def test_repr(self):
        assert "WhitespaceTokenizer" in repr(WhitespaceTokenizer(model="x"))


# ---------------------------------------------------------------------------
# HFTokenizer
# ---------------------------------------------------------------------------


class TestHFTokenizer:
    def test_backend_label(self):
        tok = HFTokenizer(model="all-minilm")
        assert tok.backend == "hf"

    def test_model_name(self):
        tok = HFTokenizer(model="all-minilm")
        assert tok.model_name == "all-minilm"

    def test_tokenize_returns_list_of_strings(self):
        tok = HFTokenizer(model="all-minilm")
        tokens = tok.tokenize("hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_tokenize_nonempty(self):
        tok = HFTokenizer(model="all-minilm")
        assert len(tok.tokenize("hello world")) > 0

    def test_count_positive(self):
        tok = HFTokenizer(model="all-minilm")
        assert tok.count("hello world") > 0

    def test_count_batch(self):
        tok = HFTokenizer(model="all-minilm")
        counts = tok.count_batch(["hello", "hello world"])
        assert counts[1] >= counts[0]

    def test_longer_text_more_tokens(self):
        tok = HFTokenizer(model="all-minilm")
        short = tok.count("hi")
        long = tok.count("the quick brown fox jumps over the lazy dog " * 3)
        assert long > short

    def test_unknown_model_uses_bert_fallback(self):
        tok = HFTokenizer(model="unknown-model-xyz")
        assert tok.count("hello world") > 0

    def test_hf_repo_override(self):
        tok = HFTokenizer(model="anything", hf_repo="bert-base-uncased")
        assert tok.count("hello") > 0

    def test_truncate_reduces_token_count(self):
        tok = HFTokenizer(model="all-minilm")
        original = "the quick brown fox jumps over the lazy dog"
        full_count = tok.count(original)
        truncated = tok.truncate(original, 3)
        assert tok.count(truncated) <= full_count

    def test_import_error_on_bad_repo(self):
        with pytest.raises(ImportError):
            HFTokenizer(model="x", hf_repo="this/repo-does-not-exist-xyz-abc-123")


# ---------------------------------------------------------------------------
# OllamaTokenizer (mocked)
# ---------------------------------------------------------------------------


class TestOllamaTokenizer:
    def _make_tok(self, response_tokens):
        tok = OllamaTokenizer(model="all-minilm", host="http://mock:11434")
        mock_resp = MagicMock()
        mock_resp.read.return_value = (
            '{"tokens": ' + str(response_tokens) + "}"
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return tok, mock_resp

    def test_backend_label(self):
        assert OllamaTokenizer().backend == "ollama"

    def test_model_name(self):
        assert OllamaTokenizer(model="bge-m3").model_name == "bge-m3"

    def test_tokenize_calls_api(self):
        tok, mock_resp = self._make_tok([1, 2, 3])
        with patch("urllib.request.urlopen", return_value=mock_resp):
            tokens = tok.tokenize("hello world")
        assert tokens == ["1", "2", "3"]

    def test_count_uses_token_length(self):
        tok, mock_resp = self._make_tok([10, 20, 30, 40])
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert tok.count("anything") == 4

    def test_connection_error_on_url_error(self):
        import urllib.error

        tok = OllamaTokenizer(model="all-minilm", host="http://mock:11434")
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
                tok.tokenize("hello")


# ---------------------------------------------------------------------------
# get_tokenizer factory
# ---------------------------------------------------------------------------


class TestGetTokenizer:
    def test_returns_hf_by_default(self):
        tok = get_tokenizer("all-minilm")
        assert tok.backend == "hf"

    def test_returns_ollama_when_preferred(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"tokens": [1, 2]}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            tok = get_tokenizer(
                "all-minilm",
                ollama_host="http://mock:11434",
                prefer_ollama=True,
            )
        assert tok.backend == "ollama"

    def test_prefer_ollama_without_host_falls_back(self):
        # prefer_ollama=True but no host → should NOT return Ollama
        tok = get_tokenizer("all-minilm", ollama_host=None, prefer_ollama=True)
        assert tok.backend != "ollama"

    def test_falls_back_to_whitespace_when_hf_unavailable(self):
        import kbcraft.tokenizer as tm

        original = tm._PREFER_HF
        tm._PREFER_HF = False
        try:
            tok = get_tokenizer("all-minilm")
            assert tok.backend == "whitespace"
        finally:
            tm._PREFER_HF = original

    def test_tokenizer_plugs_into_chunker(self):
        from kbcraft.chunker import Chunker

        tok = get_tokenizer("all-minilm")
        # Build a text long enough to force multiple chunks
        text = " ".join(["word"] * 60)
        chunker = Chunker(max_chunk_tokens=20, chunk_overlap=0, tokenize=tok.tokenize)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            # BERT adds [CLS] and [SEP] special tokens, so actual count may
            # slightly exceed max_chunk_tokens for the last chunk.
            assert chunk.token_count <= 25
