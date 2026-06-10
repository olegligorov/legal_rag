import pytest
from unittest.mock import patch
from langchain_core.documents import Document

# Patch config constants before importing Chunker so tests are independent of config values
with patch("config.CHUNK_MAX_CHARS", 100), patch("config.CHUNK_OVERLAP_CHARS", 20):
    import importlib
    import rag.chunker as chunker_module

    importlib.reload(chunker_module)
    Chunker = chunker_module.Chunker
    MAX_CHARS = 100
    OVERLAP_CHARS = 20


def make_doc(content, **meta):
    return Document(page_content=content, metadata={"law_id": "Test", "article": "Чл. 1", **meta})


class TestSplitByLength:
    def setup_method(self):
        self.chunker = Chunker()

    def test_short_text_returns_single_segment(self):
        text = "a" * 50
        assert self.chunker._split_by_length(text) == [text]

    def test_exact_max_chars_not_split(self):
        text = "a" * MAX_CHARS
        result = self.chunker._split_by_length(text)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_splits_at_word_boundary(self):
        # 6 words of ~17 chars each -> over MAX_CHARS=100
        text = " ".join(["word" * 4] * 6)
        result = self.chunker._split_by_length(text)
        assert len(result) > 1
        for seg in result:
            assert len(seg) <= MAX_CHARS

    def test_multiple_paragraphs_combined_under_limit(self):
        # Two short paragraphs that fit together
        text = "short para\nanother short para"
        result = self.chunker._split_by_length(text)
        assert len(result) == 1

    def test_multiple_paragraphs_split_when_over_limit(self):
        # Each paragraph is fine alone but combined would exceed MAX_CHARS
        para = "x" * 60
        text = f"{para}\n{para}"
        result = self.chunker._split_by_length(text)
        assert len(result) == 2

    def test_overlap_carried_to_next_segment(self):
        # Build a para that needs splitting; the second segment should start
        # with the last OVERLAP_CHARS chars of the first segment's content
        words = ["word"] * 30  # well over 100 chars
        text = " ".join(words)
        result = self.chunker._split_by_length(text)
        assert len(result) >= 2
        # Second segment should share some tail text with first
        tail = result[0][-OVERLAP_CHARS:]
        assert result[1].startswith(tail.strip())

    def test_no_infinite_loop_on_unbreakable_long_token(self):
        # A single token longer than MAX_CHARS with no spaces — would loop forever
        # if advance logic is wrong
        text = "а" * (MAX_CHARS * 3)
        result = self.chunker._split_by_length(text)
        assert len(result) >= 2
        total = sum(len(s) for s in result)
        # Some chars are duplicated via overlap but total should be >= original
        assert total >= len(text)

    def test_empty_string_returns_empty(self):
        assert self.chunker._split_by_length("") == []

    def test_only_newlines_returns_empty(self):
        assert self.chunker._split_by_length("\n\n\n") == []


class TestChunkDoc:
    def setup_method(self):
        self.chunker = Chunker()

    def test_short_doc_returns_single_chunk(self):
        doc = make_doc("short content")
        chunks = self.chunker._chunk_doc(doc)
        assert len(chunks) == 1
        assert chunks[0].page_content == "short content"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1

    def test_long_doc_returns_multiple_chunks(self):
        content = " ".join(["word"] * 60)  # well over 100 chars
        doc = make_doc(content)
        chunks = self.chunker._chunk_doc(doc)
        assert len(chunks) > 1

    def test_chunk_index_and_total_correct(self):
        content = " ".join(["word"] * 60)
        doc = make_doc(content)
        chunks = self.chunker._chunk_doc(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_metadata_preserved_in_chunks(self):
        doc = make_doc("short", chapter="Глава I", section="Раздел 1")
        chunks = self.chunker._chunk_doc(doc)
        assert chunks[0].metadata["chapter"] == "Глава I"
        assert chunks[0].metadata["section"] == "Раздел 1"
        assert chunks[0].metadata["law_id"] == "Test"

    def test_no_prefix_in_page_content(self):
        # Prefix must NOT be injected into page_content anymore
        doc = make_doc("article body text", chapter="Глава I")
        chunks = self.chunker._chunk_doc(doc)
        for chunk in chunks:
            assert "Глава" not in chunk.page_content
            assert "Test" not in chunk.page_content

    def test_each_chunk_under_max_chars(self):
        content = " ".join(["word"] * 80)
        doc = make_doc(content)
        for chunk in self.chunker._chunk_doc(doc):
            assert len(chunk.page_content) <= MAX_CHARS


class TestCreateChunks:
    def setup_method(self):
        self.chunker = Chunker()

    def test_empty_input(self):
        assert self.chunker.create_chunks([]) == []

    def test_multiple_docs_all_chunked(self):
        docs = [make_doc("short"), make_doc(" ".join(["word"] * 60))]
        chunks = self.chunker.create_chunks(docs)
        assert len(chunks) >= 2

    def test_metadata_not_shared_between_docs(self):
        doc1 = make_doc("doc one content", article="Чл. 1")
        doc2 = make_doc("doc two content", article="Чл. 2")
        chunks = self.chunker.create_chunks([doc1, doc2])
        articles = [c.metadata["article"] for c in chunks]
        assert "Чл. 1" in articles
        assert "Чл. 2" in articles
