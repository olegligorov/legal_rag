"""Unit tests for backend/evaluation/metrics.py.

Covers the pure-Python additions for citation/cost metrics — the
LLM/embedding-backed metrics still need integration testing through
``Evaluator`` since they depend on real models.
"""

from __future__ import annotations

import pytest

from evaluation.metrics import (
    citation_precision,
    citation_recall,
    extract_cited_articles,
    precision_at_k,
    recall_at_k,
    tool_call_count,
    _normalize_article,
    _normalize_law_id,
    _split_citation,
)


KT = "КОДЕКС НА ТРУДА"
ZP = "ЗАКОН ЗА ЗАЩИТА НА ПОТРЕБИТЕЛИТЕ"


class TestNormalize:
    def test_law_id_uppercase(self):
        assert _normalize_law_id("Кодекс на труда") == KT

    def test_law_id_collapses_whitespace(self):
        assert _normalize_law_id("Кодекс  на\tтруда") == KT

    def test_article_strips_trailing_period(self):
        assert _normalize_article("Чл. 155.") == "ЧЛ. 155"

    def test_article_strips_alenia_qualifier(self):
        assert _normalize_article("Чл. 155, ал. 4") == "ЧЛ. 155"

    def test_article_handles_complex_alenia(self):
        assert _normalize_article("Чл. 155, ал. 4а,") == "ЧЛ. 155"

    def test_article_with_letter_suffix(self):
        # Real article in the corpus: ЗЗП Чл. 161з. Letter suffix must
        # survive normalization — without this, citations to 161з would
        # incorrectly match Чл. 161 in the eval.
        assert _normalize_article("Чл. 161з") == "ЧЛ. 161З"

    def test_article_letter_suffix_with_alenia(self):
        assert _normalize_article("Чл. 161з, ал. 2") == "ЧЛ. 161З"

    def test_article_letter_suffix_with_trailing_period(self):
        assert _normalize_article("Чл. 161з.") == "ЧЛ. 161З"

    def test_law_id_nfc_normalization(self):
        # Compose two visually-identical strings, one with combining
        # characters. After NFC both should compare equal. (Defensive:
        # the corpus is consistently composed, but mixed-source text
        # could decompose.)
        import unicodedata
        composed = "Кодекс на труда"
        decomposed = unicodedata.normalize("NFD", composed)
        assert _normalize_law_id(composed) == _normalize_law_id(decomposed)


class TestSplitCitation:
    def test_simple(self):
        assert _split_citation("[Чл. 155, Кодекс на труда]") == ("ЧЛ. 155", KT)

    def test_with_alenia(self):
        assert _split_citation("[Чл. 155, ал. 4, Кодекс на труда]") == ("ЧЛ. 155", KT)

    def test_with_multiple_alenia_parts(self):
        # If the model emits both ал. + точка, we still want article + law
        assert _split_citation("[Чл. 50, ал. 1, Закон за защита на потребителите]") == (
            "ЧЛ. 50",
            ZP,
        )

    def test_malformed_returns_none(self):
        assert _split_citation("[just text]") is None

    def test_empty_returns_none(self):
        assert _split_citation("[]") is None


class TestExtractCitedArticles:
    def test_no_citations(self):
        assert extract_cited_articles("plain text") == []

    def test_single_citation(self):
        assert extract_cited_articles("answer [Чл. 155, Кодекс на труда].") == [
            ("ЧЛ. 155", KT)
        ]

    def test_multiple_distinct(self):
        text = "A [Чл. 155, Кодекс на труда] and B [Чл. 50, Закон за защита на потребителите]."
        assert extract_cited_articles(text) == [("ЧЛ. 155", KT), ("ЧЛ. 50", ZP)]

    def test_duplicates_preserved_in_list(self):
        text = "[Чл. 155, Кодекс на труда] [Чл. 155, ал. 4, Кодекс на труда]"
        # Both normalize to same article — extraction keeps both, dedup is
        # the metric's responsibility.
        assert extract_cited_articles(text) == [("ЧЛ. 155", KT), ("ЧЛ. 155", KT)]


class TestCitationRecall:
    def test_full_match(self):
        expected = [{"law_id": KT, "article": "Чл. 155."}]
        answer = "result [Чл. 155, Кодекс на труда]."
        assert citation_recall(answer, expected) == 1.0

    def test_partial_match(self):
        expected = [
            {"law_id": KT, "article": "Чл. 155."},
            {"law_id": KT, "article": "Чл. 156."},
        ]
        answer = "[Чл. 155, Кодекс на труда]"
        assert citation_recall(answer, expected) == 0.5

    def test_no_citations(self):
        expected = [{"law_id": KT, "article": "Чл. 155."}]
        assert citation_recall("plain text", expected) == 0.0

    def test_empty_expected(self):
        assert citation_recall("[Чл. 155, Кодекс на труда]", []) == 0.0

    def test_alenia_qualifier_still_matches(self):
        expected = [{"law_id": KT, "article": "Чл. 155."}]
        answer = "[Чл. 155, ал. 4, Кодекс на труда]"
        assert citation_recall(answer, expected) == 1.0

    def test_cross_law_recall(self):
        expected = [
            {"law_id": KT, "article": "Чл. 155."},
            {"law_id": ZP, "article": "Чл. 50."},
        ]
        answer = "[Чл. 155, Кодекс на труда] and [Чл. 50, Закон за защита на потребителите]"
        assert citation_recall(answer, expected) == 1.0


class TestCitationPrecision:
    def test_all_correct(self):
        expected = [{"law_id": KT, "article": "Чл. 155."}]
        answer = "[Чл. 155, Кодекс на труда]"
        assert citation_precision(answer, expected) == 1.0

    def test_extra_irrelevant_citation(self):
        expected = [{"law_id": KT, "article": "Чл. 155."}]
        answer = "[Чл. 155, Кодекс на труда] and [Чл. 999, Кодекс на труда]"
        assert citation_precision(answer, expected) == 0.5

    def test_no_citations_returns_none(self):
        # Per docstring: an answer with no citations returns None so the
        # aggregator can distinguish "didn't cite" from "cited wrong"
        # rather than silently averaging 0.0.
        expected = [{"law_id": KT, "article": "Чл. 155."}]
        assert citation_precision("plain answer", expected) is None

    def test_dedup_by_normalized_id(self):
        # Same article cited twice with different ал. qualifiers — counts once.
        expected = [{"law_id": KT, "article": "Чл. 155."}]
        answer = "[Чл. 155, Кодекс на труда] [Чл. 155, ал. 4, Кодекс на труда]"
        assert citation_precision(answer, expected) == 1.0


class TestRetrievalMetricsRespectNormalization:
    """Existing precision_at_k / recall_at_k now route through normalizer.

    Verifies they don't break when retrieved law_ids differ in case from
    the expected ones.
    """

    def test_recall_case_insensitive_law(self):
        retrieved = [{"law_id": "Кодекс на труда", "article": "Чл. 155."}]
        expected = [{"law_id": KT, "article": "Чл. 155."}]
        assert recall_at_k(retrieved, expected, 5) == 1.0

    def test_precision_case_insensitive_law(self):
        retrieved = [{"law_id": "Кодекс на труда", "article": "Чл. 155."}]
        expected = [{"law_id": KT, "article": "Чл. 155."}]
        assert precision_at_k(retrieved, expected, 1) == 1.0


class TestToolCallCount:
    def test_none_trace(self):
        assert tool_call_count(None) == 0

    def test_empty_trace(self):
        assert tool_call_count([]) == 0

    def test_counts_tool_calls(self):
        trace = [
            {"type": "thought", "content": "x"},
            {"type": "tool_call", "name": "query_rag_tool"},
            {"type": "tool_result", "content": {}},
            {"type": "tool_call", "name": "batch_query_tool"},
            {"type": "tool_result", "content": {}},
            {"type": "answer", "content": "..."},
        ]
        assert tool_call_count(trace) == 2

    def test_ignores_non_tool_call_events(self):
        trace = [
            {"type": "thought", "content": "..."},
            {"type": "answer", "content": "..."},
        ]
        assert tool_call_count(trace) == 0


@pytest.mark.parametrize(
    "answer, expected, want_recall, want_precision",
    [
        # fully right
        (
            "[Чл. 1, Кодекс на труда] and [Чл. 2, Кодекс на труда]",
            [{"law_id": KT, "article": "Чл. 1."}, {"law_id": KT, "article": "Чл. 2."}],
            1.0,
            1.0,
        ),
        # half right
        (
            "[Чл. 1, Кодекс на труда]",
            [{"law_id": KT, "article": "Чл. 1."}, {"law_id": KT, "article": "Чл. 2."}],
            0.5,
            1.0,
        ),
        # over-cites
        (
            "[Чл. 1, Кодекс на труда] [Чл. 999, Кодекс на труда]",
            [{"law_id": KT, "article": "Чл. 1."}],
            1.0,
            0.5,
        ),
    ],
)
def test_combined_recall_precision(answer, expected, want_recall, want_precision):
    assert citation_recall(answer, expected) == want_recall
    assert citation_precision(answer, expected) == want_precision
