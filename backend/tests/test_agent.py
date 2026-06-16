"""Unit tests for backend/agent/.

Covers the pure helpers that don't require a running MCP server or LLM:
- _parse_tool_result (content-shape normalization from MCP)
- _SourceCollector (dedup across nested batch results)
- _validate_citations (preserved/dropped/invented diff)
- _detect_repeated_call (loop detector)
- AGENT_ALLOWED_TOOLS filter

End-to-end tests (full graph + live MCP) are in tests/test_agent_e2e.py.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import LOOP_DETECTOR_WINDOW, _detect_repeated_call
from agent.mcp_client import AGENT_ALLOWED_TOOLS
from agent.runner import (
    CITATION_PATTERN,
    _SourceCollector,
    _parse_tool_result,
    _validate_citations,
)


class TestParseToolResult:
    """Tool-result content can arrive as dict, JSON string, or content-block list."""

    def test_dict_passthrough(self):
        assert _parse_tool_result({"a": 1}) == {"a": 1}

    def test_json_string_parsed(self):
        assert _parse_tool_result('{"answer": "x"}') == {"answer": "x"}

    def test_non_json_string_returned_as_is(self):
        assert _parse_tool_result("not json") == "not json"

    def test_content_block_list_parsed(self):
        # langchain-mcp-adapters wraps tool results in [{type:text, text:<json>}]
        content = [{"type": "text", "text": '{"answer": "x", "sources": []}', "id": "abc"}]
        assert _parse_tool_result(content) == {"answer": "x", "sources": []}

    def test_content_block_list_concatenates_multiple_text_blocks(self):
        content = [
            {"type": "text", "text": '{"answer":"'},
            {"type": "text", "text": 'split json"}'},
        ]
        assert _parse_tool_result(content) == {"answer": "split json"}

    def test_content_block_list_with_unparseable_text_returns_string(self):
        content = [{"type": "text", "text": "not json", "id": "abc"}]
        assert _parse_tool_result(content) == "not json"

    def test_content_block_list_without_text_blocks_returns_raw(self):
        content = [{"type": "image", "url": "..."}]
        assert _parse_tool_result(content) == content

    def test_non_string_text_field_coerced(self):
        # Defensive: text might not be a string in pathological inputs.
        # The str() guard ensures we don't crash; the resulting JSON-decoded
        # value (or raw string) is whatever falls out, but importantly no
        # exception escapes.
        content = [{"type": "text", "text": 42}]
        result = _parse_tool_result(content)  # must not raise
        # 42 happens to be valid JSON for an integer; that's fine.
        assert result == 42

    def test_non_string_text_field_with_garbage(self):
        # Truly weird input: ensure it doesn't crash, returns something.
        content = [{"type": "text", "text": ["nested", "list"]}]
        # Should not raise
        _parse_tool_result(content)


class TestSourceCollector:
    """Dedup is by article identity (law_id, article) when available, otherwise
    by (source_file, snippet_prefix). Never by rank, which is per-call."""

    def test_simple_collection(self):
        c = _SourceCollector()
        c.absorb({"sources": [{"rank": 1, "source": "a.json", "snippet": "x" * 200}]})
        assert len(c.sources) == 1

    def test_dedup_across_calls_same_article_different_rank(self):
        c = _SourceCollector()
        # Same article retrieved twice, ranked differently each time —
        # should NOT be duplicated. This is finding G from the review.
        c.absorb({"sources": [{"rank": 1, "law_id": "L", "article": "Чл. 1", "snippet": "x"}]})
        c.absorb({"sources": [{"rank": 3, "law_id": "L", "article": "Чл. 1", "snippet": "x"}]})
        assert len(c.sources) == 1

    def test_dedup_falls_back_to_snippet_when_no_ids(self):
        # Old MCP payloads (pre-law_id/article fields) still dedup correctly
        c = _SourceCollector()
        c.absorb({"sources": [{"rank": 1, "source": "a.json", "snippet": "x" * 200}]})
        c.absorb({"sources": [{"rank": 3, "source": "a.json", "snippet": "x" * 200}]})
        assert len(c.sources) == 1

    def test_different_articles_kept(self):
        c = _SourceCollector()
        c.absorb({"sources": [{"rank": 1, "law_id": "L", "article": "Чл. 1", "snippet": "X"}]})
        c.absorb({"sources": [{"rank": 1, "law_id": "L", "article": "Чл. 2", "snippet": "Y"}]})
        assert len(c.sources) == 2

    def test_recursion_into_batch_results(self):
        # batch_query_tool payload nests sub-results; collector must descend
        payload = {
            "results": [
                {"sources": [{"rank": 1, "law_id": "L1", "article": "Чл. 1", "snippet": "A"}]},
                {"sources": [{"rank": 1, "law_id": "L2", "article": "Чл. 1", "snippet": "B"}]},
            ]
        }
        c = _SourceCollector()
        c.absorb(payload)
        assert len(c.sources) == 2

    def test_non_dict_source_skipped(self):
        c = _SourceCollector()
        c.absorb({"sources": ["not a dict", {"law_id": "L", "article": "Чл. 1", "snippet": "x"}]})
        assert len(c.sources) == 1

    def test_partial_id_law_only_uses_id_key(self):
        # If only law_id is present (article missing), still key on the
        # id-pair — not on snippet. Legal text often shares boilerplate
        # opening clauses, so two different articles with empty article
        # field would collapse if we fell back to snippet.
        c = _SourceCollector()
        c.absorb({"sources": [
            {"law_id": "L", "article": "", "snippet": "boilerplate intro"},
            # Different snippet with same law_id+article — should be a
            # duplicate under the id-pair key (both have article="").
            {"law_id": "L", "article": "", "snippet": "different content"},
        ]})
        assert len(c.sources) == 1

    def test_partial_id_article_only(self):
        c = _SourceCollector()
        c.absorb({"sources": [
            {"law_id": "", "article": "Чл. 1", "snippet": "boilerplate"},
            {"law_id": "", "article": "Чл. 1", "snippet": "elsewhere"},
        ]})
        assert len(c.sources) == 1

    def test_no_ids_falls_back_to_snippet(self):
        # Both fields empty — must fall back to snippet to dedup.
        c = _SourceCollector()
        c.absorb({"sources": [
            {"law_id": "", "article": "", "source": "a.json", "snippet": "x" * 200},
            {"law_id": "", "article": "", "source": "a.json", "snippet": "x" * 200},
        ]})
        assert len(c.sources) == 1


class TestValidateCitations:
    """Citation passthrough metric: preserved / dropped / invented."""

    def _tool_message(self, answer: str) -> ToolMessage:
        return ToolMessage(
            tool_call_id="t1",
            name="query_rag_tool",
            content=json.dumps({"answer": answer, "sources": []}),
        )

    def test_all_preserved(self):
        msgs = [self._tool_message("Some text [Чл. 155, Кодекс на труда].")]
        v = _validate_citations(msgs, "Final [Чл. 155, Кодекс на труда].")
        assert v["preserved"] == ["[Чл. 155, Кодекс на труда]"]
        assert v["dropped"] == []
        assert v["invented"] == []

    def test_dropped_citation(self):
        msgs = [self._tool_message("[Чл. 155, Кодекс на труда] [Чл. 156, Кодекс на труда]")]
        v = _validate_citations(msgs, "Final mentions [Чл. 155, Кодекс на труда] only.")
        assert v["dropped"] == ["[Чл. 156, Кодекс на труда]"]

    def test_invented_citation(self):
        msgs = [self._tool_message("[Чл. 155, Кодекс на труда]")]
        v = _validate_citations(msgs, "Final invents [Чл. 999, Fake].")
        assert "[Чл. 999, Fake]" in v["invented"]

    def test_batch_results_walked(self):
        # citation in a sub-result of batch_query_tool should be visible
        payload = {
            "results": [
                {"answer": "[Чл. 50, Закон за защита на потребителите]", "sources": []},
                {"answer": "[Чл. 1, Кодекс на труда]", "sources": []},
            ]
        }
        msgs = [
            ToolMessage(tool_call_id="t1", name="batch_query_tool", content=json.dumps(payload))
        ]
        v = _validate_citations(msgs, "Final [Чл. 50, Закон за защита на потребителите]")
        assert "[Чл. 50, Закон за защита на потребителите]" in v["preserved"]
        assert "[Чл. 1, Кодекс на труда]" in v["dropped"]

    def test_no_messages_no_citations(self):
        v = _validate_citations([], "")
        assert v == {
            "sub_answer_citations": [],
            "final_answer_citations": [],
            "preserved": [],
            "dropped": [],
            "invented": [],
        }


class TestCitationPattern:
    """The regex must match real citations the agent emits — not hand-waved."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("[Чл. 155, Кодекс на труда]", ["[Чл. 155, Кодекс на труда]"]),
            (
                "[Чл. 155, ал. 4, Кодекс на труда]",
                ["[Чл. 155, ал. 4, Кодекс на труда]"],
            ),
            (
                "Two: [Чл. 1, X] and [Чл. 2, Y]",
                ["[Чл. 1, X]", "[Чл. 2, Y]"],
            ),
            ("No citation here", []),
            # Don't match generic brackets
            ("[note: see article]", []),
        ],
    )
    def test_extraction(self, text, expected):
        assert CITATION_PATTERN.findall(text) == expected


class TestRepeatedCallDetector:
    """Loop detector: same (tool, args) called twice within the window → detected."""

    def _ai(self, name: str, args: dict, call_id: str = "x") -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"id": call_id, "name": name, "args": args}],
        )

    def _tm(self, call_id: str = "x") -> ToolMessage:
        return ToolMessage(tool_call_id=call_id, name="query_rag_tool", content="{}")

    def test_no_repeat_returns_none(self):
        msgs = [
            HumanMessage(content="q"),
            self._ai("query_rag_tool", {"question": "A"}, "1"),
            self._tm("1"),
            self._ai("query_rag_tool", {"question": "B"}, "2"),
        ]
        assert _detect_repeated_call(msgs) is None

    def test_same_call_twice_detected(self):
        msgs = [
            HumanMessage(content="q"),
            self._ai("query_rag_tool", {"question": "A"}, "1"),
            self._tm("1"),
            self._ai("query_rag_tool", {"question": "A"}, "2"),
        ]
        result = _detect_repeated_call(msgs)
        assert result is not None
        assert result[0] == "query_rag_tool"

    def test_arg_dict_order_irrelevant(self):
        # JSON sort_keys=True — semantically equal args must compare equal
        msgs = [
            self._ai("query_rag_tool", {"a": 1, "b": 2}, "1"),
            self._tm("1"),
            self._ai("query_rag_tool", {"b": 2, "a": 1}, "2"),
        ]
        assert _detect_repeated_call(msgs) is not None

    def test_different_tools_with_same_args_not_a_loop(self):
        msgs = [
            self._ai("query_rag_tool", {"q": "x"}, "1"),
            self._tm("1"),
            self._ai("batch_query_tool", {"q": "x"}, "2"),
        ]
        assert _detect_repeated_call(msgs) is None

    def test_window_limits_lookback(self):
        # Build a sequence longer than LOOP_DETECTOR_WINDOW with the duplicate
        # at the very beginning — should NOT be detected because window cuts off.
        msgs: list = [self._ai("query_rag_tool", {"q": "OLD"}, "0"), self._tm("0")]
        for i in range(LOOP_DETECTOR_WINDOW + 2):
            msgs.append(self._ai("query_rag_tool", {"q": f"new-{i}"}, f"i{i}"))
            msgs.append(self._tm(f"i{i}"))
        # The "OLD" call is now beyond the window; not a repeat
        assert _detect_repeated_call(msgs) is None


class TestAgentAllowedTools:
    """Decision D: only safe tools are visible to the agent."""

    def test_allowed_set(self):
        assert AGENT_ALLOWED_TOOLS == frozenset({"query_rag_tool", "batch_query_tool"})

    def test_debug_tools_not_allowed(self):
        assert "retrieve_documents" not in AGENT_ALLOWED_TOOLS
        assert "check_rag_health" not in AGENT_ALLOWED_TOOLS
