"""Top-level entrypoint for the agent.

Exposes:

- ``run_agent(question)`` — non-streaming. Returns
  ``{question, answer, sources, trace, tool_calls_used, citation_validation}``.
- ``stream_agent(question)`` — async generator yielding SSE-friendly events
  (see schema in the function docstring).

The trace is a flat list of structured events suitable for thesis-style
inspection. Each tool call and tool result becomes one entry; the agent's
textual reasoning between them is captured as a ``thought`` entry.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from citations import (
    CITATION_PATTERN,
    extract_citation_markers,
)

from .graph import get_graph

logger = logging.getLogger(__name__)


# Re-export for backward compatibility — older imports of CITATION_PATTERN
# from this module continue to work; the canonical home is ``citations``.
__all__ = ["CITATION_PATTERN", "run_agent", "stream_agent"]


def _parse_tool_result(content: Any) -> dict | list | str:
    """Tool message content arrives in one of three shapes from the MCP layer.

    - A JSON string (FastMCP older versions / direct serialization).
    - A list of content blocks ``[{"type": "text", "text": "<JSON>"}]`` —
      this is how ``langchain-mcp-adapters`` wraps tool results in current
      versions.
    - Already a dict / list — defensive case.

    Returns the deepest parseable JSON value, or the raw value if not JSON.
    """
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        # Content-block list: concatenate text blocks and try to JSON-parse the result.
        texts = [
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        if texts:
            joined = "".join(texts)
            try:
                return json.loads(joined)
            except json.JSONDecodeError:
                return joined
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return str(content)


class _SourceCollector:
    """Accumulates sources across tool results, deduplicated by article identity.

    Article identity is ``(law_id, article)`` — both fields together. If
    EITHER field is non-empty, we key on the pair (treating empty
    strings as a valid value); only when BOTH are empty do we fall back
    to ``(source_file, snippet_prefix)``.

    This preserves cross-call dedup of the same article (the common
    case) while not collapsing distinct articles whose first 80 chars
    happen to match — a real risk in legal text where articles often
    open with the same boilerplate clause.
    """

    def __init__(self) -> None:
        self.sources: list[dict] = []
        self._seen: set[tuple] = set()

    def absorb(self, payload: Any) -> None:
        if isinstance(payload, dict):
            for src in payload.get("sources", []) or []:
                if not isinstance(src, dict):
                    continue
                law_id = src.get("law_id") or ""
                article = src.get("article") or ""
                if law_id or article:
                    key: tuple = ("id", law_id, article)
                else:
                    key = ("snippet", src.get("source"), src.get("snippet", "")[:80])
                if key in self._seen:
                    continue
                self._seen.add(key)
                self.sources.append(src)
            # batch_query_tool payload nests per-question results.
            for item in payload.get("results", []) or []:
                self.absorb(item)


def _collect_sub_answer_citations(messages: list) -> set[str]:
    """Return the set of `[Чл. ...]` citation markers present in any
    ``query_rag_tool`` / ``batch_query_tool`` sub-answer.

    Used as the "ground truth" set against which the final answer's
    citations are compared.
    """
    cites: set[str] = set()

    def _walk(payload: Any) -> None:
        if isinstance(payload, dict):
            ans = payload.get("answer")
            if isinstance(ans, str):
                cites.update(extract_citation_markers(ans))
            for item in payload.get("results", []) or []:
                _walk(item)

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        _walk(_parse_tool_result(msg.content))

    return cites


def _validate_citations(messages: list, final_answer: str) -> dict:
    """Compute citation passthrough metrics for one agent run.

    The synthesis prompt rule is "preserve [Чл. X, …] citations verbatim
    from sub-answers." We don't enforce — we record. The thesis
    evaluation harness can aggregate these to report a citation-
    preservation rate across runs.

    Returns:
        {
          "sub_answer_citations": list[str],   # citations seen in tool results
          "final_answer_citations": list[str], # citations in the final answer
          "preserved": list[str],              # intersection
          "dropped": list[str],                # in sub-answers but not final
          "invented": list[str],               # in final but not in sub-answers
        }
    """
    sub = _collect_sub_answer_citations(messages)
    final = set(extract_citation_markers(final_answer or ""))
    return {
        "sub_answer_citations": sorted(sub),
        "final_answer_citations": sorted(final),
        "preserved": sorted(sub & final),
        "dropped": sorted(sub - final),
        "invented": sorted(final - sub),
    }


def _extract_sources_and_trace(messages: list) -> tuple[list[dict], list[dict]]:
    """Walk the conversation and pull out (sources, trace).

    Trace events:
      - ``{"type": "thought", "content": ...}``       agent's reasoning text
      - ``{"type": "tool_call", "id", "name", "args"}``
      - ``{"type": "tool_result", "tool_call_id", "name", "content"}``
      - ``{"type": "answer", "content"}``             final assistant message
    """
    trace: list[dict] = []
    collector = _SourceCollector()

    last_ai_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage) and not messages[i].tool_calls:
            last_ai_idx = i
            break

    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            continue

        if isinstance(msg, AIMessage):
            text = msg.content if isinstance(msg.content, str) else ""
            if text.strip() and i != last_ai_idx:
                trace.append({"type": "thought", "content": text.strip()})

            for call in msg.tool_calls or []:
                trace.append(
                    {
                        "type": "tool_call",
                        "id": call.get("id"),
                        "name": call.get("name"),
                        "args": call.get("args", {}),
                    }
                )

            if i == last_ai_idx and text.strip():
                trace.append({"type": "answer", "content": text.strip()})

        elif isinstance(msg, ToolMessage):
            parsed = _parse_tool_result(msg.content)
            collector.absorb(parsed)
            trace.append(
                {
                    "type": "tool_result",
                    "tool_call_id": msg.tool_call_id,
                    "name": getattr(msg, "name", None),
                    "content": parsed,
                }
            )

    return collector.sources, trace


def _final_answer(messages: list) -> str:
    """Return the text of the last assistant message without tool calls."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


async def run_agent(question: str) -> dict:
    """Run the agent on a single user question (non-streaming).

    Returns:
        ``{question, answer, sources, trace, tool_calls_used, citation_validation}``.

    ``citation_validation`` is informational — see ``_validate_citations``.
    The endpoint may pass it through or drop it as needed.
    """
    graph = await get_graph()
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "tool_calls_used": 0,
    }

    logger.info("Running agent for question=%r", question)
    final_state = await graph.ainvoke(initial_state)

    messages = final_state["messages"]
    sources, trace = _extract_sources_and_trace(messages)
    answer = _final_answer(messages)
    citation_validation = _validate_citations(messages, answer)

    if citation_validation["dropped"] or citation_validation["invented"]:
        logger.info(
            "Citation diff: preserved=%d dropped=%d invented=%d",
            len(citation_validation["preserved"]),
            len(citation_validation["dropped"]),
            len(citation_validation["invented"]),
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "trace": trace,
        "tool_calls_used": final_state.get("tool_calls_used", 0),
        "citation_validation": citation_validation,
    }


async def stream_agent(question: str):
    """Stream the agent's ReAct trace + final answer as a sequence of events.

    Event schema (each event is a dict ready to be JSON-serialized):

    - ``{type: "metadata", question}``                       — first event
    - ``{type: "thought", content}``                         — full text of a non-final reasoning turn (one per turn, end-of-turn)
    - ``{type: "tool_call", id, name, args}``                — agent invokes a tool
    - ``{type: "tool_result", tool_call_id, name, content}`` — tool returns
    - ``{type: "chunk", content}``                           — incremental token of the FINAL answer only
    - ``{type: "done", sources, tool_calls_used, citation_validation}`` — terminal
    - ``{type: "error", message}``                           — terminal on failure

    Streaming policy:

    - Text tokens stream live as ``chunk`` events as soon as the model
      produces them — for a fast UX on the final answer, which is the
      common case (no tool calls, ``chunk`` events represent the answer
      directly).
    - If the turn ends up being a tool turn (the model also emitted tool
      calls), a ``thought`` event with the full turn text is emitted at
      end-of-turn, and the frontend SHOULD treat any preceding
      ``chunk`` events (back to the previous ``tool_result``,
      ``thought``, or ``metadata``) as a live preview of that thought —
      replace them with the thought's content.

      Concretely: ``chunk`` events that are followed by a ``thought``
      before any ``tool_call`` belong to that thought; ``chunk`` events
      followed by ``done`` (with no intervening ``thought``) are the
      final answer.
    """
    graph = await get_graph()
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "tool_calls_used": 0,
    }

    yield {"type": "metadata", "question": question}

    collector = _SourceCollector()
    tool_calls_used = 0
    final_messages: list = []  # captured at on_chain_end for citation validation

    # Buffer the running turn's text. We stream each token live as a
    # ``chunk`` event AND keep a copy in the buffer; if the turn turns out
    # to have been a reasoning turn, we emit a final ``thought`` event with
    # the full text (and the frontend collapses the preceding chunks into
    # that thought per the contract above).
    current_turn_tokens: list[str] = []

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            etype = event["event"]

            if etype == "on_chat_model_start":
                current_turn_tokens.clear()

            elif etype == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk is None:
                    continue
                text = getattr(chunk, "content", "")
                if isinstance(text, list):
                    text = "".join(
                        str(b.get("text", ""))
                        for b in text
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if text:
                    current_turn_tokens.append(text)
                    yield {"type": "chunk", "content": text}

            elif etype == "on_chat_model_end":
                output = event["data"].get("output")
                if output is None:
                    current_turn_tokens.clear()
                    continue
                tool_calls = getattr(output, "tool_calls", None) or []
                turn_text = "".join(current_turn_tokens).strip()
                current_turn_tokens.clear()

                if tool_calls:
                    # Reasoning turn — emit consolidating thought, then tool_calls.
                    # Per the contract, the frontend collapses the preceding
                    # `chunk` events into this thought.
                    if turn_text:
                        yield {"type": "thought", "content": turn_text}
                    for call in tool_calls:
                        tool_calls_used += 1
                        yield {
                            "type": "tool_call",
                            "id": call.get("id"),
                            "name": call.get("name"),
                            "args": call.get("args", {}),
                        }
                else:
                    # Final synthesis turn — tokens already streamed live as
                    # ``chunk`` events during on_chat_model_stream. Nothing
                    # to emit at end-of-turn.
                    pass

            elif etype == "on_tool_end":
                output = event["data"].get("output")
                # ``output`` is normally a ToolMessage but LangGraph batches
                # parallel tool calls into a list — handle both.
                tool_messages = output if isinstance(output, list) else [output]
                for tm in tool_messages:
                    raw_content = getattr(tm, "content", tm)
                    parsed = _parse_tool_result(raw_content)
                    collector.absorb(parsed)
                    yield {
                        "type": "tool_result",
                        "tool_call_id": getattr(tm, "tool_call_id", None),
                        "name": getattr(tm, "name", None) or event.get("name"),
                        "content": parsed,
                    }

            elif etype == "on_chain_end" and event.get("name") == "LangGraph":
                # End of the whole graph — capture final messages for
                # citation validation.
                final_state = event["data"].get("output") or {}
                final_messages = final_state.get("messages") or []

        # Citation validation needs access to the full message history; we
        # stitched together what we could from streamed events but the
        # canonical source is the final state. Fall back gracefully if
        # on_chain_end didn't fire (older LangGraph versions).
        final_answer_text = ""
        if final_messages:
            final_answer_text = _final_answer(final_messages)
        citation_validation = (
            _validate_citations(final_messages, final_answer_text)
            if final_messages
            else None
        )

        done_event: dict = {
            "type": "done",
            "sources": collector.sources,
            "tool_calls_used": tool_calls_used,
        }
        if citation_validation is not None:
            done_event["citation_validation"] = citation_validation
        yield done_event

    except Exception as exc:  # noqa: BLE001
        logger.exception("Streaming agent failed for question=%r", question)
        yield {
            "type": "error",
            "message": f"Agent run failed: {exc!r}",
        }
