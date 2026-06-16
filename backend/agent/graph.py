"""LangGraph state graph for the multi-hop legal RAG agent.

Topology::

    START
      ▼
    call_model ──(no tool calls)─────────────────────────► END
      │
      │(tool calls)
      ▼
    after_model ──(projected usage > cap)──► force_synthesize ──► END
      │                                          ▲
      │(under budget)                            │
      ▼                                          │
    tools ──► tick ──(repeated call detected)────┤
                │                                │
                └──(continue)──► call_model      │
                                                 │
The cap is applied on the projected usage *before* invoking tools, so the
agent never gets to execute a turn that would push it over budget. ``tick``
runs after ``tools`` and increments ``tool_calls_used`` by the number of
calls that were actually executed; it also short-circuits to
``force_synthesize`` if the same (tool, args) pair has been called more
than once recently — a cheap loop detector.
"""

from __future__ import annotations

import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from config import (
    AGENT_MAX_TOOL_CALLS,
    AGENT_MODEL,
    CLAUDE_API_KEY,
    CLAUDE_URL,
    LLM_TEMPERATURE,
)

from .mcp_client import get_tools
from .prompts import FORCE_SYNTHESIS_PROMPT, REACT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# How many recent tool calls to look back across when detecting loops. Two
# matches in a row of the same (tool, args) is enough to abort — the prompt
# already permits one reformulation per failed sub-question.
LOOP_DETECTOR_WINDOW = 4


class AgentState(MessagesState):
    """Conversation state plus a tool-call counter."""

    tool_calls_used: int


def _build_llm() -> ChatAnthropic:
    """Build the Claude client used for both reasoning and synthesis."""
    return ChatAnthropic(
        model=AGENT_MODEL,
        base_url=f"{CLAUDE_URL}/anthropic",
        api_key=CLAUDE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=2048,
    )


def _trailing_tool_messages(messages: list) -> list[ToolMessage]:
    """Return the contiguous trailing block of ToolMessages."""
    out: list[ToolMessage] = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            out.append(msg)
        else:
            break
    return list(reversed(out))


def _detect_repeated_call(messages: list) -> tuple[str, str] | None:
    """Look back over recent assistant tool calls and report the first
    (tool, args) pair that appears more than once.

    We collect tool_calls from AIMessages within ``LOOP_DETECTOR_WINDOW``
    of the tail and treat a duplicate as a loop. Args are stable-serialized
    so dict ordering doesn't cause false negatives.
    """
    seen: dict[tuple[str, str], int] = {}
    looked = 0
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        for call in msg.tool_calls or []:
            name = call.get("name", "")
            args_key = json.dumps(call.get("args", {}), sort_keys=True, ensure_ascii=False)
            key = (name, args_key)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                return key
        looked += 1
        if looked >= LOOP_DETECTOR_WINDOW:
            break
    return None


_compiled_graph = None


async def get_graph():
    """Build and cache the compiled graph on first use.

    No lock here: two concurrent first-callers may both compile and the
    later assignment wins. Compiling the graph is pure (no side effects),
    so a duplicated build is wasted work but not a correctness issue.
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    tools = await get_tools()
    llm = _build_llm()
    llm_with_tools = llm.bind_tools(tools)
    llm_no_tools = llm  # used for force_synthesize

    react_message = SystemMessage(
        content=REACT_SYSTEM_PROMPT.format(max_tool_calls=AGENT_MAX_TOOL_CALLS)
    )
    # force_synthesize replaces (does not extend) the ReAct system prompt:
    # the budget-exhausted instructions override loop guidance entirely.
    force_synthesis_message = SystemMessage(content=FORCE_SYNTHESIS_PROMPT)

    async def call_model(state: AgentState) -> dict:
        messages = [react_message, *state["messages"]]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def force_synthesize(state: AgentState) -> dict:
        # The conversation may end on an AIMessage with unanswered tool_calls
        # (we routed here precisely because the agent tried to call more tools
        # past the budget). Anthropic's API rejects an unanswered tool_use, so
        # we synthesize a "budget exhausted" tool_result for each pending call
        # before asking the model to compose the final answer without tools.
        last = state["messages"][-1]
        budget_messages: list = []
        if isinstance(last, AIMessage) and last.tool_calls:
            budget_messages = [
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call.get("name", "unknown"),
                    content="ERROR: tool-call budget exhausted; this tool was not executed.",
                )
                for call in last.tool_calls
            ]

        messages = [force_synthesis_message, *state["messages"], *budget_messages]
        response = await llm_no_tools.ainvoke(messages)
        return {"messages": [*budget_messages, response]}

    def tick(state: AgentState) -> dict:
        """Count tool calls just executed by the ``tools`` node.

        The number of trailing ToolMessages == the number of tool calls in
        the AIMessage immediately before them.
        """
        executed = len(_trailing_tool_messages(state["messages"]))
        if not executed:
            return {}
        return {"tool_calls_used": state.get("tool_calls_used", 0) + executed}

    def after_model(state: AgentState) -> str:
        """Route after the model speaks, projecting the budget."""
        last = state["messages"][-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return END

        used = state.get("tool_calls_used", 0)
        proposed = len(last.tool_calls)
        if used + proposed > AGENT_MAX_TOOL_CALLS:
            logger.info(
                "Projected usage %d + %d exceeds cap %d. Forcing synthesis.",
                used,
                proposed,
                AGENT_MAX_TOOL_CALLS,
            )
            return "force_synthesize"
        return "tools"

    def after_tick(state: AgentState) -> str:
        """Route after a tool round: bail to synthesis on a detected loop."""
        repeat = _detect_repeated_call(state["messages"])
        if repeat is not None:
            tool_name, _ = repeat
            logger.info(
                "Repeated tool call detected (%s); forcing synthesis.", tool_name
            )
            return "force_synthesize"
        return "call_model"

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("tick", tick)
    builder.add_node("force_synthesize", force_synthesize)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        after_model,
        {"tools": "tools", "force_synthesize": "force_synthesize", END: END},
    )
    builder.add_edge("tools", "tick")
    builder.add_conditional_edges(
        "tick",
        after_tick,
        {"call_model": "call_model", "force_synthesize": "force_synthesize"},
    )
    builder.add_edge("force_synthesize", END)

    _compiled_graph = builder.compile()
    logger.info("Agent graph compiled (max_tool_calls=%d)", AGENT_MAX_TOOL_CALLS)
    return _compiled_graph
