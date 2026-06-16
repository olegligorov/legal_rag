"""MCP client lifecycle for the agent.

The agent reaches the RAG via the local FastMCP server. We use a single
``MultiServerMCPClient`` per process, lazily initialized on first use and
cached for the process lifetime.

The client doesn't keep a persistent socket open — ``get_tools`` returns
adapter objects that open a fresh MCP session per tool invocation. So
"singleton" here just means "one client config, one tool list, cached".

The agent only sees a curated subset of the MCP tools (``query_rag_tool``,
``batch_query_tool``). The MCP server still exposes ``retrieve_documents``
and ``check_rag_health`` for other consumers, but routing the agent through
them would bypass the grounding step — see ``AGENT_PLAN.md`` decision 5.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MCP_SERVER_URL, MCP_TRANSPORT

logger = logging.getLogger(__name__)

# Tools the agent is allowed to call. Filter is applied on top of whatever
# the MCP server advertises so that adding a new debug-only tool to the
# server does not silently widen the agent's surface.
AGENT_ALLOWED_TOOLS = frozenset({"query_rag_tool", "batch_query_tool"})

_client: MultiServerMCPClient | None = None
_tools: list[BaseTool] | None = None
_lock = asyncio.Lock()


def _build_client() -> MultiServerMCPClient:
    server_config: dict[str, Any] = {
        "transport": MCP_TRANSPORT,
        "url": MCP_SERVER_URL,
    }
    return MultiServerMCPClient({"legal_rag": server_config})


async def get_tools() -> list[BaseTool]:
    """Return the cached list of agent-visible MCP tools, fetching on first call."""
    global _client, _tools

    if _tools is not None:
        return _tools

    async with _lock:
        if _tools is not None:  # double-check under lock
            return _tools

        client = _build_client()
        logger.info("Fetching MCP tools from %s (%s)", MCP_SERVER_URL, MCP_TRANSPORT)
        try:
            all_tools = await client.get_tools()
        except Exception:
            # Don't leave a half-built client cached — next call will retry.
            logger.exception("Failed to fetch MCP tools")
            raise

        filtered = [t for t in all_tools if t.name in AGENT_ALLOWED_TOOLS]
        dropped = [t.name for t in all_tools if t.name not in AGENT_ALLOWED_TOOLS]
        if dropped:
            logger.info("Hiding non-agent MCP tools from agent: %s", dropped)

        _client = client
        _tools = filtered
        logger.info("Agent tool list: %s", [t.name for t in filtered])

    return _tools


async def reset_tools() -> None:
    """Drop the cached tool list and client.

    Call when the MCP server has been restarted or its tool surface has
    changed. The next ``get_tools`` call will fetch fresh.
    """
    global _client, _tools
    async with _lock:
        _client = None
        _tools = None
    logger.info("MCP tool cache cleared")


async def prefetch_tools() -> list[BaseTool]:
    """Eagerly fetch the tool list. Use from FastAPI ``lifespan`` so a
    missing MCP server fails the backend startup instead of the first
    /api/agent request.
    """
    return await get_tools()
