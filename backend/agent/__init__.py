"""LangGraph ReAct agent for multi-hop legal QA over MCP.

The agent decomposes a user question into sequential sub-queries, calls the
RAG tools exposed by the MCP server, and synthesizes a final cited answer.
"""

from .runner import run_agent, stream_agent

__all__ = ["run_agent", "stream_agent"]
