from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import structlog
from fastmcp import Context, FastMCP
from pydantic_settings import BaseSettings

from src import tools
from src.client import RAGClient

logger = structlog.get_logger(__name__)

class MCPSettings(BaseSettings):
    query_url: str = "http://localhost:8000"
    api_timeout: float = 30.0
    batch_max_concurrency: int = 5
    server_name: str = "mcp"
    server_version: str = "0.1.0"

    model_config = {"env_prefix": "RAG_"}

@dataclass
class AppContext:
    client: RAGClient

def _get_client(ctx: Context) -> RAGClient:
    return ctx.request_context.lifespan_context.client

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    client = RAGClient(
        base_url=settings.query_url,
        timeout=settings.api_timeout,
    )
    async with client:
        yield AppContext(client=client)


settings = MCPSettings()    

mcp = FastMCP(settings.server_name, version=settings.server_version, lifespan=app_lifespan)

@mcp.tool()
async def retrieve_documents(
    question: str,
    top_n: int | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Return ranked source articles for a question without generating an answer.

    Use this when you need to inspect which law articles are relevant before
    deciding how to answer, or when the agent will synthesize the answer itself.
    Returns: question, sources (list of ranked articles with rank, source, snippet, score).
    """
    return await tools.retrieve_documents(_get_client(ctx), question, top_n)


@mcp.tool()
async def query_rag_tool(
    question: str,
    top_n: int | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Answer a question about Bulgarian law using the RAG system.

    Searches the Labour Code (Кодекс на труда), Consumer Protection Act
    (Закон за защита на потребителите), and Obligations and Contracts Act
    (Закон за задълженията и договорите). Returns an LLM-generated answer
    with citations. Ask in Bulgarian or English; the answer will match the
    question language.

    Use this for a single legal question. For multiple related questions,
    prefer batch_query_tool.

    Returns: question, answer (string), sources (list of cited articles).
    """
    return await tools.query_rag(_get_client(ctx), question, top_n)


@mcp.tool()
async def batch_query_tool(
    questions: list[str],
    top_n: int | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Answer multiple legal questions in parallel.

    Use this instead of calling query_rag_tool repeatedly when you have 2 or
    more independent questions. Runs all questions concurrently and returns
    partial results even if some fail.

    Returns: results (list, one entry per question), total, successful, failed.
    Each result is either a full query_rag_tool response or {"error": "..."}.
    """
    return await tools.batch_query(_get_client(ctx), questions, top_n, settings.batch_max_concurrency)


@mcp.tool()
async def check_rag_health(ctx: Context = None) -> dict[str, Any]:
    """Check whether the RAG backend is reachable and the pipeline is loaded.

    Returns: status ("ok"), message, pipeline_loaded (bool).
    Call this before other tools if you are unsure the backend is running.
    """
    return await tools.check_rag_health(_get_client(ctx))

if __name__ == "__main__":
    mcp.run()