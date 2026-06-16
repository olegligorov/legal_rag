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
async def query_rag_tool(
    question: str,
    top_n: int | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Ask one focused legal question and get a grounded answer with citations.

    The RAG searches three Bulgarian statutes — Кодекс на труда (Labour Code),
    Закон за защита на потребителите (Consumer Protection Act), and Закон за
    задълженията и договорите (Obligations and Contracts Act) — and returns an
    LLM answer cited inline as `[Чл. X, <law_id>]`. Phrase `question` as a
    single, focused legal question in Bulgarian or English; the answer comes
    back in the question's language.

    This is the workhorse for sequential reasoning: call once, observe the
    answer + citations, then decide the next sub-question based on what you
    learned. Prefer this for the next step when its sub-question DEPENDS on
    the result of the previous step. Use `batch_query_tool` only when you can
    name 2+ INDEPENDENT sub-questions whose answers don't inform each other.

    If the answer string starts with `INSUFFICIENT_CONTEXT:`, the retrieval
    came up short. Do not treat this as a final answer. Either reformulate
    the sub-question (more specific terms, different angle, the law's name in
    Bulgarian) and call again, or pivot to a different sub-question. Do this
    at most once per failed sub-question — repeated INSUFFICIENT_CONTEXT on
    the same topic means the indexed laws genuinely don't cover it; record
    that gap and continue.

    Returns: {question, answer, sources}. Preserve every `[Чл. X, <law_id>]`
    citation verbatim into the final synthesis — never rewrite or drop them.
    """
    return await tools.query_rag(_get_client(ctx), question, top_n)


@mcp.tool()
async def batch_query_tool(
    questions: list[str],
    top_n: int | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Answer 2+ INDEPENDENT legal sub-questions in parallel.

    Use ONLY when each question can be answered without knowing the others'
    answers. Example fits: "What is the notice period under Кодекс на труда?"
    AND "What is the 14-day withdrawal right under Закон за защита на
    потребителите?" — different statutes, no dependency.

    Do NOT use when a later question's wording depends on an earlier
    question's answer (e.g., "What does Чл. X say?" then "How does that
    interact with Y?"). For dependent chains, call `query_rag_tool` once per
    step so each step can react to the previous result.

    Each item in `questions` is processed independently, like a separate
    `query_rag_tool` call. Failed items become `{"error": "..."}` in the
    results array; partial success is normal — the call itself does not fail.

    Returns: {results, total, successful, failed}. Apply the same
    INSUFFICIENT_CONTEXT and citation-preservation rules from
    `query_rag_tool` to every successful item.
    """
    return await tools.batch_query(_get_client(ctx), questions, top_n, settings.batch_max_concurrency)


@mcp.tool()
async def retrieve_documents(
    question: str,
    top_n: int | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Return ranked source snippets without generating an answer.

    Use this only when you specifically need to see WHICH articles the
    retriever is finding for a query — for example to debug a confusing
    answer, to confirm an article number is in the index, or to check the
    relevance scores before deciding how to phrase the next sub-question.

    For normal question answering, prefer `query_rag_tool` — it does the
    same retrieval and gives you a grounded answer with citations. Calling
    `retrieve_documents` and then synthesizing yourself bypasses the
    grounding prompt and risks paraphrasing the law incorrectly.

    Returns: {question, sources} where each source has rank, source, snippet,
    score. Snippets are excerpts, not the full article text.
    """
    return await tools.retrieve_documents(_get_client(ctx), question, top_n)


@mcp.tool()
async def check_rag_health(ctx: Context = None) -> dict[str, Any]:
    """Check whether the RAG backend is reachable.

    Diagnostic only. Do not call as part of normal answering — go straight to
    `query_rag_tool`. Use this only if a previous tool call failed with a
    backend/connection error and you need to confirm the service is up
    before retrying.

    Returns: {status, message, pipeline_loaded}.
    """
    return await tools.check_rag_health(_get_client(ctx))

if __name__ == "__main__":
    mcp.run()