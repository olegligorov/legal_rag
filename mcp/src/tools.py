from typing import Any

import httpx
from fastmcp.exceptions import ToolError

from src.client import RAGClient, RAGClientError


def _tool_error(e: Exception) -> ToolError:
    if isinstance(e, RAGClientError):
        return ToolError(f"{e.message} (status {e.status_code})")
    if isinstance(e, httpx.ConnectError):
        return ToolError("Backend unreachable")
    if isinstance(e, httpx.TimeoutException):
        return ToolError("Request timed out")
    if isinstance(e, httpx.HTTPStatusError):
        return ToolError(f"HTTP {e.response.status_code}: {e.response.text}")
    return ToolError(f"Unexpected error: {e}")


async def query_rag(
    client: RAGClient, question: str, top_n: int | None = None
) -> dict[str, Any]:
    try:
        return await client.query(question, top_n)
    except Exception as e:
        raise _tool_error(e) from e


async def batch_query(
    client: RAGClient,
    questions: list[str],
    top_n: int | None = None,
    max_concurrency: int = 5,
) -> dict[str, Any]:
    raw = await client.batch_query(questions, top_n, max_concurrency)
    results = []
    failed = 0
    for item in raw:
        if isinstance(item, Exception):
            failed += 1
            results.append({"error": str(item)})
        else:
            results.append(item)
    return {"results": results, "total": len(results), "successful": len(results) - failed, "failed": failed}


async def retrieve_documents(
    client: RAGClient, question: str, top_n: int | None = None
) -> dict[str, Any]:
    try:
        return await client.retrieve(question, top_n)
    except Exception as e:
        raise _tool_error(e) from e


async def check_rag_health(client: RAGClient) -> dict[str, Any]:
    try:
        return await client.health()
    except Exception as e:
        raise _tool_error(e) from e
