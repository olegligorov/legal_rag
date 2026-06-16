"""
FastAPI server for Plug and Play RAG System
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import run_agent, stream_agent
from agent.mcp_client import prefetch_tools
from config import DATA_PATH, RERANKER_TOP_N
from models.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline and prefetch the MCP tool list on startup.

    Loading the embedding + reranker models and building/loading the FAISS
    + BM25 indices runs once here. We also eagerly fetch the MCP tool list
    so a missing or unreachable MCP server fails the backend startup
    instead of the first /api/agent request.
    """
    print("Starting up... Initializing RAG Pipeline...")
    global rag_pipeline
    rag_pipeline = RAGPipeline(data_directory=DATA_PATH)
    print("RAG Pipeline ready!")

    try:
        tools = await prefetch_tools()
        print(f"MCP tools loaded: {[t.name for t in tools]}")
    except Exception as exc:
        # Don't crash the backend — agent endpoints will still surface 500s,
        # but /api/query keeps working without MCP.
        print(f"WARNING: MCP prefetch failed ({exc!r}); /api/agent will be unavailable.")

    yield

    print("Shutting down...")


app = FastAPI(
    title="Plug and Play RAG API",
    description="Plug and Play documentation RAG system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    top_n: int = Field(RERANKER_TOP_N, ge=1, le=20, description="Number of documents to retrieve")


class SourceResponse(BaseModel):
    rank: int
    source: str
    snippet: str
    score: float | None = None
    # Article-level identity, useful for evaluation harnesses that need to
    # match retrieved articles against an expected_articles list. Older
    # consumers of the API ignore the new fields.
    law_id: str = ""
    article: str = ""


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceResponse]


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")


class AgentResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceResponse]
    # Trace event types are documented in backend/agent/runner.py:stream_agent.
    trace: list[dict]
    tool_calls_used: int
    # Citation passthrough diff (preserved/dropped/invented) — informational,
    # not enforced. See _validate_citations in agent/runner.py.
    citation_validation: dict | None = None


@app.get("/api/health")
def health_check():
    """
    Health check endpoint to verify server and pipeline status.
    """
    return {
        "status": "ok",
        "message": "server is running",
        "pipeline_loaded": rag_pipeline is not None,
    }


@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Query the RAG system with a question.
    Returns LLM-generated answer with sources.
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Please wait for startup to complete.",
        )

    try:
        result = rag_pipeline.query(query=request.question, top_n=request.top_n)

        sources = [
            SourceResponse(
                rank=source["rank"],
                source=source["source"],
                snippet=source["snippet"],
                score=source.get("score"),
                law_id=source.get("law_id", ""),
                article=source.get("article", ""),
            )
            for source in result["sources"]
        ]

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources,
        )

    except Exception:
        logger.exception("Query processing failed for question=%r", request.question)
        raise HTTPException(status_code=500, detail="Query processing failed")


@app.post("/api/query/stream")
def query_stream_endpoint(request: QueryRequest):
    """
    Query the RAG system with streaming response.
    Returns LLM-generated answer as Server-Sent Events (SSE).

    The response stream format:
    1. First event: metadata with sources (JSON)
    2. Subsequent events: answer chunks (text)
    3. Final event: [DONE] marker
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Please wait for startup to complete.",
        )

    def generate():
        try:
            sources, answer_stream = rag_pipeline.query_stream(
                query=request.question, top_n=request.top_n
            )

            metadata = {"type": "metadata", "sources": sources, "question": request.question}

            # \n\n -> According to the SSE specification: Events are separated by blank lines (two consecutive newline characters)
            yield f"data: {json.dumps(metadata)}\n\n"

            # Stream answer chunks
            for chunk in answer_stream:
                if chunk:  # Only send non-empty chunks
                    chunk_data = {"type": "chunk", "content": chunk}
                    yield f"data: {json.dumps(chunk_data)}\n\n"

            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception:
            logger.exception("Streaming query failed for question=%r", request.question)
            error_data = {"type": "error", "message": "Streaming query failed"}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )


@app.post("/api/agent", response_model=AgentResponse)
async def agent_endpoint(request: AgentRequest):
    """Run the multi-hop ReAct agent against the local MCP server.

    The agent decomposes the question, calls RAG tools sequentially over
    MCP, and composes a final cited answer. Slower than `/api/query` but
    handles questions that need evidence from multiple articles.

    Requires the FastMCP server to be running and reachable at
    ``MCP_SERVER_URL`` (default ``http://127.0.0.1:8001/sse``).
    """
    try:
        result = await run_agent(question=request.question)
    except Exception:
        logger.exception("Agent run failed for question=%r", request.question)
        raise HTTPException(status_code=500, detail="Agent run failed")

    sources = [
        SourceResponse(
            rank=src.get("rank", 0),
            source=src.get("source", "unknown"),
            snippet=src.get("snippet", ""),
            score=src.get("score"),
            law_id=src.get("law_id", ""),
            article=src.get("article", ""),
        )
        for src in result["sources"]
    ]

    return AgentResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        trace=result["trace"],
        tool_calls_used=result["tool_calls_used"],
        citation_validation=result.get("citation_validation"),
    )


@app.post("/api/agent/stream")
async def agent_stream_endpoint(request: AgentRequest):
    """Stream the agent's ReAct trace and final answer as Server-Sent Events.

    Event types (each `data: <json>\\n\\n`):
      - `metadata` — first event, echoes the question.
      - `thought` — full text of a non-final reasoning turn.
      - `tool_call` — `{id, name, args}`.
      - `tool_result` — `{tool_call_id, name, content}` (content is the parsed MCP payload).
      - `chunk` — incremental token of the final answer.
      - `done` — `{sources, tool_calls_used}`. Terminal.
      - `error` — `{message}`. Terminal on failure.
    """

    async def generate():
        # ``stream_agent`` already catches its own exceptions and yields a
        # terminal ``error`` event, so an outer try/except here would be
        # unreachable. The only thing we add at this layer is the SSE
        # framing.
        async for event in stream_agent(request.question):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
