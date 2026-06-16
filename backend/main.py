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

from config import DATA_PATH, RERANKER_TOP_N
from models.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the RAG pipeline on server startup.
    Loads all models and builds indices once.
    """
    print("Starting up... Initializing RAG Pipeline...")
    global rag_pipeline
    rag_pipeline = RAGPipeline(data_directory=DATA_PATH)
    print("RAG Pipeline ready!")

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


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceResponse]


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
