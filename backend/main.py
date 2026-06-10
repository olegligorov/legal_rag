"""
FastAPI server for Plug and Play RAG System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List
import json
import os
from pathlib import Path
import markdown

from config import DATA_PATH, RERANK_TOP_N
from models.rag_pipeline import RAGPipeline

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    top_n: int = Field(RERANK_TOP_N, ge=1, le=20, description="Number of documents to retrieve")


class SourceResponse(BaseModel):
    rank: int
    source: str
    snippet: str
    score: float = None


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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


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
            retrieved_docs, scores, answer_stream = rag_pipeline.query_stream(
                query=request.question, top_n=request.top_n
            )

            sources = [
                {
                    "rank": idx + 1,
                    "source": doc.metadata.get("source", "unknown"),
                    # "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    "snippet": doc.page_content,
                    "score": round(scores[idx], 4) if idx < len(scores) else None,
                }
                for idx, doc in enumerate(retrieved_docs)
            ]

            # First send the sources
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

        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
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


@app.get("/api/file")
def get_file(path: str):
    """
    Serve a markdown file from the k8s_data directory as HTML.
    """
    try:
        project_root = Path(__file__).parent.parent
        absolute_path = Path(path).resolve()

        if not absolute_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if not str(absolute_path).startswith(str(project_root)):
            raise HTTPException(status_code=403, detail="Access denied")

        with open(absolute_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Convert markdown to HTML
        html_content = markdown.markdown(content, extensions=["extra", "codehilite", "toc"])

        # Wrap in a styled HTML template matching the UI's dark theme
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{absolute_path.name}</title>
            <style>
                body {{
                    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
                    line-height: 1.6;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 2rem;
                    background: oklch(0.145 0 0);
                    color: oklch(0.985 0 0);
                }}
                h1, h2, h3, h4, h5, h6 {{
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                    font-weight: 600;
                    color: oklch(0.985 0 0);
                }}
                h1 {{ font-size: 2em; border-bottom: 2px solid oklch(1 0 0 / 10%); padding-bottom: 0.3em; }}
                h2 {{ font-size: 1.5em; border-bottom: 1px solid oklch(1 0 0 / 10%); padding-bottom: 0.3em; }}
                code {{
                    background: oklch(0.205 0 0);
                    padding: 2px 6px;
                    border-radius: 0.5rem;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
                    font-size: 0.9em;
                    color: oklch(0.985 0 0);
                }}
                pre {{
                    background: oklch(0.205 0 0);
                    padding: 1em;
                    border-radius: 0.5rem;
                    overflow-x: auto;
                    border: 1px solid oklch(1 0 0 / 10%);
                }}
                pre code {{
                    background: none;
                    padding: 0;
                }}
                a {{
                    color: oklch(0.488 0.243 264.376);
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                blockquote {{
                    border-left: 4px solid oklch(1 0 0 / 10%);
                    padding-left: 1em;
                    color: oklch(0.708 0 0);
                    margin: 1em 0;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1em 0;
                }}
                th, td {{
                    border: 1px solid oklch(1 0 0 / 10%);
                    padding: 8px 12px;
                    text-align: left;
                }}
                th {{
                    background: oklch(0.205 0 0);
                    font-weight: 600;
                }}
                .file-path {{
                    background: oklch(0.205 0 0);
                    padding: 0.5em 1em;
                    border-radius: 0.5rem;
                    margin-bottom: 1em;
                    font-size: 0.9em;
                    color: oklch(0.708 0 0);
                    word-break: break-all;
                    border: 1px solid oklch(1 0 0 / 10%);
                }}
            </style>
        </head>
        <body>
            <div class="file-path">📄 {absolute_path}</div>
            {html_content}
        </body>
        </html>
        """

        return HTMLResponse(content=styled_html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
