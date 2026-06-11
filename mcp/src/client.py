"""
Async HTTP client for communicating with the RAG FastAPI server.
"""
import asyncio
from typing import Any
import httpx

class RAGClientError(Exception):
    message: str
    status_code: int | None

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class RAGClient:
    """
    Async HTTP client for FastAPI RAG endpoints.

    Features:
    - Connection pooling for efficiency
    - HTTP/2 support for request multiplexing
    - Configurable timeouts
    - Proper resource cleanup via context manager
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        
    async def __aenter__(self) -> "RAGClient":
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            limits=limits,
            http2=True,
        )
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("RAGClient must be used as an async context manager")
        return self._client            

    async def query(
        self, question: str, top_n: int | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"question": question}
        if top_n is not None:
            body["top_n"] = top_n
        response: httpx.Response = await self.client.post("/api/query", json=body)
        response.raise_for_status()
        return response.json()
    
    async def health(self) -> dict[str, Any]:
        response: httpx.Response = await self.client.get("/api/health")
        response.raise_for_status()
        return response.json()

    async def retrieve(
        self, question: str, top_n: int | None = None
    ) -> dict[str, Any]:
        result = await self.query(question, top_n)
        return {"question": result["question"], "sources": result["sources"]}

    async def batch_query(
        self,
        questions: list[str],
        top_n: int | None = None,
        max_concurrency: int = 5,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _query_one(q: str) -> dict[str, Any]:
            async with semaphore:
                return await self.query(q, top_n)

        return list(await asyncio.gather(*[_query_one(q) for q in questions], return_exceptions=True))