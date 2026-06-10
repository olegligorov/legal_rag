from pathlib import Path
import json

from langchain_core.documents import Document

from config import (
    CACHE_DIR,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    EMBEDDING_MODEL,
    MIN_RETRIEVED_DOCS,
    RERANKER_MODEL,
    RERANKER_SCORE_THRESHOLD,
    RERANKER_TOP_N,
)
from rag.chunker import Chunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

from rag.retrieval import HybridRetriever
from rag.generation import Generator

_MANIFEST_FILE = "manifest.json"


def _build_manifest(data_directory: str) -> dict:
    """Fingerprint the inputs that affect the cached indices.

    All values must be JSON-roundtrip-stable: tuples are encoded as lists,
    so we use lists here too — otherwise a fresh manifest never compares
    equal to the cached one and the cache is rebuilt every restart.

    To force a rebuild after non-config changes (e.g. chunker logic),
    run `scripts/clear_cache.py`.
    """
    mtimes = sorted([str(p), p.stat().st_mtime] for p in Path(data_directory).glob("**/*.json"))
    return {
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
        "source_files": mtimes,
    }


class RAGPipeline:
    def __init__(self, data_directory: str | Path, use_cache: bool = True):
        self._data_directory = data_directory

        print("Initialising RAG Pipeline...")
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_cache_path = self.cache_dir / "semantic_chunks.pkl"
        self.faiss_cache_path = self.cache_dir / "faiss_index"
        self.bm25_cache_path = self.cache_dir / "bm25_retriever.pkl"
        self.manifest_path = self.cache_dir / _MANIFEST_FILE

        print("Loading embedding model...")
        self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        print("Initialising LLM Generator...")
        self._generator = Generator()

        if use_cache and self._cache_valid():
            print("Loading indices from cache...")
            try:
                self._load_from_cache()
                print("RAG Pipeline initialised successfully from cache!")
                return
            except Exception as e:
                print(f"Cache loading failed: {e}. Building indices from scratch...")

        print("Building indices from scratch...")
        raw_docs = self.load_documents(data_directory)

        chunker = Chunker()
        self.semantic_docs = chunker.create_chunks(raw_docs)

        self._retriever = HybridRetriever(
            semantic_docs=self.semantic_docs, embeddings=self._embeddings
        )

        if use_cache:
            print("Saving indices to cache...")
            self._save_to_cache()

        print("RAG Pipeline initialized successfully!")

    def load_documents(self, data_directory: str | Path) -> list[Document]:
        docs = []
        for json_path in sorted(Path(data_directory).glob("**/*.json")):
            with open(json_path, encoding="utf-8") as f:
                articles = json.load(f)
            for article in articles:
                content = article.get("content", "").strip()
                if not content:
                    continue
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(json_path.relative_to(Path(data_directory))),
                            "law_id": article.get("law_id", ""),
                            "chapter": article.get("chapter", ""),
                            "section": article.get("section", ""),
                            "article": article.get("article", ""),
                            "title": article.get("title", ""),
                        },
                    )
                )

        print(f"Loaded {len(docs)} documents from {data_directory}.")
        return docs

    def _cache_valid(self) -> bool:
        """Return True if cache files exist and match the current config + source files."""
        if not (
            self.chunks_cache_path.exists()
            and self.faiss_cache_path.exists()
            and self.bm25_cache_path.exists()
            and self.manifest_path.exists()
        ):
            return False

        try:
            with open(self.manifest_path) as f:
                cached = json.load(f)
            return cached == _build_manifest(self._data_directory)
        except Exception:
            return False

    def _save_to_cache(self) -> None:
        try:
            with open(self.chunks_cache_path, "wb") as f:
                pickle.dump(self.semantic_docs, f)

            self._retriever.save(self.faiss_cache_path, self.bm25_cache_path)

            with open(self.manifest_path, "w") as f:
                json.dump(_build_manifest(self._data_directory), f, indent=2)

            print(f"Indices cached successfully at {self.cache_dir}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _load_from_cache(self) -> None:
        with open(self.chunks_cache_path, "rb") as f:
            self.semantic_docs = pickle.load(f)

        print(f"Loaded {len(self.semantic_docs)} cached semantic chunks")

        print("Loading FAISS index from cache...")
        vector_db = FAISS.load_local(
            str(self.faiss_cache_path), self._embeddings, allow_dangerous_deserialization=True
        )

        print("Loading BM25 index from cache...")
        with open(self.bm25_cache_path, "rb") as f:
            bm25_retriever = pickle.load(f)

        self._retriever = HybridRetriever.from_indices(vector_db, bm25_retriever)

    def query(self, query: str, top_n: int = RERANKER_TOP_N) -> dict:
        retrieved_docs, scores = self._retriever.search(
            query=query,
            top_n=top_n,
            score_threshold=RERANKER_SCORE_THRESHOLD,
            min_docs=MIN_RETRIEVED_DOCS,
        )

        return self._generator.generate(
            query=query, documents=retrieved_docs, scores=scores, include_sources=True
        )

    def query_stream(self, query: str, top_n: int = RERANKER_TOP_N):
        retrieved_docs, scores = self._retriever.search(
            query=query,
            top_n=top_n,
            score_threshold=RERANKER_SCORE_THRESHOLD,
            min_docs=MIN_RETRIEVED_DOCS,
        )

        sources = self._generator.format_sources(retrieved_docs, scores)
        return sources, self._generator.generate_stream(query=query, documents=retrieved_docs)

    def query_with_contexts(self, query: str, top_n: int = RERANKER_TOP_N) -> dict:
        """Same as `query` but also returns the raw retrieved contexts.

        Shape matches what RAG evaluation harnesses (RAGAS, TruLens) expect:
        `question / answer / contexts / sources`. Use this from evaluation
        pipelines; the FastAPI endpoint uses `query` instead.
        """
        retrieved_docs, scores = self._retriever.search(
            query=query,
            top_n=top_n,
            score_threshold=RERANKER_SCORE_THRESHOLD,
            min_docs=MIN_RETRIEVED_DOCS,
        )

        result = self._generator.generate(
            query=query, documents=retrieved_docs, scores=scores, include_sources=True
        )

        return {
            "question": query,
            "answer": result["answer"],
            "contexts": [doc.page_content for doc in retrieved_docs],
            "sources": result["sources"],
        }
