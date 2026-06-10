from pathlib import Path
import json

from langchain_core.documents import Document

from config import (
    CACHE_DIR,
    EMBEDDING_MODEL,
    MIN_RETRIEVED_DOCS,
    RERANKER_SCORE_THRESHOLD,
    RERANKER_TOP_N,
)
from rag.chunker import Chunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

from rag.retrieval import HybridRetriever

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline orchestrator.

    This class manages the complete RAG workflow:
    1. Document loading from markdown files
    2. Semantic chunking to split documents at natural boundaries
    3. Building hybrid retrieval indices (vector + BM25)
    4. Query processing with retrieval and re-ranking
    5. LLM-based answer generation with retrieved context

    The pipeline is initialized once (typically on server startup).
    """

    def __init__(self, data_directory, use_cache=True):
        """
        Initialize the RAG Pipeline with documents loading, chunking and indexing

        1. Load embedding model
        2. Load documents
        3. Chunk documents
        4. Build retrieval indicies and save them

        Args:
            data_directory (str): Path to directory containing markdown (.md) documents.
                Will recursively load all .md files from this directory.
            use_cache (bool): If True, attempts to load cached indices. If False or cache
                doesn't exist, builds indices from scratch and saves them.
        """

        print("Initialising RAG Pipeline...")
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_cache_path = self.cache_dir / "semantic_chunks.pkl"
        self.faiss_cache_path = self.cache_dir / "faiss_index"
        self.bm25_cache_path = self.cache_dir / "bm25_retriever.pkl"

        print("Loading embedding model...")
        self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        print("Initialising LLM Generator...")
        self._generator = Generator()

        if use_cache and self._cache_exists():
            print("Loading indices from cache...")
            try:
                self._load_from_cache()
                print("RAG Pipeline initialised succesfully from cache!")
                return
            except Exception as e:
                print(f"Cache loading failed: {e}. Building indices from scratch...")

        print("Building indices from scratch...")

        # 1. load docs
        print("Loading documents...")
        raw_docs = self.load_documents(data_directory)

        # 2. Chunk
        chunker = Chunker()
        self.semantic_docs = chunker.create_chunks(raw_docs)

        # 3. Initialize retriever
        self._retriever = HybridRetriever(
            semantic_docs=self.semantic_docs, embeddings=self._embeddings
        )

        if use_cache:
            print("Saving indices to cache...")
            self._save_to_cache()

        print("RAG Pipeline initialized successfully!")

    def load_documents(self, data_directory: str) -> list[Document]:
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
                            "source": str(json_path),
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

    def _cache_exists(self):
        """
        Check if all required cache files exist.

        Returns:
            bool: True if all cache files exist, False otherwise
        """
        return (
            self.chunks_cache_path.exists()
            and self.faiss_cache_path.exists()
            and self.bm25_cache_path.exists()
        )

    def _save_to_cache(self):
        """
        Save semantic chunks and retriever indices to disk for faster loading.

        Saves:
        - Semantic chunks (pickled documents)
        - FAISS vector index
        - BM25 retriever (pickled)
        """
        try:
            # Save semantic chunks
            with open(self.chunks_cache_path, "wb") as f:
                pickle.dump(self.semantic_docs, f)

            # Save FAISS index
            self._retriever.vector_retriever.vectorstore.save_local(str(self.faiss_cache_path))

            # Save BM25 retriever
            with open(self.bm25_cache_path, "wb") as f:
                pickle.dump(self._retriever.bm25_retriever, f)

            print(f"Indices cached successfully at {self.cache_dir}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _load_from_cache(self):
        """
        Load semantic chunks and retriever indices from cache.

        This is much faster than rebuilding indices from scratch.
        Typically reduces startup time from ~30 seconds to ~2-3 seconds.
        """

        # Load semantic chunks
        with open(self.chunks_cache_path, "rb") as f:
            self.semantic_docs = pickle.load(f)

        print(f"Loaded {len(self.semantic_docs)} cached semantic chunks")

        # Recreate retriever with cached data
        print("Loading FAISS index from cache...")
        vector_db = FAISS.load_local(
            str(self.faiss_cache_path), self._embeddings, allow_dangerous_deserialization=True
        )

        print("Loading BM25 index from cache...")
        with open(self.bm25_cache_path, "rb") as f:
            bm25_retriever = pickle.load(f)

        self._retriever = HybridRetriever.__new__(HybridRetriever)
        self._retriever.vector_retriever = vector_db.as_retriever(search_kwargs={"k": 25})
        self._retriever.bm25_retriever = bm25_retriever

        self._retriever.reranker = Reranker()

    def retrieve(self, query: str, top_n: int = RERANKER_TOP_N):
        """
        Retrieve relevant documents without generating an answer.

        Executes the hybrid retrieval and re-ranking workflow:
        1. Embeds the query using the same embedding model as documents
        2. Retrieves candidates via vector search (FAISS) and keyword search (BM25)
        3. Fuses results using Reciprocal Rank Fusion (RRF)
        4. Re-ranks fused results using cross-encoder model
        5. Returns top-n most relevant document chunks

        Args:
            query (str): User's question or search query
            top_n (int, optional): Number of top-ranked documents to return.
                Defaults to 5.

        Returns:
            list[Document]: Top-n most relevant document chunks, sorted by relevance.

        Example:
            >>> pipeline = RAGPipeline("./k8s_docs")
            >>> docs = pipeline.retrieve("How do I set memory limits?", top_n=3)
        """
        return self._retriever.search(
            query=query,
            top_n=top_n,
            score_threshold=RERANKER_SCORE_THRESHOLD,
            min_docs=MIN_RETRIEVED_DOCS,
        )

    def query(self, query: str, top_n: int = RERANKER_TOP_N):
        """
        Process a user query through the complete RAG pipeline.

        Full RAG workflow:
        1. Hybrid Retrieval: Retrieve relevant documents using vector + BM25 search
        2. Re-ranking: Re-rank results with cross-encoder for maximum relevance
        3. Score Filtering: Filter by relevance threshold (if configured)
        4. Generation: Generate natural language answer using LLM with retrieved context

        Args:
            query (str): User's question or search query
            top_n (int, optional): Number of top-ranked documents to retrieve.
                Defaults to 5. These documents provide context for the LLM.

        Returns:
            dict with 'answer', 'sources'
        """
        retrieved_docs = self._retriever.search(
            query=query,
            top_n=top_n,
            score_threshold=RERANKER_SCORE_THRESHOLD,
            min_docs=MIN_RETRIEVED_DOCS,
        )

        result = self._generator.generate(
            query=query, documents=retrieved_docs, include_sources=True
        )

        return result

    def query_stream(self, query: str, top_n: int = RERANKER_TOP_N):
        """
        Process a user query through the RAG pipeline with streaming generation.

        Full RAG workflow with streaming:
        1. Hybrid Retrieval: Retrieve relevant documents using vector + BM25 search
        2. Re-ranking: Re-rank results with cross-encoder for maximum relevance
        3. Streaming Generation: Stream LLM answer as it's generated

        Args:
            query (str): User's question or search query
            top_n (int, optional): Number of top-ranked documents to retrieve.
                Defaults to 5. These documents provide context for the LLM.

        Yields:
            dict: First yields metadata with sources, then yields answer chunks

        Example:
            >>> pipeline = RAGPipeline("./k8s_docs")
            >>> for chunk in pipeline.query_stream("How do I set memory limits?"):
            >>>     print(chunk, end="", flush=True)
        """
        retrieved_docs = self._retriever.search(
            query=query,
            top_n=top_n,
            score_threshold=RERANKER_SCORE_THRESHOLD,
            min_docs=MIN_RETRIEVED_DOCS,
        )

        return retrieved_docs, self._generator.generate_stream(
            query=query, documents=retrieved_docs
        )

    def query_with_contexts(self, query: str, top_n: int = RERANKER_TOP_N):
        """
        Process a user query and return answer WITH full retrieved contexts.

        This method is designed for evaluation purposes. Unlike query() which
        returns only snippets, this returns complete context texts needed for
        faithfulness evaluation.

        Full RAG workflow:
        1. Hybrid Retrieval: Retrieve relevant documents
        2. Re-ranking: Re-rank for maximum relevance
        3. Generation: Generate answer using LLM
        4. Context Extraction: Extract full text of retrieved contexts

        Args:
            query (str): User's question or search query
            top_n (int, optional): Number of top-ranked documents to retrieve.
                Defaults to 5.

        Returns:
            dict: {
                "question": str,
                "answer": str,
                "contexts": List[str],  # Full text of retrieved chunks
                "sources": List[dict]   # Source metadata
            }

        Example:
            >>> pipeline = RAGPipeline("./k8s_docs")
            >>> result = pipeline.query_with_contexts("What is a Pod?", top_n=5)
            >>> print(result["answer"])
            >>> print(result["contexts"])  # Full context texts for evaluation
        """
        retrieved_docs = self._retriever.search(
            query=query,
            top_n=top_n,
            score_threshold=RERANKER_SCORE_THRESHOLD,
            min_docs=MIN_RETRIEVED_DOCS,
        )

        result = self._generator.generate(
            query=query, documents=retrieved_docs, include_sources=True
        )

        contexts = [doc.page_content for doc in retrieved_docs]

        return {
            "question": query,
            "answer": result["answer"],
            "contexts": contexts,
            "sources": result["sources"],
        }
