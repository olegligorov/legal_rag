from config import (
    BM25_RETRIEVAL_K,
    EMBEDDING_MODEL,
    MIN_RETRIEVED_DOCS,
    RERANKER_SCORE_THRESHOLD,
    RERANKER_TOP_N,
    VECTOR_RETRIEVAL_K,
)
from langchain_community.vectorstores import FAISS
import hashlib
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from rag.reranker import Reranker


class HybridRetriever:
    """
    Hybrid retrieval system combining dense vector search and sparse keyword search.

    - Dense retrieval: FAISS vector database with semantic embeddings
    - Sparse retrieval: BM25 algorithm for keyword matching
    - Reciprocal Rank Fusion (RRF): Fuses results from both retrievers
    - Cross-encoder re-ranking: Final re-ranking for optimal relevance
    """

    def __init__(self, semantic_docs: list[Document], embeddings=None):
        """
        Initialize HybridRetriever with vector and BM25 retrievers.

        Args:
            semantic_docs (list[Document]): List of chunked documents to index for retrieval.
            embeddings (optional): Pre-loaded embeddings instance.
                If None, creates a new embedding model using config.
                Passing a pre-loaded model saves memory when sharing across components.
        """

        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        vector_db = FAISS.from_documents(documents=semantic_docs, embedding=embeddings)

        self.vector_retriever = vector_db.as_retriever(search_kwargs={"k": VECTOR_RETRIEVAL_K})
        self.bm25_retriever = BM25Retriever.from_documents(semantic_docs)
        self.bm25_retriever.k = BM25_RETRIEVAL_K

        self.reranker = Reranker()

    def rrf(self, vector_results, bm25_results, k=60):
        """
        Implements a Reciprocal Rank Fusion.
        k is smoothing constant. Standard is 60.
        RRF = sum(1 / (k + rank))
        Optimized to avoid duplicate Document objects.

        Studies have shown that k = 60 performs well across various datasets and retrieval tasks.
        It provides a good balance between the influence of top-ranked and lower-ranked items. For example:
        - For rank 1: 1/(1+60) ≈ 0.0164
        - For rank 10: 1/(10+60) ≈ 0.0143
        - For rank 100: 1/(100+60) ≈ 0.00625

        k = 60 helps break ties effectively, especially for lower-ranked items where small differences in the original rankings might not be significant.
        This value has shown to be robust across different types of retrieval systems and data distributions.
        """

        rrf_scores = {}
        doc_map = {}

        for rank, doc in enumerate(vector_results, start=1):
            doc_id = self._get_id(doc)
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))

        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = self._get_id(doc)
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, score in sorted_ids]

    def _get_id(self, doc):
        """
        Generate a deterministic unique ID for a document.

        Uses MD5 hash of document content to ensure uniqueness. If 'source' metadata
        exists (e.g., filename), combines it with a truncated hash for readability.

        This is critical for RRF fusion to correctly identify duplicate documents
        retrieved by both vector and BM25 methods.

        Args:
            doc (Document): LangChain Document object

        Returns:
            str: Unique identifier in format "source_hash16" or just "hash" if no source

        Example:
            doc with source="configmaps.md" and content "..."
            -> "configmaps.md_a1b2c3d4e5f6g7h8"
        """

        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        if "source" in doc.metadata:
            return f"{doc.metadata['source']}_{content_hash[:16]}"

        return content_hash

    def search(
        self,
        query: str,
        top_n=RERANKER_TOP_N,
        score_threshold=RERANKER_SCORE_THRESHOLD,
        min_docs=MIN_RETRIEVED_DOCS,
    ):
        """
        Perform hybrid search with re-ranking to retrieve most relevant documents.

        Pipeline:
        1. Vector Retrieval: Retrieve top-k documents using FAISS semantic search
        2. BM25 Retrieval: Retrieve top-k documents using BM25 keyword matching
        3. RRF Fusion: Combine and rank results using Reciprocal Rank Fusion
        4. Re-ranking: Apply cross-encoder model to re-rank fused results
        5. Score Filtering: Filter documents by relevance score threshold (optional)
        6. Return: Top-n highest scoring documents after re-ranking and filtering

        This multi-stage approach ensures both semantic relevance and keyword precision.

        Args:
            query (str): User's search query or question
            top_n (int, optional): Number of final documents to return after re-ranking.
                Defaults to RERANK_TOP_N.
            score_threshold (float, optional): Minimum reranker score threshold.
                Documents below this threshold are filtered out. If None, no filtering.
                Typical cross-encoder scores range from -10 to +10, with positive
                scores indicating relevance. Recommended thresholds:
                - 0.5-1.0: Strict filtering (high precision)
                - 0.0-0.5: Moderate filtering (balanced)
                - -1.0-0.0: Lenient filtering (high recall)
            min_docs (int, optional): Minimum number of documents to return even if
                they don't meet the threshold. Prevents returning empty results.
                Defaults to 1.

        Returns:
            tuple: (list[Document], list[float]) - Top-n most relevant documents with their scores,
                sorted by relevance score. Each document includes page_content and metadata (source, etc.)
        """

        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        fused_results = self.rrf(vector_results=vector_docs, bm25_results=bm25_docs)

        scored_docs = self.reranker.rerank_with_scores(query, fused_results, top_n=top_n)

        if score_threshold is not None:
            filtered_docs = [(score, doc) for score, doc in scored_docs if score >= score_threshold]

            if len(filtered_docs) < min_docs and len(scored_docs) >= min_docs:
                filtered_docs = scored_docs[:min_docs]
        else:
            filtered_docs = scored_docs

        if not filtered_docs:
            return [], []

        top_scores, top_docs = zip(*filtered_docs)
        return list(top_docs), [float(score) for score in top_scores]
