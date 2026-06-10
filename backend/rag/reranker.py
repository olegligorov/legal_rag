from config import RERANKER_MODEL, RERANKER_TOP_N
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import torch


class Reranker:
    """
    Cross-encoder based document re-ranker.
    Re-ranks retrieved documents based on query-document relevance.
    """

    def __init__(self):
        device = self._detect_device()
        self.model = CrossEncoder(RERANKER_MODEL, device=device)

    def _detect_device(self) -> str:
        """Auto-detect the best available device. Priority: MPS (Mac) > CUDA > CPU."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def rerank_with_scores(
        self, query: str, documents: list[Document], top_n: int = RERANKER_TOP_N
    ) -> list[tuple[float, Document]]:
        """
        Re-rank documents by query-document relevance and return scored pairs.

        Args:
            query: Search query string.
            documents: Candidate documents to re-rank.
            top_n: Number of top documents to return.

        Returns:
            List of (score, document) tuples sorted by descending score.
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return scored_docs[:top_n]
