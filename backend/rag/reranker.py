from config import RERANKER_MODEL, RERANKER_TOP_N
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import math
import torch


class Reranker:
    """
    Cross-encoder based document re-ranker.
    Re-ranks retrieved documents based on query-document relevance.

    Scores are sigmoid-normalized to the [0, 1] range so that the
    `RERANKER_SCORE_THRESHOLD` config value has a model-independent meaning
    (≈ probability of relevance) rather than a raw, unbounded logit.
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

    @staticmethod
    def _sigmoid(x: float) -> float:
        # math.exp guards against overflow for very negative logits implicitly:
        # exp(-large) -> 0, so 1/(1+0) = 1. For very positive logits, exp blows up;
        # cap the input to a sane range.
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
        return math.exp(max(x, -60.0)) / (1.0 + math.exp(max(x, -60.0)))

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
            Scores are in [0, 1] (sigmoid of the cross-encoder's raw logit).
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        raw_scores = self.model.predict(pairs)
        scores = [self._sigmoid(float(s)) for s in raw_scores]

        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return scored_docs[:top_n]
