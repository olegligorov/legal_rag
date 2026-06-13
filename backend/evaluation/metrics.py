"""Generation quality metrics — no ragas dependency.

All three metrics use the same embedding model and LLM already in the pipeline.

- faithfulness:       LLM-as-judge — are all answer claims supported by the contexts?
- answer_relevancy:   cosine similarity between the question and the answer (via embeddings)
- answer_correctness: cosine similarity between the answer and the ground-truth answer
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Retrieval metrics (pure Python, no deps)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[dict], expected: list[dict], k: int) -> float:
    """Fraction of top-k retrieved articles that are in the expected set."""
    if k <= 0:
        return 0.0
    expected_set = {(e["law_id"], e["article"]) for e in expected}
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if (r["law_id"], r["article"]) in expected_set)
    return hits / k


def recall_at_k(retrieved: list[dict], expected: list[dict], k: int) -> float:
    """Fraction of expected articles found in the top-k retrieved."""
    if not expected or k <= 0:
        return 0.0
    expected_set = {(e["law_id"], e["article"]) for e in expected}
    top_k_set = {(r["law_id"], r["article"]) for r in retrieved[:k]}
    hits = len(top_k_set & expected_set)
    return hits / len(expected_set)


# ---------------------------------------------------------------------------
# Generation metrics (need embeddings + LLM from the pipeline)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def answer_relevancy(question: str, answer: str, embeddings) -> float:
    """Cosine similarity between question and answer embeddings.

    High score → answer is topically aligned with the question.
    """
    vecs = embeddings.embed_documents([question, answer])
    return round(_cosine(vecs[0], vecs[1]), 4)


def answer_correctness(answer: str, ground_truth: str, embeddings) -> float:
    """Cosine similarity between answer and ground-truth answer embeddings.

    High score → answer conveys the same information as the reference.
    """
    vecs = embeddings.embed_documents([answer, ground_truth])
    return round(_cosine(vecs[0], vecs[1]), 4)


_FAITHFULNESS_PROMPT = """\
You are a strict fact-checker. Given a set of source passages and an answer, \
determine whether every factual claim in the answer is directly supported by \
the source passages. Do NOT use outside knowledge.

Source passages:
{contexts}

Answer:
{answer}

Reply with ONLY a JSON object: {{"score": <float between 0 and 1>, "reasoning": "<one sentence>"}}
A score of 1.0 means every claim is fully supported. 0.0 means the answer contains \
unsupported or contradictory claims.
"""


def faithfulness(answer: str, contexts: list[str], llm) -> float:
    """LLM-as-judge: fraction of answer claims supported by the retrieved contexts."""
    combined = "\n\n---\n\n".join(contexts)
    prompt = _FAITHFULNESS_PROMPT.format(contexts=combined, answer=answer)

    import json as _json

    raw = llm.invoke(prompt).content.strip()
    # Strip markdown fences if the model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = _json.loads(raw)
        return round(float(data["score"]), 4)
    except Exception:
        return 0.0
