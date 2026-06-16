"""Generation quality metrics — no ragas dependency.

All three metrics use the same embedding model and LLM already in the pipeline.

- faithfulness:       LLM-as-judge — are all answer claims supported by the contexts?
- answer_relevancy:   cosine similarity between the question and the answer (via embeddings)
- answer_correctness: cosine similarity between the answer and the ground-truth answer

Plus citation-quality metrics that work directly on the answer string and the
expected-articles list (no LLM needed):

- citation_recall:     fraction of expected articles cited in the answer
- citation_precision:  fraction of cited articles that are in the expected set
- set_recall_at_k:     unordered set-membership recall (used for the agent path
                       where retrieval order is not a relevance ranking)

Citation parsing is delegated to ``backend.citations`` so the agent's
runtime validator and this evaluator can't drift apart.
"""

from __future__ import annotations

import math

from citations import (
    count_unparseable_citations,
    expected_articles_to_set,
    extract_cited_articles,
    normalize_article,
    normalize_law_id,
)


# ---------------------------------------------------------------------------
# Test-friendly aliases. ``test_metrics.py`` imports these names; keeping
# them exported preserves the existing test suite and gives external
# callers a stable surface even if ``citations`` is reorganized.
# ---------------------------------------------------------------------------

_normalize_law_id = normalize_law_id
_normalize_article = normalize_article


def _split_citation(marker: str) -> tuple[str, str] | None:
    """Backward-compatible alias for ``citations.parse_citation``."""
    from citations import parse_citation as _parse  # local to avoid circular noise

    return _parse(marker)


def _expected_set(expected: list[dict]) -> set[tuple[str, str]]:
    return expected_articles_to_set(expected)


# Re-export for callers that imported these names from this module historically.
__all__ = [
    "answer_correctness",
    "answer_relevancy",
    "citation_precision",
    "citation_recall",
    "count_unparseable_citations",
    "extract_cited_articles",
    "faithfulness",
    "precision_at_k",
    "recall_at_k",
    "set_recall_at_k",
    "tool_call_count",
]


# ---------------------------------------------------------------------------
# Retrieval metrics (pure Python, no deps)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[dict], expected: list[dict], k: int) -> float:
    """Fraction of top-k retrieved articles that are in the expected set.

    Assumes ``retrieved`` is ranked (highest relevance first) — used for
    the baseline single-shot pipeline whose retriever produces a real
    rerank score. The agent path's retrieved list is NOT rank-ordered;
    use ``set_recall_at_k`` for it instead.
    """
    if k <= 0:
        return 0.0
    expected_set = _expected_set(expected)
    top_k = retrieved[:k]
    hits = sum(
        1
        for r in top_k
        if (normalize_article(r["article"]), normalize_law_id(r["law_id"])) in expected_set
    )
    return hits / k


def recall_at_k(retrieved: list[dict], expected: list[dict], k: int) -> float:
    """Fraction of expected articles found in the top-k retrieved.

    Same caveat as ``precision_at_k`` — for the rank-ordered baseline
    only.
    """
    if not expected or k <= 0:
        return 0.0
    expected_set = _expected_set(expected)
    top_k_set = {
        (normalize_article(r["article"]), normalize_law_id(r["law_id"]))
        for r in retrieved[:k]
    }
    hits = len(top_k_set & expected_set)
    return hits / len(expected_set)


def set_recall_at_k(retrieved: list[dict], expected: list[dict], k: int) -> float:
    """Unordered set-membership recall over the first ``k`` retrieved articles.

    Use this when ``retrieved`` is not a relevance ranking — for example
    the agent's ``sources`` list, which is ordered by first-appearance
    across tool calls. The metric reports "of the expected articles, how
    many are anywhere in the agent's retrieved set up to size k?".

    Reporting both ``recall_at_k`` (rank-aware) and ``set_recall_at_k``
    (rank-agnostic) lets the head-to-head comparison be apples-to-apples
    at the set level while still surfacing rank quality on the baseline
    side.
    """
    if not expected or k <= 0:
        return 0.0
    expected_set = _expected_set(expected)
    seen: set[tuple[str, str]] = set()
    for r in retrieved:
        key = (normalize_article(r.get("article", "")), normalize_law_id(r.get("law_id", "")))
        if not key[0] or not key[1]:
            continue
        seen.add(key)
        if len(seen) >= k:
            break
    return len(seen & expected_set) / len(expected_set)


# ---------------------------------------------------------------------------
# Citation-quality metrics — ANSWER vs EXPECTED, no LLM.
# ---------------------------------------------------------------------------

def citation_recall(answer: str, expected: list[dict]) -> float:
    """Fraction of expected articles that the answer actually cites.

    Uses the ``[Чл. X, <law_id>]`` markers in the answer text. Matching is
    case-insensitive and strips ``ал. N`` qualifiers from the citation —
    a citation of ``[Чл. 155, ал. 4, Кодекс на труда]`` counts as a hit
    for an expected article of ``{law_id: КОДЕКС НА ТРУДА, article: Чл. 155.}``.
    """
    if not expected:
        return 0.0
    cited = set(extract_cited_articles(answer))
    expected_set = _expected_set(expected)
    return len(cited & expected_set) / len(expected_set)


def citation_precision(answer: str, expected: list[dict]) -> float | None:
    """Fraction of distinct articles cited in the answer that are expected.

    Returns ``None`` when the answer contains no citations at all so the
    aggregator can distinguish "didn't cite" from "cited wrong" (a real
    issue raised in code review: averaging 0.0 here biased toward systems
    that produced more answers, regardless of correctness).
    """
    cited = set(extract_cited_articles(answer))
    if not cited:
        return None
    expected_set = _expected_set(expected)
    return len(cited & expected_set) / len(cited)


# ---------------------------------------------------------------------------
# Cost / efficiency
# ---------------------------------------------------------------------------

def tool_call_count(trace: list[dict] | None) -> int:
    """Count tool calls in an agent trace.

    Single-shot baseline doesn't have a trace — pass ``None`` and the
    function returns 0. Counts every ``{type: "tool_call"}`` entry.
    """
    if not trace:
        return 0
    return sum(1 for evt in trace if evt.get("type") == "tool_call")


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
