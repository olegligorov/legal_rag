"""Single source of truth for citation parsing.

Citations in the legal RAG project are emitted by the LLM as inline
markers like ``[Чл. 155, Кодекс на труда]`` or ``[Чл. 50, ал. 1, Закон
за защита на потребителите]``. They appear in:

- Sub-answers returned by ``query_rag_tool`` / ``batch_query_tool``
- The final agent answer (and ideally preserved verbatim from sub-answers)
- The baseline ``/api/query`` answer
- The ground truth in ``expected_articles`` (different shape: structured
  ``{law_id, article}`` rather than the bracket marker)

This module provides:

- ``CITATION_PATTERN`` — regex for the bracket marker shape
- ``extract_citation_markers`` — pull markers out of a string
- ``parse_citation`` — turn one marker into ``(article, law_id)``
- ``extract_cited_articles`` — both of the above composed
- ``normalize_law_id`` / ``normalize_article`` — case-insensitive
  comparison keys (handle the trailing ``.``, ``ал. N`` qualifiers,
  Cyrillic-letter article suffixes like ``Чл. 161з``)

The agent-side validator (``backend/agent/runner.py``) and the eval-side
metrics (``backend/evaluation/metrics.py``) BOTH import from here so a
prompt drift can't make them disagree silently.
"""

from __future__ import annotations

import re
import unicodedata

# Bracket-style citation: starts with ``Чл.`` (article), ends at ``]``.
# Everything inside the brackets is parsed by ``parse_citation``.
CITATION_PATTERN = re.compile(r"\[Чл\.[^\[\]]+?\]")

_AL_RE = re.compile(r"ал\.\s*[0-9а-яa-z]+", re.IGNORECASE)
_TRAILING_PUNCT_RE = re.compile(r"[.,;\s]+$")


def normalize_law_id(law_id: str) -> str:
    """Case- and whitespace-insensitive law identifier.

    Also runs Unicode NFC normalization so visually identical strings
    that differ only in combining-character form (rare, but possible
    when text comes from mixed sources) compare equal.
    """
    return " ".join(unicodedata.normalize("NFC", law_id).upper().split())


def normalize_article(article: str) -> str:
    """Normalize an article tag.

    ``Чл. 155.`` → ``ЧЛ. 155``
    ``Чл. 155, ал. 4`` → ``ЧЛ. 155``
    ``Чл. 161з`` → ``ЧЛ. 161З`` (letter suffix preserved)
    ``Чл. 161з, ал. 2,`` → ``ЧЛ. 161З``
    """
    text = unicodedata.normalize("NFC", article)
    text = _AL_RE.sub("", text)  # drop "ал. X" parts
    text = _TRAILING_PUNCT_RE.sub("", text.strip())
    text = " ".join(text.split())
    return text.upper()


def parse_citation(marker: str) -> tuple[str, str] | None:
    """Parse ``[Чл. 155, ал. 4, Кодекс на труда]`` → (``ЧЛ. 155``, ``КОДЕКС НА ТРУДА``).

    Returns ``None`` if the marker doesn't follow the expected shape.
    Comma-separated parts; first part is the article; the last
    non-``ал.`` part is the law identifier.
    """
    inner = marker.strip("[]").strip()
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    article = parts[0]
    law_parts = [p for p in parts[1:] if not p.lower().startswith("ал.")]
    if not law_parts:
        return None
    law_id = law_parts[-1]
    return normalize_article(article), normalize_law_id(law_id)


def extract_citation_markers(text: str) -> list[str]:
    """Return the raw bracket markers from a string, in order.

    Useful when the caller needs the original text (e.g. to check
    citation passthrough between a sub-answer and the final answer).
    Duplicates are kept; deduplication is the caller's job.
    """
    return CITATION_PATTERN.findall(text or "")


def extract_cited_articles(answer: str) -> list[tuple[str, str]]:
    """Return the list of normalized ``(article, law_id)`` tuples cited.

    Order preserved. Malformed markers (parse_citation returns None)
    are silently dropped — call ``count_unparseable_citations`` if
    you need to detect drift in the marker shape.
    """
    out: list[tuple[str, str]] = []
    for marker in extract_citation_markers(answer):
        parsed = parse_citation(marker)
        if parsed is not None:
            out.append(parsed)
    return out


def count_unparseable_citations(answer: str) -> int:
    """How many bracket markers in ``answer`` failed to parse.

    Non-zero values indicate a drift in the citation shape — useful
    as a per-question diagnostic so a regex regression shows up as a
    metric rather than disappearing into a 0.0 precision/recall.
    """
    n = 0
    for marker in extract_citation_markers(answer):
        if parse_citation(marker) is None:
            n += 1
    return n


def expected_articles_to_set(expected: list[dict]) -> set[tuple[str, str]]:
    """Convert ``[{law_id, article}, ...]`` to the comparison-keyed set."""
    return {
        (normalize_article(e["article"]), normalize_law_id(e["law_id"]))
        for e in expected
    }
