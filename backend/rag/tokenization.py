"""
Bulgarian-aware BM25 tokenizer.

LangChain's BM25Retriever uses a default tokenizer of `text.split()`, which is
too naive for Cyrillic legal text:
  - it does not lowercase, so "Работодател" and "работодател" miss
  - it does not strip punctuation, so "договор," and "договор" miss
  - it does not handle Bulgarian's heavy noun inflection (definite article
    suffixes, plural endings), so "работодател" and "работодателят" miss

This module implements a light tokenizer that is good enough for retrieval-
quality BM25 over Bulgarian law text:
  1. Lowercase (Unicode-aware — Cyrillic uppercase has lowercase mappings).
  2. Split on non-letter/digit characters.
  3. Drop tokens shorter than 2 chars.
  4. Strip a small set of common Bulgarian suffixes (definite articles,
     plural endings) from tokens that are long enough to keep a meaningful
     stem after the strip.

It deliberately stops short of a real stemmer: that would be a heavier
dependency and the failure modes (over-stemming) hurt legal terminology
more than they help.
"""

from __future__ import annotations

import re

# Suffixes are tried in this order; the first match wins. Keep longest-first
# inside each "family" so we don't strip "а" from "та" before "та" is tried.
_BG_SUFFIXES: tuple[str, ...] = (
    # Definite article — singular
    "ите",
    "ият",
    "ьт",
    "ът",
    "ят",
    "та",
    "то",
    "те",
    # Plural
    "ове",
    "еве",
    "ища",
    "и",
    # Verb / participle tail vowels (very conservative)
    "ам",
    "ят",
    "ят",
    "ят",
    "ат",
    "а",
    "я",
    "е",
    "о",
    "ъ",
)

# Min stem length to keep after suffix stripping. Below this we leave the
# token alone — short tokens are usually function words or already a stem.
_MIN_STEM = 4

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _strip_suffix(token: str) -> str:
    for suf in _BG_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= _MIN_STEM:
            return token[: -len(suf)]
    return token


def bulgarian_bm25_tokenize(text: str) -> list[str]:
    """Tokenize Bulgarian (or mixed BG/EN) text for BM25 indexing/querying.

    - Unicode-aware lowercase.
    - Split on non-letter characters (digits and underscores excluded).
    - Drop tokens shorter than 2 chars.
    - Light suffix stripping for Bulgarian inflection.
    """
    out: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        tok = match.group(0)
        if len(tok) < 2:
            continue
        out.append(_strip_suffix(tok))
    return out
