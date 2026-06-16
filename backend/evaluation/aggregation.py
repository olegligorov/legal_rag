"""Per-metric aggregation across evaluation rows.

Both ``Evaluator`` and ``AgentEvaluator`` produce a list of per-question
rows; this module reduces them to summary statistics. The previous
mean-only approach was insufficient for thesis-grade reporting (one tail
outlier dragged the mean; counts averaged as floats).

For each metric we report::

    {
      "n": int,                  # rows that contributed (non-None values)
      "mean": float | None,
      "median": float | None,
      "p95": float | None,
      "min": float | None,
      "max": float | None,
      "stdev": float | None,     # population stdev; None if n < 2
      "sum": int | float | None, # only meaningful for count-like metrics
    }

Skipping ``None`` values automatically gives us the "abstention rate"
information requested in the review — ``n`` per metric tells the reader
how many questions actually contributed.
"""

from __future__ import annotations

import statistics
from typing import Iterable


def _safe_quantile(values: list[float], q: float) -> float:
    """Stdlib ``quantiles`` needs n >= 2; degrade gracefully for n=1."""
    if len(values) <= 1:
        return values[0] if values else None  # type: ignore[return-value]
    # ``method='inclusive'`` to match what numpy users expect.
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    idx = max(0, min(int(round(q * 100)) - 1, len(cuts) - 1))
    return cuts[idx]


def aggregate_metric(values: Iterable[float | int | None]) -> dict:
    """Reduce a stream of values to summary stats. ``None`` entries are skipped."""
    nums = [v for v in values if v is not None]
    n = len(nums)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "stdev": None,
            "sum": None,
        }
    return {
        "n": n,
        "mean": round(statistics.fmean(nums), 4),
        "median": round(statistics.median(nums), 4),
        "p95": round(_safe_quantile(sorted(nums), 0.95), 4),
        "min": round(min(nums), 4),
        "max": round(max(nums), 4),
        "stdev": round(statistics.pstdev(nums), 4) if n >= 2 else None,
        "sum": round(sum(nums), 4),
    }


def aggregate_rows(rows: list[dict], keys: list[str]) -> dict:
    """Compute the per-metric aggregate for ``keys`` across ``rows``.

    Rows missing the key (e.g. ``None`` for citation_precision when the
    answer had no citations, or rows that errored out and only carry
    ``error``) contribute zero to ``n`` for that key.
    """
    out: dict = {}
    for key in keys:
        values = [r.get(key) for r in rows]
        out[key] = aggregate_metric(values)
    return out


def deltas(baseline: dict, agent: dict) -> dict:
    """Compute mean-delta = agent.mean - baseline.mean for shared keys.

    The shape mirrors the pre-refactor head-to-head output but operates
    on the new richer aggregate. ``stdev`` is also surfaced so the
    reader can spot when the delta is dwarfed by within-run variance.
    """
    out: dict = {}
    for key in baseline:
        if key not in agent:
            continue
        b = baseline[key]
        a = agent[key]
        if b.get("mean") is None or a.get("mean") is None:
            out[key] = {"baseline": b, "agent": a, "delta_mean": None}
            continue
        out[key] = {
            "baseline": b,
            "agent": a,
            "delta_mean": round(a["mean"] - b["mean"], 4),
        }
    return out
