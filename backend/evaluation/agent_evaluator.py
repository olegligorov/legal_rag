"""Agent-path evaluator — runs ``backend.agent.run_agent`` per question.

Mirrors ``Evaluator`` so head-to-head comparisons line up. Differences:

- Uses a SINGLE event loop for all questions (one ``asyncio.run`` over a
  ``_run_async`` method) — module-level caches in ``agent.mcp_client`` and
  ``agent.graph`` are tied to whichever loop instantiated them, and reusing
  them across loops produces "Event loop is closed" hangs.
- Catches exceptions per question. A failure becomes an error row; the
  aggregate keeps going. ``--limit 15`` losing question 12 to a transient
  Anthropic 5xx no longer wastes the whole run.
- For retrieval, reports ``set_recall_at_k`` (rank-agnostic) since the
  agent's source list is not a relevance ranking — see
  ``evaluation/metrics.py:set_recall_at_k`` for the rationale. ``recall@k``
  is also reported but flagged as unranked in the JSON.
- Records ``tool_calls_used``, ``citation_recall/precision``, citation
  passthrough breakdown, latency.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from agent import run_agent
from evaluation.aggregation import aggregate_rows
from evaluation.metrics import (
    answer_correctness,
    answer_relevancy,
    citation_precision,
    citation_recall,
    faithfulness,
    recall_at_k,
    set_recall_at_k,
    tool_call_count,
)

if TYPE_CHECKING:
    from models.rag_pipeline import RAGPipeline


KS = [3, 5]


class AgentEvaluator:
    """Run the agent over a dataset and produce metric rows."""

    def __init__(
        self,
        pipeline: "RAGPipeline",
        dataset_path: str | Path | None = None,
        test_cases: list[dict] | None = None,
    ):
        """``test_cases`` (if provided) wins over ``dataset_path`` — caller
        already loaded/sliced the dataset in memory."""
        self._pipeline = pipeline
        self._embeddings = pipeline._embeddings
        self._llm: object | None = None
        self._test_cases = test_cases
        self._dataset_path = (
            Path(dataset_path)
            if dataset_path is not None
            else Path(__file__).parent / "dataset.json"
        )

    def run(self, skip_generation: bool = False) -> dict:
        if self._test_cases is not None:
            test_cases = self._test_cases
        else:
            with open(self._dataset_path, encoding="utf-8") as f:
                test_cases = json.load(f)

        if not skip_generation:
            self._llm = self._pipeline._generator.llm

        # ONE event loop for the whole run. Module-level MCP client +
        # compiled graph are bound to the first loop they touch; spawning
        # a new loop per question (asyncio.run inside the for-body) leads
        # to "Event loop is closed" once the SSE/HTTP transport sees
        # operations on a different loop.
        per_question, errors = asyncio.run(self._run_async(test_cases, skip_generation))

        aggregate = aggregate_rows(per_question, _aggregable_keys(skip_generation))
        return {
            "aggregate": aggregate,
            "per_question": per_question,
            "errors": errors,
        }

    async def _run_async(
        self, test_cases: list[dict], skip_generation: bool
    ) -> tuple[list[dict], int]:
        per_question: list[dict] = []
        errors = 0
        for tc in test_cases:
            try:
                row = await self._evaluate_one(tc, skip_generation)
            except Exception as exc:  # noqa: BLE001 — we want to keep going
                errors += 1
                row = {
                    "id": tc.get("id"),
                    "question": tc.get("question"),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                print(f"  [{tc.get('id')}] ERROR: {exc!r}", flush=True)
                per_question.append(row)
                continue

            per_question.append(row)
            print(
                f"  [{tc['id']}] tool_calls={row['tool_calls_used']}"
                f" set_recall@5={row['set_recall@5']}"
                f" cite_recall={row['citation_recall']}",
                end="",
                flush=True,
            )
            if not skip_generation:
                print(
                    f" faith={row.get('faithfulness', '?')}"
                    f" corr={row.get('answer_correctness', '?')}",
                    end="",
                    flush=True,
                )
            print(f" ({row['latency_s']:.1f}s)", flush=True)
        return per_question, errors

    async def _evaluate_one(self, tc: dict, skip_generation: bool) -> dict:
        t0 = time.monotonic()
        result = await run_agent(question=tc["question"])
        latency = time.monotonic() - t0

        # Build retrieved_articles in {law_id, article} shape. Sources are
        # in first-appearance order — NOT a relevance ranking, hence
        # set_recall_at_k below rather than the rank-aware precision_at_k.
        retrieved = [
            {"law_id": s.get("law_id", ""), "article": s.get("article", "")}
            for s in result["sources"]
            if s.get("law_id") and s.get("article")
        ]
        expected = tc["expected_articles"]
        answer = result["answer"]

        row: dict = {
            "id": tc["id"],
            "question": tc["question"],
            "answer": answer,
            "retrieved_articles": retrieved,
            "expected_articles": expected,
            "tool_calls_used": result["tool_calls_used"],
            "trace_event_count": len(result["trace"]),
            "latency_s": round(latency, 3),
            "citation_recall": round(citation_recall(answer, expected), 4),
            "citations_preserved": len(result["citation_validation"]["preserved"]),
            "citations_dropped": len(result["citation_validation"]["dropped"]),
            "citations_invented": len(result["citation_validation"]["invented"]),
        }

        cp = citation_precision(answer, expected)
        row["citation_precision"] = round(cp, 4) if cp is not None else None

        for k in KS:
            row[f"set_recall@{k}"] = round(set_recall_at_k(retrieved, expected, k), 4)
            # recall@k retained for diagnostics but the agent's order is
            # not a real ranking — the aggregator will treat both the
            # same way; the JSON column name signals the difference.
            row[f"recall@{k}"] = round(recall_at_k(retrieved, expected, k), 4)

        row["tool_calls_in_trace"] = tool_call_count(result["trace"])
        if row["tool_calls_in_trace"] != row["tool_calls_used"]:
            print(
                f"    WARN [{tc['id']}] tool_calls_in_trace="
                f"{row['tool_calls_in_trace']} != tool_calls_used={row['tool_calls_used']}"
            )

        if not skip_generation:
            contexts = [s.get("snippet", "") for s in result["sources"]]
            # When the agent has no contexts (all sub-queries returned
            # INSUFFICIENT_CONTEXT), faithfulness is undefined rather than
            # 0.0 — using None lets the aggregator skip and report a
            # separate ``faithfulness_coverage`` rate elsewhere.
            row["faithfulness"] = (
                faithfulness(answer, contexts, self._llm) if contexts else None
            )
            row["answer_relevancy"] = answer_relevancy(tc["question"], answer, self._embeddings)
            gt = tc.get("ground_truth")
            if gt:
                row["answer_correctness"] = answer_correctness(answer, gt, self._embeddings)

        return row


def _aggregable_keys(skip_generation: bool) -> list[str]:
    keys = (
        [f"set_recall@{k}" for k in KS]
        + [f"recall@{k}" for k in KS]
        + [
            "citation_recall",
            "citation_precision",
            "tool_calls_used",
            "latency_s",
            "citations_preserved",
            "citations_dropped",
            "citations_invented",
        ]
    )
    if not skip_generation:
        keys += ["faithfulness", "answer_relevancy", "answer_correctness"]
    return keys
