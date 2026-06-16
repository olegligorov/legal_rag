"""RAG evaluation orchestrator — no ragas dependency.

Usage:
    from evaluation.evaluator import Evaluator
    ev = Evaluator(pipeline)
    results = ev.run(skip_generation=False)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from evaluation.aggregation import aggregate_rows
from evaluation.metrics import (
    answer_correctness,
    answer_relevancy,
    citation_precision,
    citation_recall,
    faithfulness,
    precision_at_k,
    recall_at_k,
)

if TYPE_CHECKING:
    from models.rag_pipeline import RAGPipeline

DATASET_PATH = Path(__file__).parent / "dataset.json"
KS = [3, 5]


class Evaluator:
    def __init__(
        self,
        pipeline: "RAGPipeline",
        dataset_path: str | Path | None = None,
        test_cases: list[dict] | None = None,
    ):
        """``test_cases`` (if provided) wins over ``dataset_path`` — caller
        already loaded/sliced the dataset in memory and there's nothing to
        write to disk."""
        self._pipeline = pipeline
        self._embeddings = pipeline._embeddings  # HuggingFaceEmbeddings, already loaded
        self._llm: object | None = None  # lazy-init only when generation metrics needed
        self._test_cases = test_cases
        self._dataset_path = Path(dataset_path) if dataset_path else DATASET_PATH

    def run(self, skip_generation: bool = False) -> dict:
        if self._test_cases is not None:
            test_cases = self._test_cases
        else:
            with open(self._dataset_path, encoding="utf-8") as f:
                test_cases = json.load(f)

        if not skip_generation:
            self._llm = self._pipeline._generator.llm

        per_question = []
        for tc in test_cases:
            row = self._evaluate_one(tc, skip_generation)
            per_question.append(row)
            print(
                f"  [{tc['id']}] recall@5={row['recall@5']}"
                f" cite_recall={row['citation_recall']}",
                end="",
                flush=True,
            )
            if not skip_generation:
                print(
                    f"  faith={row.get('faithfulness', '?')}"
                    f"  rel={row.get('answer_relevancy', '?')}"
                    f"  corr={row.get('answer_correctness', '?')}",
                    end="",
                    flush=True,
                )
            print(f" ({row['latency_s']:.1f}s)", flush=True)

        aggregate = aggregate_rows(per_question, _aggregable_keys(skip_generation))
        return {"aggregate": aggregate, "per_question": per_question}

    def _evaluate_one(self, tc: dict, skip_generation: bool) -> dict:
        t0 = time.monotonic()
        result = self._pipeline.query_with_contexts(tc["question"])
        latency = time.monotonic() - t0

        retrieved = result["retrieved_articles"]
        expected = tc["expected_articles"]
        answer = result["answer"]

        row: dict = {
            "id": tc["id"],
            "question": tc["question"],
            "answer": answer,
            "retrieved_articles": retrieved,
            "expected_articles": expected,
            "latency_s": round(latency, 3),
            "citation_recall": round(citation_recall(answer, expected), 4),
        }
        cp = citation_precision(answer, expected)
        row["citation_precision"] = round(cp, 4) if cp is not None else None

        for k in KS:
            row[f"precision@{k}"] = round(precision_at_k(retrieved, expected, k), 4)
            row[f"recall@{k}"] = round(recall_at_k(retrieved, expected, k), 4)

        if not skip_generation:
            row["faithfulness"] = (
                faithfulness(answer, result["contexts"], self._llm) if result["contexts"] else None
            )
            row["answer_relevancy"] = answer_relevancy(tc["question"], answer, self._embeddings)
            gt = tc.get("ground_truth")
            if gt:
                row["answer_correctness"] = answer_correctness(answer, gt, self._embeddings)

        return row


def _aggregable_keys(skip_generation: bool) -> list[str]:
    keys = (
        [f"precision@{k}" for k in KS]
        + [f"recall@{k}" for k in KS]
        + ["citation_recall", "citation_precision", "latency_s"]
    )
    if not skip_generation:
        keys += ["faithfulness", "answer_relevancy", "answer_correctness"]
    return keys
