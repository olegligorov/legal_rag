"""RAG evaluation orchestrator — no ragas dependency.

Usage:
    from evaluation.evaluator import Evaluator
    ev = Evaluator(pipeline)
    results = ev.run(skip_generation=False)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from evaluation.metrics import (
    answer_correctness,
    answer_relevancy,
    faithfulness,
    precision_at_k,
    recall_at_k,
)

if TYPE_CHECKING:
    from models.rag_pipeline import RAGPipeline

DATASET_PATH = Path(__file__).parent / "dataset.json"
KS = [3, 5]


class Evaluator:
    def __init__(self, pipeline: "RAGPipeline"):
        self._pipeline = pipeline
        self._embeddings = pipeline._embeddings  # HuggingFaceEmbeddings, already loaded
        self._llm: object | None = None  # lazy-init only when generation metrics needed

    def run(self, skip_generation: bool = False) -> dict:
        with open(DATASET_PATH, encoding="utf-8") as f:
            test_cases = json.load(f)

        if not skip_generation:
            self._llm = self._pipeline._generator.llm

        per_question = []
        for tc in test_cases:
            row = self._evaluate_one(tc, skip_generation)
            per_question.append(row)
            print(f"  [{tc['id']}] recall@5={row['recall@5']}", end="")
            if not skip_generation:
                print(
                    f"  faith={row.get('faithfulness', '?')}"
                    f"  rel={row.get('answer_relevancy', '?')}"
                    f"  corr={row.get('answer_correctness', '?')}",
                    end="",
                )
            print()

        aggregate = self._aggregate(per_question, skip_generation)
        return {"aggregate": aggregate, "per_question": per_question}

    def _evaluate_one(self, tc: dict, skip_generation: bool) -> dict:
        result = self._pipeline.query_with_contexts(tc["question"])
        retrieved = result["retrieved_articles"]
        expected = tc["expected_articles"]

        row: dict = {
            "id": tc["id"],
            "question": tc["question"],
            "retrieved_articles": retrieved,
            "expected_articles": expected,
        }
        for k in KS:
            row[f"precision@{k}"] = round(precision_at_k(retrieved, expected, k), 4)
            row[f"recall@{k}"] = round(recall_at_k(retrieved, expected, k), 4)

        if not skip_generation:
            row["answer"] = result["answer"]
            row["faithfulness"] = faithfulness(result["answer"], result["contexts"], self._llm)
            row["answer_relevancy"] = answer_relevancy(
                tc["question"], result["answer"], self._embeddings
            )
            gt = tc.get("ground_truth")
            if gt:
                row["answer_correctness"] = answer_correctness(
                    result["answer"], gt, self._embeddings
                )

        return row

    def _aggregate(self, rows: list[dict], skip_generation: bool) -> dict:
        keys = [f"precision@{k}" for k in KS] + [f"recall@{k}" for k in KS]
        if not skip_generation:
            keys += ["faithfulness", "answer_relevancy", "answer_correctness"]

        agg = {}
        for key in keys:
            values = [r[key] for r in rows if key in r]
            agg[key] = round(sum(values) / len(values), 4) if values else None
        return agg
