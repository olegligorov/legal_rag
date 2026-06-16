"""Run baseline `/api/query` and the agent on the same dataset, side-by-side.

Usage (from backend/):
    # Full run, both pipelines, generation metrics on:
    python scripts/run_head_to_head.py

    # Skip the LLM-judge metrics (faithfulness/correctness):
    python scripts/run_head_to_head.py --skip-generation

    # Limit to N questions for quick smoke runs:
    python scripts/run_head_to_head.py --limit 3

    # Use a different dataset (e.g. multihop):
    python scripts/run_head_to_head.py --dataset evaluation/dataset_multihop.json

The MCP server must be running before invocation; the agent connects to it
over SSE. The baseline pipeline runs in-process and is unaffected.

Output: ``results/head_to_head_<timestamp>.json`` with per-question rows
for both paths, an aggregate {mean, median, p95, min, max, stdev, n} per
metric, deltas between aggregates, and a ``config`` block that records
every knob (model, temperature, retrieval params, git SHA, dataset hash)
so the run is reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Allow imports from backend/
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    AGENT_MAX_TOOL_CALLS,
    AGENT_MODEL,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    CLAUDE_MODEL,
    DATA_PATH,
    EMBEDDING_MODEL,
    LLM_TEMPERATURE,
    MCP_SERVER_URL,
    RERANKER_MODEL,
    RERANKER_SCORE_THRESHOLD,
    RERANKER_TOP_N,
)
from evaluation.agent_evaluator import AgentEvaluator
from evaluation.aggregation import deltas
from evaluation.evaluator import DATASET_PATH as DEFAULT_DATASET_PATH, Evaluator
from models.rag_pipeline import RAGPipeline


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
        )
        return out.decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=2
        )
        return bool(out.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _build_config_envelope(dataset_path: Path) -> dict:
    """Capture every knob that affects the run for the result JSON."""
    return {
        "claude_model": CLAUDE_MODEL,
        "agent_model": AGENT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "agent_max_tool_calls": AGENT_MAX_TOOL_CALLS,
        "reranker_top_n": RERANKER_TOP_N,
        "reranker_score_threshold": RERANKER_SCORE_THRESHOLD,
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
        "mcp_server_url": MCP_SERVER_URL,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _file_sha256(dataset_path) if dataset_path.exists() else None,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _print_summary(deltas_table: dict, agent_only: dict, baseline_only: dict) -> None:
    print("\n=== Head-to-Head: Baseline vs Agent ===")
    print(f"{'Metric':<22} {'Baseline mean':>14} {'Agent mean':>12} {'Δ mean':>10}")
    print("-" * 60)
    for metric, values in deltas_table.items():
        b_mean = values["baseline"].get("mean")
        a_mean = values["agent"].get("mean")
        b = "—" if b_mean is None else f"{b_mean:.4f}"
        a = "—" if a_mean is None else f"{a_mean:.4f}"
        d = (
            "—"
            if values["delta_mean"] is None
            else f"{values['delta_mean']:+.4f}"
        )
        print(f"{metric:<22} {b:>14} {a:>12} {d:>10}")

    if agent_only:
        print("\n=== Agent-only metrics (mean) ===")
        for k, v in agent_only.items():
            mean = v.get("mean")
            v_str = "—" if mean is None else f"{mean:.4f}"
            print(f"  {k:<22} {v_str}  (n={v.get('n', 0)})")

    if baseline_only:
        print("\n=== Baseline-only metrics (mean) ===")
        for k, v in baseline_only.items():
            mean = v.get("mean")
            v_str = "—" if mean is None else f"{mean:.4f}"
            print(f"  {k:<22} {v_str}  (n={v.get('n', 0)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head: baseline RAG vs agent.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the eval JSON dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N questions (smoke runs).",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip the LLM-judge generation metrics (faithfulness/relevancy/correctness).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to results/head_to_head_<timestamp>.json.",
    )
    args = parser.parse_args()

    original_dataset = Path(args.dataset)
    print(f"Dataset: {original_dataset} (limit={args.limit})", flush=True)

    pipeline = RAGPipeline(data_directory=DATA_PATH)

    # Load the dataset once and slice in memory — no temp files. The hash
    # in the config envelope is the FULL dataset's hash; ``limit`` is
    # captured separately and applied here.
    with open(original_dataset, encoding="utf-8") as f:
        all_cases = json.load(f)
    test_cases = all_cases if args.limit is None else all_cases[: args.limit]

    config = _build_config_envelope(original_dataset)

    print("\n--- Running baseline ---", flush=True)
    baseline = Evaluator(pipeline, test_cases=test_cases).run(
        skip_generation=args.skip_generation
    )

    print("\n--- Running agent ---", flush=True)
    agent = AgentEvaluator(pipeline, test_cases=test_cases).run(
        skip_generation=args.skip_generation
    )

    deltas_table = deltas(baseline["aggregate"], agent["aggregate"])
    agent_only_keys = set(agent["aggregate"]) - set(baseline["aggregate"])
    baseline_only_keys = set(baseline["aggregate"]) - set(agent["aggregate"])
    agent_only = {k: agent["aggregate"][k] for k in sorted(agent_only_keys)}
    baseline_only = {k: baseline["aggregate"][k] for k in sorted(baseline_only_keys)}

    _print_summary(deltas_table, agent_only, baseline_only)

    out: dict = {
        "config": config,
        "limit": args.limit,
        "skip_generation": args.skip_generation,
        # Tells thesis-charting code which retrieval columns are
        # rank-aware. The agent's first-appearance ordering is NOT a
        # rerank, so its precision@k is omitted; set_recall@k is
        # rank-agnostic and the apples-to-apples comparison.
        "retrieval_notes": {
            "baseline": {
                "rank_aware": True,
                "metric_columns": ["precision@3", "precision@5", "recall@3", "recall@5"],
            },
            "agent": {
                "rank_aware": False,
                "metric_columns": ["set_recall@3", "set_recall@5"],
                "note": (
                    "Agent sources are ordered by first appearance across tool calls, "
                    "not by a relevance score. Use set_recall@k for the head-to-head; "
                    "recall@k is reported only for diagnostic continuity."
                ),
            },
        },
        "baseline": baseline,
        "agent": agent,
        "deltas": deltas_table,
        "agent_only": agent_only,
        "baseline_only": baseline_only,
    }

    out_path = args.output
    if out_path is None:
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"head_to_head_{ts}.json"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
