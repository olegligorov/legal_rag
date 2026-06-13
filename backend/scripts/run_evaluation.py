"""Run RAG evaluation.

Usage (from backend/):
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --skip-generation
    python scripts/run_evaluation.py --output results/my_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Allow importing backend modules when run from backend/
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rag_pipeline import RAGPipeline
from evaluation.evaluator import Evaluator, DATASET_PATH
from config import DATA_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the legal RAG pipeline.")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only compute retrieval metrics (no RAGAS, no LLM calls).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results. Defaults to results/eval_<timestamp>.json.",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline(data_directory=DATA_PATH)
    evaluator = Evaluator(pipeline)

    with open(DATASET_PATH, encoding="utf-8") as f:
        n_cases = len(json.load(f))

    print(f"\nRunning evaluation on {n_cases} test cases...")
    if args.skip_generation:
        print("(Generation metrics skipped)\n")

    results = evaluator.run(skip_generation=args.skip_generation)

    # Print summary table
    print("\n=== Aggregate Metrics ===")
    for metric, value in results["aggregate"].items():
        bar = ""
        if value is not None:
            filled = int(value * 20)
            bar = f" [{'█' * filled}{'░' * (20 - filled)}]"
        print(f"  {metric:<20} {value:.4f}{bar}")

    # Save results
    out_path = args.output
    if out_path is None:
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"eval_{ts}.json"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
