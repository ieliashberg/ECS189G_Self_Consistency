#!/usr/bin/env python
"""
Run UL2 (google/ul2) on AQUA with self-consistency (k=40) via local HuggingFace inference.

No Batch API — UL2 runs locally on GPU. Responses are disk-cached for free re-runs.

Usage:
    python run_aqua_k40_ul2.py
    python run_aqua_k40_ul2.py --max-samples 50   # quick test on 50 questions
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_data import load_benchmark_dataset
from pipeline_eval import evaluate
from pipeline_io import write_results
from prompts import BenchmarkType


MODEL = "google/ul2"
BENCHMARK = BenchmarkType.AQUA
K = 40
TEMPERATURE = 0.7
OUTPUT_CSV = Path("results/aqua_k40_ul2.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset to N samples (for testing).")
    args = parser.parse_args()

    # Load dataset
    dataset = load_benchmark_dataset(BENCHMARK, max_samples=args.max_samples)
    print(f"Dataset: {BENCHMARK.value} — {len(dataset)} questions")
    print(f"Model:   {MODEL}")
    print(f"Method:  self_consistency, k={K}")

    # Run evaluation (local inference, disk-cached)
    results = evaluate(
        model=MODEL,
        benchmark=BENCHMARK,
        dataset=dataset,
        method="self_consistency",
        k=K,
        self_consistency_temperature=TEMPERATURE,
    )

    print(f"\n=== Results ===")
    print(f"Model:    {results.model}")
    print(f"Dataset:  {results.dataset}")
    print(f"Method:   {results.method}")
    print(f"k:        {results.k}")
    print(f"Accuracy: {results.accuracy:.4f} ({results.correct}/{results.total})")
    print(f"Parsed:   {results.parsed}/{results.total}")

    write_results([results], output_csv=OUTPUT_CSV)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
