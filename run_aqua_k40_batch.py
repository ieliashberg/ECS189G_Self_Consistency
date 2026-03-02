#!/usr/bin/env python
"""
Run GPT-3.5-Turbo on AQUA with self-consistency (k=40) via OpenAI Batch API.

Usage:
    python run_aqua_k40_batch.py                  # submit + wait + evaluate
    python run_aqua_k40_batch.py --submit-only     # submit and print batch ID
    python run_aqua_k40_batch.py --resume BATCH_ID  # resume a previously submitted batch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline_io import load_openai_key_from_envfile, write_results
from pipeline_data import load_benchmark_dataset
from prompts import BenchmarkType
from batch_runner import BatchPipeline, evaluate_batch_results, estimate_cost


MODEL = "gpt-3.5-turbo"
BENCHMARK = BenchmarkType.AQUA
K = 40
TEMPERATURE = 0.7
OUTPUT_CSV = Path("results/aqua_k40_gpt35_batch.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--submit-only", action="store_true",
                       help="Submit the batch and exit without waiting.")
    group.add_argument("--resume", metavar="BATCH_ID",
                       help="Resume waiting on & evaluating a previously submitted batch.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset to N samples (for testing).")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status polls (default: 30).")
    args = parser.parse_args()

    load_openai_key_from_envfile()
    bp = BatchPipeline()

    # Load dataset
    dataset = load_benchmark_dataset(BENCHMARK, max_samples=args.max_samples)
    print(f"Dataset: {BENCHMARK.value} — {len(dataset)} questions")

    # Cost estimate
    est = estimate_cost(dataset, BENCHMARK, n=K)
    print(f"Estimated batch cost: ${est['batch_cost_usd']}")

    if args.resume:
        batch_id = args.resume
        print(f"Resuming batch: {batch_id}")
    else:
        batch_id = bp.submit_batch(
            model=MODEL,
            benchmark=BENCHMARK,
            dataset=dataset,
            temperature=TEMPERATURE,
            n=K,
        )
        if args.submit_only:
            print(f"\nBatch submitted: {batch_id}")
            print(f"Resume later with: python {sys.argv[0]} --resume {batch_id}")
            return

    # Wait for completion
    bp.wait_for_batch(batch_id, poll_interval=args.poll_interval)

    # Download and evaluate
    batch_responses = bp.download_results(batch_id)
    results = evaluate_batch_results(
        batch_responses=batch_responses,
        model=MODEL,
        benchmark=BENCHMARK,
        dataset=dataset,
        method="self_consistency",
        k=K,
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
