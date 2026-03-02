#!/usr/bin/env python
"""
Run GPT-5-mini on ALL benchmarks (AQUA, GSM8K, SVAMP, StrategyQA) with
self-consistency (k=40) via OpenAI Batch API.

Submits one batch per benchmark, waits for all, evaluates, and saves
each to a separate CSV.

Usage:
    python run_all_k40_gpt5mini_batch.py                    # full run
    python run_all_k40_gpt5mini_batch.py --submit-only      # submit all and exit
    python run_all_k40_gpt5mini_batch.py --max-samples 50   # quick test
    python run_all_k40_gpt5mini_batch.py --benchmarks aqua,gsm8k  # subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline_io import load_openai_key_from_envfile, write_results
from pipeline_data import load_benchmark_dataset
from prompts import BenchmarkType
from batch_runner import BatchPipeline, evaluate_batch_results, estimate_cost


MODEL = "gpt-5-mini"
K = 40
TEMPERATURE = 0.7

ALL_BENCHMARKS = {
    "aqua": BenchmarkType.AQUA,
    "gsm8k": BenchmarkType.GSM8K,
    "svamp": BenchmarkType.SVAMP,
    "strategy_qa": BenchmarkType.STRATEGY_QA,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--submit-only", action="store_true",
                       help="Submit all batches and exit without waiting.")
    group.add_argument("--resume", nargs="+", metavar="BENCH:BATCH_ID",
                       help="Resume batches, e.g. --resume aqua:batch_abc gsm8k:batch_def")
    parser.add_argument("--benchmarks", default=",".join(ALL_BENCHMARKS.keys()),
                        help="Comma-separated benchmarks to run (default: all).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit each dataset to N samples (for testing).")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status polls (default: 30).")
    args = parser.parse_args()

    load_openai_key_from_envfile()
    bp = BatchPipeline()

    # Parse which benchmarks to run
    bench_names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    benchmarks = {}
    for name in bench_names:
        if name not in ALL_BENCHMARKS:
            print(f"Unknown benchmark: {name}. Valid: {list(ALL_BENCHMARKS.keys())}")
            sys.exit(1)
        benchmarks[name] = ALL_BENCHMARKS[name]

    # Parse resume mapping if provided
    resume_map: dict[str, str] = {}
    if args.resume:
        for item in args.resume:
            if ":" not in item:
                print(f"Invalid --resume format: {item}. Expected BENCH:BATCH_ID")
                sys.exit(1)
            bench, batch_id = item.split(":", 1)
            resume_map[bench] = batch_id

    # Load datasets
    datasets = {}
    for name, benchmark in benchmarks.items():
        datasets[name] = load_benchmark_dataset(benchmark, max_samples=args.max_samples)
        print(f"  {name}: {len(datasets[name])} questions")

    # Cost estimates
    print(f"\n--- Cost estimates (batch API, 50% off) ---")
    total_cost = 0.0
    for name, benchmark in benchmarks.items():
        est = estimate_cost(datasets[name], benchmark, n=K)
        total_cost += est["batch_cost_usd"]
        print(f"  {name}: ${est['batch_cost_usd']}")
    print(f"  TOTAL: ${total_cost:.4f}\n")

    # Submit batches
    batch_ids: dict[str, str] = {}
    for name, benchmark in benchmarks.items():
        if name in resume_map:
            batch_ids[name] = resume_map[name]
            print(f"[{name}] Resuming batch: {resume_map[name]}")
        else:
            batch_id = bp.submit_batch(
                model=MODEL,
                benchmark=benchmark,
                dataset=datasets[name],
                temperature=TEMPERATURE,
                n=K,
            )
            batch_ids[name] = batch_id

    if args.submit_only:
        print(f"\n=== All batches submitted ===")
        resume_args = " ".join(f"{name}:{bid}" for name, bid in batch_ids.items())
        print(f"Resume later with:\n  python {sys.argv[0]} --resume {resume_args}")
        return

    # Wait, download, evaluate each
    for name, benchmark in benchmarks.items():
        batch_id = batch_ids[name]
        print(f"\n--- Waiting for {name} (batch {batch_id}) ---")
        bp.wait_for_batch(batch_id, poll_interval=args.poll_interval)

        batch_responses = bp.download_results(batch_id)
        results = evaluate_batch_results(
            batch_responses=batch_responses,
            model=MODEL,
            benchmark=benchmark,
            dataset=datasets[name],
            method="self_consistency",
            k=K,
        )

        print(f"\n=== {name} Results ===")
        print(f"  Model:    {results.model}")
        print(f"  Dataset:  {results.dataset}")
        print(f"  Method:   {results.method}")
        print(f"  k:        {results.k}")
        print(f"  Accuracy: {results.accuracy:.4f} ({results.correct}/{results.total})")
        print(f"  Parsed:   {results.parsed}/{results.total}")

        output_csv = Path(f"results/{name}_k{K}_gpt5mini_batch.csv")
        write_results([results], output_csv=output_csv)
        print(f"  Saved to {output_csv}")

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
