from __future__ import annotations

import argparse

from pipeline_io import print_results_summary
from run_pipeline import build_pipeline_config, run_benchmark_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all four benchmarks on gpt-5-mini with cache-aware self-consistency. "
            "Self-consistency generates up to max k once (default k=40) and reuses "
            "those samples for smaller k values."
        )
    )
    parser.add_argument(
        "--cache-path",
        default="results/cache/gpt5mini_generation_cache.jsonl",
        help="JSONL cache path used for generation reuse across runs.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/gpt5mini_all_benchmarks.csv",
        help="Output CSV path for aggregated benchmark rows.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per benchmark dataset (default: full dataset).",
    )
    parser.add_argument(
        "--self-consistency-temperature",
        type=float,
        default=0.7,
        help="Temperature used for self-consistency sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = build_pipeline_config(
        model="gpt-5-mini",
        datasets=["svamp", "aqua", "gsm8k", "strategy_qa"],
        methods=["greedy", "self_consistency"],
        k_values=[1, 5, 10, 20, 40],
        max_samples=args.max_samples,
        self_consistency_temperature=args.self_consistency_temperature,
        cache_path=args.cache_path,
        output_csv=args.output_csv,
    )

    rows = run_benchmark_pipeline(config)
    print_results_summary(rows, config.output_csv)


if __name__ == "__main__":
    main()
