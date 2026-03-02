#!/usr/bin/env python
"""
Estimate greedy (k=1) accuracy from existing k=40 batch results.

For each question, randomly picks ONE of the 40 sampled responses,
extracts the answer, and checks correctness. Prints final accuracy.

Usage:
    python estimate_k1_from_batch.py
    python estimate_k1_from_batch.py --csv results/aqua_k40_gpt35_batch.csv
    python estimate_k1_from_batch.py --trials 100   # average over 100 random draws
"""
from __future__ import annotations

import argparse
import csv
import random

from pipeline_eval import extract_answer, grade_answer
from prompts import BenchmarkType


BENCHMARK_MAP = {
    "aqua": BenchmarkType.AQUA,
    "gsm8k": BenchmarkType.GSM8K,
    "svamp": BenchmarkType.SVAMP,
    "strategy_qa": BenchmarkType.STRATEGY_QA,
}


def load_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def simulate_k1(rows: list[dict], seed: int | None = None) -> tuple[int, int]:
    """Pick one random response per question and score it. Returns (correct, total)."""
    rng = random.Random(seed)
    correct = 0
    total = 0

    for row in rows:
        responses = row["responses"].split("|||")
        gold = row["gold_answer"]
        benchmark = BENCHMARK_MAP[row["dataset"]]

        # Pick one random response
        chosen = rng.choice(responses)
        predicted = extract_answer(chosen, benchmark)
        if grade_answer(predicted, gold, benchmark):
            correct += 1
        total += 1

    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/aqua_k40_gpt35_batch.csv",
                        help="Path to batch results CSV.")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of random trials to average over (default: 1).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    print(f"Loaded {len(rows)} questions from {args.csv}")
    print(f"Dataset: {rows[0]['dataset']}")
    print(f"Running {args.trials} trial(s)...\n")

    accuracies = []
    for trial in range(args.trials):
        seed = (args.seed + trial) if args.seed is not None else None
        correct, total = simulate_k1(rows, seed=seed)
        acc = correct / total if total else 0.0
        accuracies.append(acc)
        if args.trials <= 10:
            print(f"  Trial {trial + 1}: {acc:.4f} ({correct}/{total})")

    avg_acc = sum(accuracies) / len(accuracies)
    min_acc = min(accuracies)
    max_acc = max(accuracies)

    print(f"\n=== Simulated k=1 Accuracy ===")
    print(f"  Avg:  {avg_acc:.4f}")
    if args.trials > 1:
        print(f"  Min:  {min_acc:.4f}")
        print(f"  Max:  {max_acc:.4f}")
        print(f"  Over: {args.trials} trials")


if __name__ == "__main__":
    main()
