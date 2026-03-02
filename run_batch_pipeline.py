from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from batch_runner import BatchPipeline
from pipeline_data import DATASET_ALIASES, load_benchmark_dataset
from pipeline_eval import extract_answer, grade_answer
from pipeline_io import load_openai_key_from_envfile, write_results
from pipeline_types import EvalStats, PipelineConfig


def _majority_vote_parsed(parsed_answers: Iterable[Optional[str]]) -> Optional[str]:
    valid = [answer for answer in parsed_answers if answer is not None]
    if not valid:
        return None
    counts = Counter(valid)
    top_count = max(counts.values())
    for answer in valid:
        if counts[answer] == top_count:
            return answer
    return None


def run_benchmark_pipeline_batch(
    config: PipelineConfig,
    *,
    max_output_tokens: int = 256,
    batch_cache_dir: str = "results/batch",
) -> list[EvalStats]:
    if config.model == "gpt-3.5-turbo":
        raise ValueError(
            "This helper is intended for /v1/responses batch models. "
            "Use gpt-5-mini (or another responses model) for Batch API runs."
        )

    load_openai_key_from_envfile()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to the environment or .env file.")

    config.validate(valid_datasets=DATASET_ALIASES.keys())
    batch_pipeline = BatchPipeline(cache_dir=batch_cache_dir)
    rows: list[EvalStats] = []

    loaded_datasets: dict = {}
    for key in config.datasets:
        benchmark = DATASET_ALIASES[key]
        loaded_datasets[benchmark] = load_benchmark_dataset(benchmark, max_samples=config.max_samples)

    for benchmark, dataset in loaded_datasets.items():
        print(f"Dataset '{benchmark.value}': {len(dataset)} examples loaded")

    for benchmark, dataset in loaded_datasets.items():
        for method in config.methods:
            if method == "greedy":
                if 1 not in config.k_values:
                    print(f"Skipping greedy for {benchmark.value}: k=1 not in k_values={config.k_values}")
                    continue
                k_values = [1]
                required_samples = 1
                temperature = 0.0
            elif method == "self_consistency":
                k_values = [k for k in config.k_values if k > 1]
                if not k_values:
                    print(
                        f"Skipping self_consistency for {benchmark.value}: "
                        f"no k>1 values in k_values={config.k_values}"
                    )
                    continue
                required_samples = max(k_values)
                temperature = config.self_consistency_temperature
            else:
                raise ValueError(f"Unsupported method: {method}")

            print(
                f"[batch] benchmark={benchmark.value} method={method} "
                f"required_samples={required_samples}"
            )
            by_question_idx = batch_pipeline.run_generation_batch(
                model=config.model,
                benchmark=benchmark,
                dataset=dataset,
                temperature=temperature,
                n=required_samples,
                max_tokens=max_output_tokens,
            )

            for k in sorted(k_values):
                correct = 0
                parsed = 0
                total = len(dataset)
                for idx, example in enumerate(dataset):
                    responses = by_question_idx.get(idx, [])[:k]
                    parsed_answers = [extract_answer(response, benchmark) for response in responses]
                    if method == "greedy":
                        predicted = parsed_answers[0] if parsed_answers else None
                    else:
                        predicted = _majority_vote_parsed(parsed_answers)

                    if predicted is not None:
                        parsed += 1
                    if grade_answer(predicted, example["final_answer"], benchmark):
                        correct += 1

                accuracy = correct / total if total else 0.0
                rows.append(
                    EvalStats(
                        model=config.model,
                        dataset=benchmark.value,
                        method=method,
                        k=k,
                        correct=correct,
                        total=total,
                        parsed=parsed,
                        accuracy=accuracy,
                    )
                )

    write_results(rows, output_csv=Path(config.output_csv))
    return rows
