from __future__ import annotations

import os
from pathlib import Path

from pipeline_data import DATASET_ALIASES, load_benchmark_dataset
from pipeline_eval import evaluate
from pipeline_io import load_openai_key_from_envfile, parse_csv_list, parse_k_values, print_results_summary, write_results
from pipeline_types import EvalStats, PipelineConfig


def build_pipeline_config(
    model: str,
    datasets: str | list[str] = "svamp,aqua,gsm8k,strategy_qa",
    methods: str | list[str] = "greedy,self_consistency",
    k_values: str | list[int] = "5",
    max_samples: int | None = None,
    self_consistency_temperature: float = 0.7,
    output_csv: str = "results/first_baseline.csv",
) -> PipelineConfig:
    dataset_list = parse_csv_list(datasets) if isinstance(datasets, str) else list(datasets)
    method_list = parse_csv_list(methods) if isinstance(methods, str) else list(methods)
    if isinstance(k_values, str):
        k_list = parse_k_values(k_values)
    else:
        k_list = [int(v) for v in k_values]

    config = PipelineConfig(
        model=model,
        datasets=dataset_list,
        methods=method_list,
        k_values=k_list,
        max_samples=max_samples,
        self_consistency_temperature=self_consistency_temperature,
        output_csv=output_csv,
    )
    config.validate(valid_datasets=DATASET_ALIASES.keys())
    return config


def run_benchmark_pipeline(config: PipelineConfig) -> list[EvalStats]:
    # Only OpenAI models require OPENAI_API_KEY.
    if config.model in {"gpt-3.5-turbo", "gpt-5.2"}:
        load_openai_key_from_envfile()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to the environment or .env file.")
    config.validate(valid_datasets=DATASET_ALIASES.keys())

    rows: list[EvalStats] = []

    loaded_datasets = {
        DATASET_ALIASES[key]: load_benchmark_dataset(DATASET_ALIASES[key], max_samples=config.max_samples)
        for key in config.datasets
    }

    for benchmark, dataset in loaded_datasets.items():
        for method in config.methods:
            if method == "greedy":
                rows.append(
                    evaluate(
                        model=config.model,
                        benchmark=benchmark,
                        dataset=dataset,
                        method=method,
                        k=1,
                        self_consistency_temperature=config.self_consistency_temperature,
                    )
                )
            else:
                for k in config.k_values:
                    rows.append(
                        evaluate(
                            model=config.model,
                            benchmark=benchmark,
                            dataset=dataset,
                            method=method,
                            k=k,
                            self_consistency_temperature=config.self_consistency_temperature,
                        )
                    )

    write_results(rows, output_csv=Path(config.output_csv))
    return rows
