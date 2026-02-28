from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class EvalStats:
    model: str
    dataset: str
    method: str
    k: int
    correct: int
    total: int
    parsed: int
    accuracy: float


@dataclass
class PipelineConfig:
    model: str = "gpt-3.5-turbo"
    datasets: list[str] = field(default_factory=lambda: ["svamp", "aqua", "gsm8k", "strategy_qa"])
    methods: list[str] = field(default_factory=lambda: ["greedy", "self_consistency"])
    k_values: list[int] = field(default_factory=lambda: [5])
    max_samples: Optional[int] = None
    self_consistency_temperature: float = 0.7
    output_csv: str = "results/first_baseline.csv"

    def validate(self, valid_datasets: Iterable[str]) -> None:
        valid_dataset_set = set(valid_datasets)
        unknown_datasets = [key for key in self.datasets if key not in valid_dataset_set]
        if unknown_datasets:
            raise ValueError(f"Unknown datasets: {unknown_datasets}. Valid: {sorted(valid_dataset_set)}")

        valid_methods = {"greedy", "self_consistency"}
        invalid_methods = [method for method in self.methods if method not in valid_methods]
        if invalid_methods:
            raise ValueError(f"Unknown methods: {invalid_methods}. Valid: {sorted(valid_methods)}")

        if any(k <= 0 for k in self.k_values):
            raise ValueError("All k values must be positive integers.")

        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be None or a positive integer.")
