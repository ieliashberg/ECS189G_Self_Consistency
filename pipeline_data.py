from __future__ import annotations

from typing import Dict, Optional

from datasets import Dataset, load_dataset

from prompts import BenchmarkType


DATASET_ALIASES: Dict[str, BenchmarkType] = {
    "svamp": BenchmarkType.SVAMP,
    "aqua": BenchmarkType.AQUA,
    "gsm8k": BenchmarkType.GSM8K,
    "strategy_qa": BenchmarkType.STRATEGY_QA,
}


def parse_svamp(example: dict) -> dict:
    full_question = f"{example['Body']} {example['Question']}"
    return {"question": full_question, "final_answer": str(example["Answer"])}


def parse_aqua(example: dict) -> dict:
    options = "\n".join(example["options"])
    return {
        "question": f"{example['question']}\nOptions:\n{options}",
        "final_answer": str(example["correct"]).strip().upper(),
    }


def parse_gsm8k(example: dict) -> dict:
    split = example["answer"].split("#### ")
    return {"question": example["question"], "final_answer": split[-1].strip()}


def parse_strategy_qa(example: dict) -> dict:
    answer = "true" if bool(example["answer"]) else "false"
    return {"question": example["question"], "final_answer": answer}


def load_benchmark_dataset(benchmark: BenchmarkType, max_samples: Optional[int] = None) -> Dataset:
    if benchmark == BenchmarkType.SVAMP:
        dataset = load_dataset("tongyx361/svamp", split="test").map(parse_svamp)
    elif benchmark == BenchmarkType.AQUA:
        dataset = load_dataset("deepmind/aqua_rat", split="test").map(parse_aqua)
    elif benchmark == BenchmarkType.GSM8K:
        dataset = load_dataset("openai/gsm8k", "main", split="test").map(parse_gsm8k)
    elif benchmark == BenchmarkType.STRATEGY_QA:
        dataset = load_dataset("ChilleD/StrategyQA", split="test").map(parse_strategy_qa)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    if max_samples is not None:
        limit = min(max_samples, len(dataset))
        dataset = dataset.select(range(limit))

    return dataset
