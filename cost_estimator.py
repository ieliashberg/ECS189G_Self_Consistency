from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from pipeline_data import DATASET_ALIASES, load_benchmark_dataset
from pipeline_io import parse_csv_list, parse_k_values
from prompts import BenchmarkType, build_prompt

BATCH_PRICING_PER_MILLION: Dict[str, Dict[str, float]] = {
    "gpt-5-mini": {
        "input": 0.25,
        "cached_input": 0.025,
        "output": 2.00,
    }
}

DEFAULT_BENCHMARKS = ["svamp", "aqua", "gsm8k", "strategy_qa"]
DEFAULT_OUTPUT_TOKENS_BY_BENCHMARK: Dict[str, int] = {
    "svamp": 180,
    "aqua": 180,
    "gsm8k": 220,
    "strategy_qa": 120,
}


@dataclass
class BenchmarkCostEstimate:
    dataset: str
    examples: int
    samples_per_question: int
    requests: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


@dataclass
class CostEstimateSummary:
    model: str
    use_batch_pricing: bool
    input_price_per_million: float
    output_price_per_million: float
    cached_input_price_per_million: float
    cached_input_fraction: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    per_benchmark: List[BenchmarkCostEstimate]


def _is_tiktoken_available() -> bool:
    try:
        import tiktoken  # noqa: F401
    except Exception:
        return False
    return True


def _estimate_tokens(text: str, model: str, prefer_tiktoken: bool = True) -> int:
    if prefer_tiktoken and _is_tiktoken_available():
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))

    # Reasonable fallback for rough estimates when tokenizer is unavailable.
    return max(1, math.ceil(len(text) / 4))


def _samples_per_question(methods: Iterable[str], k_values: Iterable[int]) -> int:
    methods_set = set(methods)
    k_list = [int(k) for k in k_values]
    total = 0
    if "greedy" in methods_set:
        total += 1
    if "self_consistency" in methods_set:
        sc_k = [k for k in k_list if k > 1]
        if sc_k:
            total += max(sc_k)
    return total


def _pricing_for_model(model: str, use_batch_pricing: bool) -> Dict[str, float]:
    if model not in BATCH_PRICING_PER_MILLION:
        raise ValueError(
            f"No pricing configured for model '{model}'. Add it to BATCH_PRICING_PER_MILLION first."
        )
    batch_prices = BATCH_PRICING_PER_MILLION[model]
    if use_batch_pricing:
        return dict(batch_prices)
    return {
        "input": batch_prices["input"] * 2.0,
        "cached_input": batch_prices["cached_input"] * 2.0,
        "output": batch_prices["output"] * 2.0,
    }


def estimate_pipeline_cost(
    *,
    model: str = "gpt-5-mini",
    datasets: Optional[Iterable[str]] = None,
    methods: Optional[Iterable[str]] = None,
    k_values: Optional[Iterable[int]] = None,
    max_samples: Optional[int] = None,
    use_batch_pricing: bool = True,
    cached_input_fraction: float = 0.0,
    output_tokens_by_benchmark: Optional[Dict[str, int]] = None,
    prefer_tiktoken: bool = True,
) -> CostEstimateSummary:
    if cached_input_fraction < 0 or cached_input_fraction > 1:
        raise ValueError("cached_input_fraction must be in [0, 1].")

    dataset_keys = list(datasets) if datasets is not None else list(DEFAULT_BENCHMARKS)
    methods_list = list(methods) if methods is not None else ["greedy", "self_consistency"]
    k_list = [int(k) for k in k_values] if k_values is not None else [1, 5, 10]
    samples_per_q = _samples_per_question(methods_list, k_list)
    if samples_per_q <= 0:
        raise ValueError("No requests would be made. Check methods and k_values.")

    output_tokens_map = dict(DEFAULT_OUTPUT_TOKENS_BY_BENCHMARK)
    if output_tokens_by_benchmark:
        output_tokens_map.update({k: int(v) for k, v in output_tokens_by_benchmark.items()})

    prices = _pricing_for_model(model, use_batch_pricing=use_batch_pricing)
    per_benchmark: List[BenchmarkCostEstimate] = []

    total_requests = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    for dataset_key in dataset_keys:
        if dataset_key not in DATASET_ALIASES:
            raise ValueError(f"Unknown dataset alias '{dataset_key}'. Valid: {sorted(DATASET_ALIASES.keys())}")
        benchmark: BenchmarkType = DATASET_ALIASES[dataset_key]
        dataset = load_benchmark_dataset(benchmark, max_samples=max_samples)
        examples = len(dataset)

        prompt_tokens_sum = 0
        for example in dataset:
            prompt = build_prompt(example["question"], benchmark, cot=True, model_name=model)
            prompt_tokens_sum += _estimate_tokens(prompt, model, prefer_tiktoken=prefer_tiktoken)

        requests = examples * samples_per_q
        input_tokens = prompt_tokens_sum * samples_per_q
        output_tokens = requests * output_tokens_map[dataset_key]

        cached_input_tokens = int(input_tokens * cached_input_fraction)
        uncached_input_tokens = input_tokens - cached_input_tokens
        input_cost = (
            (uncached_input_tokens / 1_000_000.0) * prices["input"]
            + (cached_input_tokens / 1_000_000.0) * prices["cached_input"]
        )
        output_cost = (output_tokens / 1_000_000.0) * prices["output"]
        estimated_cost = input_cost + output_cost

        per_benchmark.append(
            BenchmarkCostEstimate(
                dataset=dataset_key,
                examples=examples,
                samples_per_question=samples_per_q,
                requests=requests,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost,
            )
        )
        total_requests += requests
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost += estimated_cost

    return CostEstimateSummary(
        model=model,
        use_batch_pricing=use_batch_pricing,
        input_price_per_million=prices["input"],
        output_price_per_million=prices["output"],
        cached_input_price_per_million=prices["cached_input"],
        cached_input_fraction=cached_input_fraction,
        total_requests=total_requests,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_estimated_cost_usd=total_cost,
        per_benchmark=per_benchmark,
    )


def _print_summary(summary: CostEstimateSummary) -> None:
    print("\n=== Cost Estimate ===")
    print(f"Model: {summary.model}")
    print(f"Batch pricing: {summary.use_batch_pricing}")
    print(
        "Rates per 1M tokens: "
        f"input=${summary.input_price_per_million:.3f}, "
        f"cached_input=${summary.cached_input_price_per_million:.3f}, "
        f"output=${summary.output_price_per_million:.3f}"
    )
    print(f"Cached input fraction: {summary.cached_input_fraction:.2%}")
    print("")
    for row in summary.per_benchmark:
        print(
            f"{row.dataset:12s} | examples={row.examples:<5d} | requests={row.requests:<7d} | "
            f"in_tok={row.input_tokens:<10d} | out_tok={row.output_tokens:<10d} | "
            f"cost=${row.estimated_cost_usd:.4f}"
        )
    print("")
    print(
        f"TOTAL           | requests={summary.total_requests:<7d} | "
        f"in_tok={summary.total_input_tokens:<10d} | out_tok={summary.total_output_tokens:<10d} | "
        f"cost=${summary.total_estimated_cost_usd:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate batch cost for benchmark runs.")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--datasets", default="svamp,aqua,gsm8k,strategy_qa")
    parser.add_argument("--methods", default="greedy,self_consistency")
    parser.add_argument("--k-values", default="1,5,10")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-batch-pricing", action="store_true")
    parser.add_argument("--cached-input-fraction", type=float, default=0.0)
    parser.add_argument("--no-tiktoken", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv_list(args.datasets)
    methods = parse_csv_list(args.methods)
    k_values = parse_k_values(args.k_values)

    summary = estimate_pipeline_cost(
        model=args.model,
        datasets=datasets,
        methods=methods,
        k_values=k_values,
        max_samples=args.max_samples,
        use_batch_pricing=not args.no_batch_pricing,
        cached_input_fraction=args.cached_input_fraction,
        prefer_tiktoken=not args.no_tiktoken,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
