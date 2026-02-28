from __future__ import annotations

import re
import time
from collections import Counter
from typing import Iterable, List, Optional

from datasets import Dataset
from tqdm import tqdm

from model_wrapper import run_model
from pipeline_types import EvalStats
from prompts import BenchmarkType, build_prompt


def _normalize_numeric(text: str) -> Optional[str]:
    clean = text.replace(",", "").strip()
    try:
        return str(float(clean))
    except ValueError:
        return None


def normalize_bool(value: str) -> str:
    token = value.strip().lower()
    if token == "yes":
        return "true"
    if token == "no":
        return "false"
    return token


def extract_answer(response: str, benchmark: BenchmarkType) -> Optional[str]:
    if not response:
        return None

    if benchmark in (BenchmarkType.SVAMP, BenchmarkType.GSM8K):
        direct = re.search(r"[Tt]he answer is[^-\d]*([-+]?\d[\d,]*(?:\.\d+)?)", response)
        if direct:
            return _normalize_numeric(direct.group(1))
        hashes = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", response)
        if hashes:
            return _normalize_numeric(hashes.group(1))
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", response)
        if numbers:
            return _normalize_numeric(numbers[-1])
        return None

    if benchmark == BenchmarkType.AQUA:
        direct = re.search(r"[Tt]he answer is\s*\(?([A-Ea-e])\)?", response)
        if direct:
            return direct.group(1).upper()
        trailing = re.findall(r"\b([A-Ea-e])\b", response)
        if trailing:
            return trailing[-1].upper()
        return None

    if benchmark == BenchmarkType.STRATEGY_QA:
        direct = re.search(r"[Tt]he answer is\s*(yes|no|true|false)", response)
        if direct:
            return normalize_bool(direct.group(1))
        trailing = re.findall(r"\b(yes|no|true|false)\b", response.lower())
        if trailing:
            return normalize_bool(trailing[-1])
        return None

    return None


def grade_answer(predicted: Optional[str], gold: str, benchmark: BenchmarkType) -> bool:
    if predicted is None:
        return False

    if benchmark in (BenchmarkType.SVAMP, BenchmarkType.GSM8K):
        pred_num = _normalize_numeric(predicted)
        gold_num = _normalize_numeric(gold)
        return pred_num is not None and gold_num is not None and float(pred_num) == float(gold_num)

    if benchmark == BenchmarkType.AQUA:
        return predicted.strip().upper() == gold.strip().upper()

    if benchmark == BenchmarkType.STRATEGY_QA:
        return normalize_bool(predicted) == normalize_bool(gold)

    return predicted.strip().lower() == gold.strip().lower()


def majority_vote(responses: Iterable[str], benchmark: BenchmarkType) -> Optional[str]:
    parsed = [extract_answer(resp, benchmark) for resp in responses]
    valid = [p for p in parsed if p is not None]
    if not valid:
        return None

    counts = Counter(valid)
    top_count = max(counts.values())
    for prediction in valid:
        if counts[prediction] == top_count:
            return prediction
    return None


def query_model(
    model: str,
    prompt: str,
    temperature: float,
    num_samples: int,
    max_retries: int = 3,
) -> List[str]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return run_model(
                model_name=model,
                prompt=prompt,
                temperature=temperature,
                top_p=1.0,
                num_samples=num_samples,
            )
        except Exception as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Model request failed after {max_retries} retries: {last_error}")


def evaluate(
    model: str,
    benchmark: BenchmarkType,
    dataset: Dataset,
    method: str,
    k: int,
    self_consistency_temperature: float,
) -> EvalStats:
    correct = 0
    parsed = 0
    total = len(dataset)
    details = []

    temperature = 0.0 if method == "greedy" else self_consistency_temperature
    n_samples = 1 if method == "greedy" else k

    for example in tqdm(dataset, desc=f"{benchmark.value} | {method} | k={k}"):
        prompt = build_prompt(example["question"], benchmark, cot=True)
        responses = query_model(
            model=model,
            prompt=prompt,
            temperature=temperature,
            num_samples=n_samples,
        )

        if method == "greedy":
            predicted = extract_answer(responses[0], benchmark)
        else:
            predicted = majority_vote(responses, benchmark)

        is_correct = grade_answer(predicted, example["final_answer"], benchmark)
        if predicted is not None:
            parsed += 1
        if is_correct:
            correct += 1

        details.append({
            "question": example["question"],
            "gold_answer": example["final_answer"],
            "predicted": predicted,
            "is_correct": is_correct,
            "responses": responses,
        })

    accuracy = correct / total if total else 0.0
    return EvalStats(
        model=model,
        dataset=benchmark.value,
        method=method,
        k=k,
        correct=correct,
        total=total,
        parsed=parsed,
        accuracy=accuracy,
        details=details,
    )
