from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import Dataset
from tqdm import tqdm

from pipeline_eval import extract_answer, grade_answer, query_model
from pipeline_types import EvalStats
from prompts import BenchmarkType, build_prompt


CacheIndex = Dict[int, List[dict]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _normalize_loaded_records(records: List[dict]) -> List[dict]:
    dedup: Dict[int, dict] = {}
    for record in records:
        sample_idx = int(record["sample_idx"])
        if sample_idx not in dedup:
            dedup[sample_idx] = record
    return [dedup[idx] for idx in sorted(dedup.keys())]


def load_cache_index(
    cache_path: Path,
    model: str,
    dataset_name: str,
    method: str,
) -> CacheIndex:
    if not cache_path.exists():
        return {}

    index: CacheIndex = {}
    with cache_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if row.get("model") != model:
                continue
            if row.get("dataset") != dataset_name:
                continue
            if row.get("method") != method:
                continue
            question_idx = int(row["question_idx"])
            index.setdefault(question_idx, []).append(row)

    for question_idx, records in list(index.items()):
        index[question_idx] = _normalize_loaded_records(records)
    return index


def append_cache_records(cache_path: Path, records: List[dict]) -> None:
    if not records:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def evaluate_with_cache(
    *,
    model: str,
    benchmark: BenchmarkType,
    dataset: Dataset,
    method: str,
    k_values: List[int],
    self_consistency_temperature: float,
    cache_path: str,
    question_index_offset: int = 0,
    chunk_index: Optional[int] = None,
    num_chunks: Optional[int] = None,
) -> List[EvalStats]:
    if method not in {"greedy", "self_consistency"}:
        raise ValueError(f"Unsupported method for caching: {method}")
    if not k_values:
        return []

    cache_file = Path(cache_path)
    cache_index = load_cache_index(
        cache_path=cache_file,
        model=model,
        dataset_name=benchmark.value,
        method=method,
    )

    required_samples = 1 if method == "greedy" else max(k_values)
    temperature = 0.0 if method == "greedy" else self_consistency_temperature
    per_question_state: List[dict] = []
    generated_records_count = 0
    cache_hit_questions = 0

    progress_label = (
        f"{benchmark.value} | cache:{method} | "
        f"required_samples={required_samples}"
    )
    for local_idx, example in enumerate(tqdm(dataset, desc=progress_label)):
        question_idx = question_index_offset + local_idx
        existing_records = list(cache_index.get(question_idx, []))
        existing_records = _normalize_loaded_records(existing_records)
        existing_count = len(existing_records)

        if existing_count >= required_samples:
            cache_hit_questions += 1
        else:
            missing = required_samples - existing_count
            prompt = build_prompt(
                example["question"],
                benchmark,
                cot=True,
                model_name=model,
            )
            responses = query_model(
                model=model,
                prompt=prompt,
                temperature=temperature,
                num_samples=missing,
            )

            new_records: List[dict] = []
            base_sample_idx = existing_count + 1
            for i, response in enumerate(responses):
                sample_idx = base_sample_idx + i
                parsed_answer = extract_answer(response, benchmark)
                new_records.append(
                    {
                        "timestamp_utc": _utc_now_iso(),
                        "model": model,
                        "dataset": benchmark.value,
                        "method": method,
                        "question_idx": question_idx,
                        "sample_idx": sample_idx,
                        "question": example["question"],
                        "gold_answer": example["final_answer"],
                        "raw_response": response,
                        "parsed_answer": parsed_answer,
                        "temperature": temperature,
                        "chunk_index": chunk_index,
                        "num_chunks": num_chunks,
                    }
                )

            append_cache_records(cache_file, new_records)
            generated_records_count += len(new_records)
            existing_records.extend(new_records)
            existing_records = _normalize_loaded_records(existing_records)
            cache_index[question_idx] = existing_records

        per_question_state.append(
            {
                "gold_answer": example["final_answer"],
                "records": existing_records,
            }
        )

    print(
        f"[cache] {benchmark.value} | {method}: "
        f"cache_hits={cache_hit_questions}/{len(dataset)}, "
        f"generated_new_samples={generated_records_count}, "
        f"cache_file={cache_file}"
    )

    rows: List[EvalStats] = []
    for k in sorted(k_values):
        correct = 0
        parsed = 0
        total = len(per_question_state)

        for state in per_question_state:
            selected_records = state["records"][:k]
            parsed_answers = [record.get("parsed_answer") for record in selected_records]
            if method == "greedy":
                predicted = parsed_answers[0] if parsed_answers else None
            else:
                predicted = _majority_vote_parsed(parsed_answers)

            is_correct = grade_answer(predicted, state["gold_answer"], benchmark)
            if predicted is not None:
                parsed += 1
            if is_correct:
                correct += 1

        accuracy = correct / total if total else 0.0
        rows.append(
            EvalStats(
                model=model,
                dataset=benchmark.value,
                method=method,
                k=k,
                correct=correct,
                total=total,
                parsed=parsed,
                accuracy=accuracy,
            )
        )

    return rows
