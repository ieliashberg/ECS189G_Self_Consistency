"""
OpenAI Batch API runner — 50% cheaper than real-time requests.

Usage:
    from batch_runner import BatchPipeline
    bp = BatchPipeline()

    # 1. Prepare and submit
    batch_id = bp.submit_batch(
        model="gpt-3.5-turbo",
        benchmark=BenchmarkType.GSM8K,
        dataset=dataset,
        temperature=0.7,
        n=5,
        max_tokens=256,
    )

    # 2. Poll / wait (up to 24h, usually minutes for small jobs)
    bp.wait_for_batch(batch_id)

    # 3. Download results
    results = bp.download_results(batch_id)
"""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from datasets import Dataset

from prompts import BenchmarkType, build_prompt


class BatchPipeline:
    def __init__(self, cache_dir: str = "batch_cache"):
        self.client = OpenAI()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _build_requests_jsonl(
        self,
        model: str,
        benchmark: BenchmarkType,
        dataset: Dataset,
        temperature: float,
        n: int,
        max_tokens: int,
    ) -> Path:
        """Build a JSONL file of batch requests.

        Uses a system message for few-shot examples so the shared prefix
        is eligible for OpenAI's automatic prompt caching (50% off cached
        input tokens when prefix >= 1024 tokens).
        """
        from prompts import FEW_SHOT_EXAMPLES

        few_shot = FEW_SHOT_EXAMPLES[benchmark]
        system_msg = (
            "You are a helpful assistant that solves reasoning problems step by step. "
            "Here are some examples:\n\n" + few_shot
        )

        lines: list[str] = []
        for idx, example in enumerate(dataset):
            user_content = f"Q: {example['question']}\nA:"
            request = {
                "custom_id": f"{benchmark.value}-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "n": n,
                },
            }
            lines.append(json.dumps(request))

        # Write JSONL
        content = "\n".join(lines)
        file_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        jsonl_path = self.cache_dir / f"batch_{benchmark.value}_{file_hash}.jsonl"
        jsonl_path.write_text(content)
        print(f"[batch] Wrote {len(lines)} requests to {jsonl_path}")
        return jsonl_path

    def submit_batch(
        self,
        model: str,
        benchmark: BenchmarkType,
        dataset: Dataset,
        temperature: float = 0.7,
        n: int = 5,
        max_tokens: int = 256,
    ) -> str:
        """Prepare JSONL, upload, and submit a batch. Returns batch ID."""
        jsonl_path = self._build_requests_jsonl(
            model=model,
            benchmark=benchmark,
            dataset=dataset,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
        )

        # Upload input file
        with open(jsonl_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")
        print(f"[batch] Uploaded file: {file_obj.id}")

        # Create batch
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"benchmark": benchmark.value, "model": model},
        )
        print(f"[batch] Submitted batch: {batch.id}")

        # Save batch ID for later retrieval
        meta_path = self.cache_dir / f"batch_{batch.id}.json"
        meta_path.write_text(json.dumps({
            "batch_id": batch.id,
            "benchmark": benchmark.value,
            "model": model,
            "n": n,
            "temperature": temperature,
            "num_requests": len(dataset),
        }, indent=2))

        return batch.id

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 30,
        timeout: int = 86400,
    ) -> dict:
        """Poll until batch completes. Returns batch status dict."""
        elapsed = 0
        while elapsed < timeout:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            print(f"[batch] {batch_id} status={status}  "
                  f"completed={batch.request_counts.completed}/{batch.request_counts.total}  "
                  f"failed={batch.request_counts.failed}")

            if status == "completed":
                return batch
            if status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch {batch_id} ended with status: {status}")

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")

    def download_results(self, batch_id: str) -> Dict[str, List[str]]:
        """Download batch results. Returns {custom_id: [response_texts]}."""
        batch = self.client.batches.retrieve(batch_id)
        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch_id} is not completed (status={batch.status})")

        output_file_id = batch.output_file_id
        content = self.client.files.content(output_file_id).text

        results: Dict[str, List[str]] = {}
        for line in content.strip().split("\n"):
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            response_body = obj["response"]["body"]
            texts = [
                (choice["message"]["content"] or "").strip()
                for choice in response_body["choices"]
            ]
            results[custom_id] = texts

        # Cache results locally
        out_path = self.cache_dir / f"results_{batch_id}.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[batch] Saved {len(results)} results to {out_path}")
        return results


def evaluate_batch_results(
    batch_responses: Dict[str, List[str]],
    model: str,
    benchmark: BenchmarkType,
    dataset: Dataset,
    method: str,
    k: int,
) -> "EvalStats":
    """Score batch API responses using the same logic as pipeline_eval.evaluate.

    Parameters
    ----------
    batch_responses : dict
        Mapping of ``custom_id`` (``"{benchmark}-{idx}"``) to a list of
        response texts returned by :meth:`BatchPipeline.download_results`.
    """
    from pipeline_eval import extract_answer, grade_answer, majority_vote
    from pipeline_types import EvalStats

    correct = 0
    parsed = 0
    total = len(dataset)
    details: List[Dict[str, Any]] = []

    for idx, example in enumerate(dataset):
        custom_id = f"{benchmark.value}-{idx}"
        responses = batch_responses.get(custom_id, [])

        if not responses:
            predicted = None
        elif method == "greedy":
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


def estimate_cost(
    dataset: Dataset,
    benchmark: BenchmarkType,
    n: int = 5,
    avg_output_tokens_per_sample: int = 150,
) -> dict:
    """Estimate costs for real-time vs batch API."""
    from prompts import FEW_SHOT_EXAMPLES
    few_shot = FEW_SHOT_EXAMPLES[benchmark]

    # Rough token estimate: ~4 chars per token
    prefix_tokens = len(few_shot) // 4
    avg_question_tokens = 50  # rough estimate
    input_tokens_per_request = prefix_tokens + avg_question_tokens
    output_tokens_per_request = avg_output_tokens_per_sample * n
    num_questions = len(dataset)

    total_input = input_tokens_per_request * num_questions
    total_output = output_tokens_per_request * num_questions

    realtime_cost = (total_input * 0.50 + total_output * 1.50) / 1_000_000
    batch_cost = realtime_cost * 0.50  # 50% off
    cached_prefix_saving = (prefix_tokens * num_questions * 0.25) / 1_000_000  # additional saving if prefix cached

    return {
        "num_questions": num_questions,
        "est_input_tokens": total_input,
        "est_output_tokens": total_output,
        "realtime_cost_usd": round(realtime_cost, 4),
        "batch_cost_usd": round(batch_cost, 4),
        "with_prompt_caching_usd": round(batch_cost - cached_prefix_saving, 4),
        "n_per_question": n,
    }
