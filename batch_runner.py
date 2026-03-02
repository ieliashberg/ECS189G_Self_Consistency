from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from prompts import BenchmarkType, build_prompt

_RESPONSES_TEMPERATURE_MODELS = {"gpt-5.2"}


def _infer_batch_mode(model_name: str) -> str:
    if model_name == "gpt-3.5-turbo":
        return "chat"
    if model_name.startswith("gpt-"):
        return "responses"
    return "responses"


def _batch_endpoint_for_model(model_name: str) -> str:
    if _infer_batch_mode(model_name) == "chat":
        return "/v1/chat/completions"
    return "/v1/responses"


def _responses_supports_temperature(model_name: str) -> bool:
    return model_name in _RESPONSES_TEMPERATURE_MODELS


def _normalize_custom_id(custom_id: str) -> str:
    return re.sub(r"-s\d+$", "", custom_id)


def _extract_texts_from_batch_response_body(body: Dict[str, Any]) -> List[str]:
    if not isinstance(body, dict):
        return []

    output_text = body.get("output_text")
    if isinstance(output_text, str):
        text = output_text.strip()
        return [text] if text else []

    choices = body.get("choices")
    if isinstance(choices, list):
        values: List[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                values.append(content.strip())
        if values:
            return values

    values = []
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content_list = item.get("content")
            if not isinstance(content_list, list):
                continue
            for content in content_list:
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    values.append(text.strip())
    return values


def _batch_response_was_truncated(body: Dict[str, Any]) -> bool:
    status = body.get("status")
    if status == "incomplete":
        return True

    choices = body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict) and choice.get("finish_reason") == "length":
                return True

    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_status = item.get("status")
            if item_status == "incomplete":
                return True
    return False


def _response_content_to_text(content: Any) -> str:
    if hasattr(content, "text"):
        maybe = getattr(content, "text")
        if isinstance(maybe, str):
            return maybe
    if hasattr(content, "read"):
        data = content.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, str):
            return data
    return str(content)


class BatchPipeline:
    def __init__(self, cache_dir: str = "results/batch", client: OpenAI | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or OpenAI()

    def _build_requests_jsonl(
        self,
        *,
        model: str,
        benchmark: BenchmarkType,
        dataset: Iterable[dict],
        temperature: float,
        n: int,
        max_tokens: Optional[int],
    ) -> Path:
        mode = _infer_batch_mode(model)
        endpoint = _batch_endpoint_for_model(model)
        safe_model = model.replace("/", "_")
        output_path = self.cache_dir / f"{benchmark.value}_{safe_model}_{mode}_requests.jsonl"

        with output_path.open("w", encoding="utf-8") as handle:
            for idx, example in enumerate(dataset):
                prompt = build_prompt(example["question"], benchmark, cot=True, model_name=model)
                if mode == "chat":
                    body: Dict[str, Any] = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "n": n,
                        "temperature": temperature,
                    }
                    if max_tokens is not None:
                        body["max_tokens"] = max_tokens
                    row = {
                        "custom_id": f"{benchmark.value}-{idx}",
                        "method": "POST",
                        "url": endpoint,
                        "body": body,
                    }
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    continue

                for sample_idx in range(n):
                    body: Dict[str, Any] = {
                        "model": model,
                        "input": prompt,
                    }
                    if max_tokens is not None:
                        body["max_output_tokens"] = max_tokens
                    if _responses_supports_temperature(model):
                        body["temperature"] = temperature

                    row = {
                        "custom_id": f"{benchmark.value}-{idx}-s{sample_idx}",
                        "method": "POST",
                        "url": endpoint,
                        "body": body,
                    }
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        return output_path

    def submit_batch(self, *, model: str, requests_jsonl_path: Path) -> str:
        with requests_jsonl_path.open("rb") as input_file:
            uploaded = self.client.files.create(file=input_file, purpose="batch")

        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint=_batch_endpoint_for_model(model),
            completion_window="24h",
        )
        return batch.id

    def wait_for_completion(
        self,
        batch_id: str,
        poll_seconds: float = 15.0,
        timeout_seconds: float = 60 * 60,
    ) -> Any:
        started = time.time()
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = getattr(batch, "status", "unknown")
            if status in {"completed", "failed", "expired", "cancelled"}:
                return batch

            if (time.time() - started) > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for batch {batch_id} to complete.")
            time.sleep(poll_seconds)

    def download_results(self, batch_id: str) -> Path:
        batch = self.client.batches.retrieve(batch_id)
        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            request_counts = getattr(batch, "request_counts", None)
            failed = getattr(request_counts, "failed", "unknown")
            total = getattr(request_counts, "total", "unknown")
            details = ""
            error_file_id = getattr(batch, "error_file_id", None)
            if error_file_id:
                error_file = self.client.files.content(error_file_id)
                error_text = _response_content_to_text(error_file).strip()
                first_line = next((line for line in error_text.splitlines() if line.strip()), "")
                if first_line:
                    details = f" First error line: {first_line}"
            raise RuntimeError(
                f"Batch {batch_id} completed with no output file (failed={failed}/{total}).{details}"
            )

        content = self.client.files.content(output_file_id)
        output_path = self.cache_dir / f"{batch_id}_output.jsonl"
        output_path.write_text(_response_content_to_text(content), encoding="utf-8")
        return output_path

    def parse_output_file(self, output_jsonl_path: Path) -> Dict[str, List[str]]:
        grouped: Dict[str, List[tuple[int, str]]] = {}

        for line in output_jsonl_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str):
                continue

            response = row.get("response", {})
            if not isinstance(response, dict):
                continue
            status_code = int(response.get("status_code", 0))
            if status_code >= 400:
                continue
            body = response.get("body", {})
            if _batch_response_was_truncated(body):
                raise RuntimeError(
                    f"Batch output for custom_id='{custom_id}' was truncated by a token limit. "
                    "Increase max_output_tokens or trim few-shot examples."
                )
            texts = _extract_texts_from_batch_response_body(body)
            if not texts:
                continue

            normalized = _normalize_custom_id(custom_id)
            sample_match = re.search(r"-s(\d+)$", custom_id)
            base_sample_idx = int(sample_match.group(1)) if sample_match else -1
            bucket = grouped.setdefault(normalized, [])
            for offset, text in enumerate(texts):
                sort_idx = base_sample_idx if base_sample_idx >= 0 else len(bucket) + offset
                bucket.append((sort_idx, text))

        parsed: Dict[str, List[str]] = {}
        for custom_id, entries in grouped.items():
            entries.sort(key=lambda item: item[0])
            parsed[custom_id] = [text for _, text in entries]
        return parsed

    def run_generation_batch(
        self,
        *,
        model: str,
        benchmark: BenchmarkType,
        dataset: Iterable[dict],
        temperature: float,
        n: int,
        max_tokens: Optional[int] = 8192,
    ) -> Dict[int, List[str]]:
        requests_jsonl = self._build_requests_jsonl(
            model=model,
            benchmark=benchmark,
            dataset=dataset,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
        )
        batch_id = self.submit_batch(model=model, requests_jsonl_path=requests_jsonl)
        batch = self.wait_for_completion(batch_id)
        if getattr(batch, "status", "") != "completed":
            raise RuntimeError(f"Batch {batch_id} ended with status '{getattr(batch, 'status', 'unknown')}'.")

        output_jsonl = self.download_results(batch_id)
        by_custom_id = self.parse_output_file(output_jsonl)

        prefix = f"{benchmark.value}-"
        by_question_idx: Dict[int, List[str]] = {}
        for custom_id, texts in by_custom_id.items():
            if not custom_id.startswith(prefix):
                continue
            q_idx = int(custom_id[len(prefix) :])
            by_question_idx[q_idx] = texts
        return by_question_idx
