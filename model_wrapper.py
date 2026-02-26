from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


class ModelClient:
    def __init__(self) -> None:
        self._openai_client = None
        self._hf_models: Dict[str, Any] = {}
        self._hf_tokenizers: Dict[str, Any] = {}

    def generate(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        num_samples: int = 1,
    ) -> List[str]:
        provider, api, architecture = _infer_model_route(model_name)

        if provider == "openai":
            return self._generate_openai(
                model_name=model_name,
                prompt=prompt,
                api=api,
                temperature=temperature,
                top_p=top_p,
                num_samples=num_samples,
            )

        if provider == "huggingface":
            return self._generate_huggingface(
                model_name=model_name,
                prompt=prompt,
                architecture=architecture,
                temperature=temperature,
                top_p=top_p,
                num_samples=num_samples,
            )
        raise RuntimeError(f"Unknown provider: {provider}")

    def _generate_openai(
        self,
        model_name: str,
        prompt: str,
        api: str,
        temperature: float,
        top_p: float,
        num_samples: int,
    ) -> List[str]:
        if self._openai_client is None:
            self._openai_client = OpenAI()

        if api == "chat":
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "n": num_samples,
            }
            response = self._openai_client.chat.completions.create(**kwargs)
            return [(choice.message.content or "").strip() for choice in response.choices]

        if api == "responses":
            outputs: List[str] = []
            for _ in range(num_samples):
                kwargs = {
                    "model": model_name,
                    "input": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                response = self._openai_client.responses.create(**kwargs)
                text = getattr(response, "output_text", None)
                outputs.append(text.strip() if text else _extract_text_from_response(response).strip())
            return outputs

        raise RuntimeError(f"Unknown OpenAI API: {api}")

    def _generate_huggingface(
        self,
        model_name: str,
        prompt: str,
        architecture: str,
        temperature: float,
        top_p: float,
        num_samples: int,
    ) -> List[str]:
        resolved_model_name = _resolve_hf_model_name(model_name)

        if model_name not in self._hf_models or model_name not in self._hf_tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, use_fast=True)
            if architecture == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(resolved_model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(resolved_model_name)
            self._hf_models[model_name] = model
            self._hf_tokenizers[model_name] = tokenizer

        model = self._hf_models[model_name]
        tokenizer = self._hf_tokenizers[model_name]

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generate_kwargs: Dict[str, Any] = {
            "do_sample": temperature > 0.0,
            "temperature": max(temperature, 1e-5),
            "top_p": top_p,
            "num_return_sequences": num_samples,
            "pad_token_id": tokenizer.pad_token_id,
        }

        with torch.no_grad():
            generated = model.generate(**model_inputs, **generate_kwargs)

        outputs: List[str] = []
        input_len = model_inputs["input_ids"].shape[-1]
        for seq in generated:
            if architecture == "causal":
                seq = seq[input_len:]
            outputs.append(tokenizer.decode(seq, skip_special_tokens=True).strip())
        return outputs


def _infer_model_route(model_name: str) -> Tuple[str, str, str]:
    if model_name == "gpt-3.5-turbo":
        return "openai", "chat", ""
    if model_name == "gpt-5.2":
        return "openai", "responses", ""
    if model_name == "google/ul2":
        return "huggingface", "", "seq2seq"
    return "huggingface", "", "causal"


def _resolve_hf_model_name(model_name: str) -> str:
    if model_name == "google/ul2":
        return "google/flan-ul2"
    return model_name


def _extract_text_from_response(response: Any) -> str:
    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)
