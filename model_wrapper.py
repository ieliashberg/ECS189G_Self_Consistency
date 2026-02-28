from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from openai import OpenAI
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# so client persists
_DEFAULT_MODEL_CLIENT: ModelClient | None = None
FIXED_TOP_P = 1.0


class ModelClient:
    def __init__(self) -> None:
        self._openai_client = None
        self._hf_models: Dict[str, Any] = {}
        self._hf_tokenizers: Dict[str, Any] = {}
        self._hf_devices: Dict[str, torch.device] = {}

    def generate(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        num_samples: int = 1,
    ) -> List[str]:
        provider, api, architecture = _infer_model_route(model_name)

        if provider == "openai":
            return self._generate_openai(
                model_name=model_name,
                prompt=prompt,
                api=api,
                temperature=temperature,
                num_samples=num_samples,
            )

        if provider == "huggingface":
            return self._generate_huggingface(
                model_name=model_name,
                prompt=prompt,
                architecture=architecture,
                temperature=temperature,
                num_samples=num_samples,
            )
        raise RuntimeError(f"Unknown provider: {provider}")

    def _generate_openai(
        self,
        model_name: str,
        prompt: str,
        api: str,
        temperature: float,
        num_samples: int,
    ) -> List[str]:
        if self._openai_client is None:
            self._openai_client = OpenAI()

        if api == "chat":
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": FIXED_TOP_P,
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
                    "top_p": FIXED_TOP_P,
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
        num_samples: int,
    ) -> List[str]:
        if model_name not in self._hf_models or model_name not in self._hf_tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model_config = AutoConfig.from_pretrained(model_name)
            # UL2 checkpoints already include separate embedding weights.
            # Disabling tying avoids noisy warnings during load.
            if architecture == "seq2seq" and getattr(model_config, "tie_word_embeddings", None):
                model_config.tie_word_embeddings = False
            if architecture == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=model_config)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
            device = _select_torch_device()
            model = model.to(device)
            model.eval()
            print(f"[model_wrapper] Loaded Hugging Face model '{model_name}' on device: {device}.")
            self._hf_models[model_name] = model
            self._hf_tokenizers[model_name] = tokenizer
            self._hf_devices[model_name] = device

        model = self._hf_models[model_name]
        tokenizer = self._hf_tokenizers[model_name]
        device = self._hf_devices[model_name]

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        max_input_length = _resolve_max_input_length(model, tokenizer)
        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        do_sample = temperature > 0.0
        generate_kwargs: Dict[str, Any] = {
            "do_sample": do_sample,
            "num_return_sequences": num_samples,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = max(temperature, 1e-5)
            generate_kwargs["top_p"] = FIXED_TOP_P

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



def _extract_text_from_response(response: Any) -> str:
    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _resolve_max_input_length(model: Any, tokenizer: Any) -> int:
    candidates: List[int] = []

    for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            candidates.append(value)

    token_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(token_max, int) and 0 < token_max < 10_000_000:
        candidates.append(token_max)

    if candidates:
        return min(candidates)
    return 2048


def _select_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _get_default_model_client() -> ModelClient:
    global _DEFAULT_MODEL_CLIENT
    if _DEFAULT_MODEL_CLIENT is None:
        _DEFAULT_MODEL_CLIENT = ModelClient()
    return _DEFAULT_MODEL_CLIENT

def run_model(
    model_name: str,
    prompt: str,
    temperature: float = 0.7,
    num_samples: int = 1,
) -> List[str]:
    client = _get_default_model_client()
    return client.generate(
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
        num_samples=num_samples,
    )
