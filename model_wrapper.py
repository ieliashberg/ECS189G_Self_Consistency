from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

# so client persists
_DEFAULT_MODEL_CLIENT: ModelClient | None = None


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
        if model_name not in self._hf_models or model_name not in self._hf_tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            use_gpu = torch.cuda.is_available()
            model = None

            if use_gpu and _BNB_AVAILABLE:
                try:
                    load_kwargs: Dict[str, Any] = {
                        "quantization_config": BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                        ),
                        "device_map": "auto",
                    }
                    if architecture == "seq2seq":
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
                    else:
                        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                    print(f"[model_wrapper] Loaded {model_name} (4-bit quantized, GPU)")
                except Exception as e:
                    print(f"[model_wrapper] 4-bit loading failed: {e}")
                    model = None

            # Strategy 2: float16 on GPU
            if model is None and use_gpu:
                try:
                    load_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
                    if architecture == "seq2seq":
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
                    else:
                        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                    print(f"[model_wrapper] Loaded {model_name} (float16, GPU)")
                except Exception as e:
                    print(f"[model_wrapper] float16 GPU loading failed: {e}")
                    model = None

            # Strategy 3: CPU fallback
            if model is None:
                if architecture == "seq2seq":
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                print(f"[model_wrapper] Loaded {model_name} (float32, CPU)")

            self._hf_models[model_name] = model
            self._hf_tokenizers[model_name] = tokenizer

        model = self._hf_models[model_name]
        tokenizer = self._hf_tokenizers[model_name]

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": 256,
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

    # seq2seq models (T5 family)
    seq2seq_prefixes = ("google/ul2", "google/flan-t5", "google/flan-ul2", "t5-")
    if any(model_name.startswith(p) for p in seq2seq_prefixes):
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

def _get_default_model_client() -> ModelClient:
    global _DEFAULT_MODEL_CLIENT
    if _DEFAULT_MODEL_CLIENT is None:
        _DEFAULT_MODEL_CLIENT = ModelClient()
    return _DEFAULT_MODEL_CLIENT

def run_model(
    model_name: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 1.0,
    num_samples: int = 1,
) -> List[str]:

    client = _get_default_model_client()
    return client.generate(
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        num_samples=num_samples,
    )
