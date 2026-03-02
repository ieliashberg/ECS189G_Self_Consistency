from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from openai import OpenAI
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# so client persists
_DEFAULT_MODEL_CLIENT: ModelClient | None = None
FIXED_TOP_P = 1.0
_RESPONSES_TEMPERATURE_MODELS = {"gpt-5.2"}
_CONTINUATION_PROMPT = "Continue exactly where you stopped. Do not repeat prior text."
_MAX_OPENAI_CONTINUATIONS = 8
_MAX_HF_CONTINUATIONS = 1


def _get_tokenizer_input_limit(tokenizer: Any) -> int:
    model_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and 0 < model_max < 100_000:
        return model_max
    return 512


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
            outputs: List[str] = []
            for _ in range(num_samples):
                outputs.append(
                    self._generate_openai_chat_with_continuation(
                        model_name=model_name,
                        prompt=prompt,
                        temperature=temperature,
                    )
                )
            return outputs

        if api == "responses":
            outputs: List[str] = []
            for _ in range(num_samples):
                outputs.append(
                    self._generate_openai_responses_with_continuation(
                        model_name=model_name,
                        prompt=prompt,
                        temperature=temperature,
                    )
                )
            return outputs

        raise RuntimeError(f"Unknown OpenAI API: {api}")

    def _generate_openai_chat_with_continuation(
        self,
        *,
        model_name: str,
        prompt: str,
        temperature: float,
    ) -> str:
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        chunks: List[str] = []

        for _ in range(_MAX_OPENAI_CONTINUATIONS + 1):
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": FIXED_TOP_P,
                "n": 1,
            }
            response = self._openai_client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            text = (choice.message.content or "")
            chunks.append(text)

            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason != "length":
                return "".join(chunks).strip()

            messages.extend(
                [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": _CONTINUATION_PROMPT},
                ]
            )

        raise RuntimeError(
            f"OpenAI chat output for model '{model_name}' exceeded continuation limit "
            f"({_MAX_OPENAI_CONTINUATIONS}) and may be truncated."
        )

    def _generate_openai_responses_with_continuation(
        self,
        *,
        model_name: str,
        prompt: str,
        temperature: float,
    ) -> str:
        chunks: List[str] = []
        previous_response_id: str | None = None
        next_input = prompt

        for _ in range(_MAX_OPENAI_CONTINUATIONS + 1):
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "input": next_input,
            }
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            if _responses_supports_temperature(model_name):
                kwargs["temperature"] = temperature
                kwargs["top_p"] = FIXED_TOP_P

            response = self._openai_client.responses.create(**kwargs)
            text = getattr(response, "output_text", None)
            chunks.append(text if isinstance(text, str) else _extract_text_from_response(response))

            if not _responses_was_truncated(response):
                return "".join(chunks).strip()

            previous_response_id = getattr(response, "id", None)
            if not previous_response_id:
                raise RuntimeError(
                    f"OpenAI responses output for model '{model_name}' appears truncated but has "
                    "no response id for continuation."
                )
            next_input = _CONTINUATION_PROMPT

        raise RuntimeError(
            f"OpenAI responses output for model '{model_name}' exceeded continuation limit "
            f"({_MAX_OPENAI_CONTINUATIONS}) and may be truncated."
        )

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

        tokenizer_kwargs: Dict[str, Any] = {"return_tensors": "pt", "truncation": False}
        if architecture == "seq2seq":
            tokenizer_kwargs["verbose"] = False

        model_inputs = tokenizer(prompt, **tokenizer_kwargs).to(device)
        input_len = model_inputs["input_ids"].shape[-1]
        if architecture == "seq2seq":
            max_input_len = _get_tokenizer_input_limit(tokenizer)
            if input_len > max_input_len:
                raise RuntimeError(
                    f"Prompt length ({input_len} tokens) exceeds {model_name} input limit "
                    f"({max_input_len}). Trim few-shot examples rather than truncating question text."
                )

        do_sample = temperature > 0.0
        generate_kwargs: Dict[str, Any] = {
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        inferred_max_new_tokens = _resolve_hf_max_new_tokens(
            model=model,
            architecture=architecture,
            input_len=input_len,
            tokenizer=tokenizer,
        )
        if inferred_max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = inferred_max_new_tokens
        if do_sample:
            generate_kwargs["temperature"] = max(temperature, 1e-5)
            generate_kwargs["top_p"] = FIXED_TOP_P

        # For large seq2seq models (e.g., UL2), sampling many return sequences at once
        # can cause CUDA OOM. Generate one sample at a time to keep memory bounded.
        sample_batch_size = 1 if (architecture == "seq2seq" and num_samples > 1) else num_samples

        outputs: List[str] = []
        remaining = num_samples

        while remaining > 0:
            this_batch = min(sample_batch_size, remaining)
            batch_kwargs = dict(generate_kwargs)
            batch_kwargs["num_return_sequences"] = this_batch
            try:
                with torch.no_grad():
                    generated = model.generate(**model_inputs, **batch_kwargs)
            except RuntimeError as error:
                if "out of memory" in str(error).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

            for seq in generated:
                raw_seq = seq
                if architecture == "seq2seq":
                    raw_seq = _continue_seq2seq_generation_if_needed(
                        model=model,
                        model_inputs=model_inputs,
                        initial_sequence=raw_seq,
                        generate_kwargs=generate_kwargs,
                        max_new_tokens=inferred_max_new_tokens,
                    )
                elif _hf_output_was_truncated(
                    sequence=raw_seq,
                    architecture=architecture,
                    input_len=input_len,
                    max_new_tokens=inferred_max_new_tokens,
                ):
                    raise RuntimeError(
                        f"{model_name} output reached its configured generation limit. "
                        "Trim few-shot examples if needed."
                    )
                if architecture == "causal":
                    seq = raw_seq[input_len:]
                else:
                    seq = raw_seq
                outputs.append(tokenizer.decode(seq, skip_special_tokens=True).strip())
            remaining -= this_batch

        return outputs


def _infer_model_route(model_name: str) -> Tuple[str, str, str]:
    if model_name == "gpt-3.5-turbo":
        return "openai", "chat", ""
    if model_name == "gpt-5-mini":
        return "openai", "responses", ""
    if model_name == "gpt-5.2":
        return "openai", "responses", ""
    if model_name == "google/ul2":
        return "huggingface", "", "seq2seq"
    return "huggingface", "", "causal"


def _responses_supports_temperature(model_name: str) -> bool:
    return model_name in _RESPONSES_TEMPERATURE_MODELS


def _responses_was_truncated(response: Any) -> bool:
    status = getattr(response, "status", None)
    if status == "incomplete":
        return True

    for item in getattr(response, "output", []) or []:
        item_status = getattr(item, "status", None)
        if item_status == "incomplete":
            return True
    return False


def _hf_output_was_truncated(
    *,
    sequence: Any,
    architecture: str,
    input_len: int,
    max_new_tokens: int | None,
) -> bool:
    if max_new_tokens is None:
        return False
    if architecture == "causal":
        new_token_count = len(sequence) - input_len
    else:
        new_token_count = len(sequence)
    return new_token_count >= max_new_tokens


def _continue_seq2seq_generation_if_needed(
    *,
    model: Any,
    model_inputs: Dict[str, Any],
    initial_sequence: Any,
    generate_kwargs: Dict[str, Any],
    max_new_tokens: int | None,
) -> Any:
    sequence = initial_sequence
    if max_new_tokens is None:
        return sequence

    if len(sequence) < max_new_tokens:
        return sequence

    for _ in range(_MAX_HF_CONTINUATIONS):
        if hasattr(sequence, "unsqueeze"):
            decoder_prefix = sequence.unsqueeze(0)
        else:
            if hasattr(torch, "tensor"):
                input_ids = model_inputs.get("input_ids")
                if input_ids is None or not hasattr(input_ids, "device"):
                    decoder_prefix = torch.tensor(sequence).unsqueeze(0)
                else:
                    decoder_prefix = torch.tensor(sequence, device=input_ids.device).unsqueeze(0)
            else:
                decoder_prefix = [list(sequence)]

        continuation_kwargs = dict(generate_kwargs)
        continuation_kwargs["num_return_sequences"] = 1
        continuation_kwargs["decoder_input_ids"] = decoder_prefix
        with torch.no_grad():
            generated = model.generate(**model_inputs, **continuation_kwargs)
        if len(generated) == 0:
            raise RuntimeError("Seq2seq continuation returned no sequences.")
        next_sequence = generated[0]
        if len(next_sequence) <= len(sequence):
            raise RuntimeError("Seq2seq continuation did not advance the decoded sequence.")
        newly_generated = len(next_sequence) - len(sequence)
        sequence = next_sequence
        if newly_generated < max_new_tokens:
            return sequence

    raise RuntimeError(
        f"Seq2seq output still reached generation limit after {_MAX_HF_CONTINUATIONS} continuation."
    )


def _resolve_hf_max_new_tokens(
    *,
    model: Any,
    architecture: str,
    input_len: int,
    tokenizer: Any,
) -> int | None:
    model_config = getattr(model, "config", None)
    context_candidates = [
        getattr(model_config, "max_position_embeddings", None),
        getattr(model_config, "n_positions", None),
        getattr(model_config, "max_seq_len", None),
        getattr(model_config, "seq_length", None),
    ]
    model_context_limit = next(
        (int(value) for value in context_candidates if isinstance(value, int) and value > 0),
        None,
    )

    if model_context_limit is None:
        tokenizer_limit = _get_tokenizer_input_limit(tokenizer)
        if tokenizer_limit > 0:
            model_context_limit = tokenizer_limit

    if model_context_limit is not None:
        if architecture == "causal":
            if input_len >= model_context_limit:
                raise RuntimeError(
                    f"Prompt length ({input_len} tokens) reached model context limit "
                    f"({model_context_limit}). Trim few-shot examples."
                )
            return model_context_limit - input_len
        return model_context_limit

    return None



def _extract_text_from_response(response: Any) -> str:
    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


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
