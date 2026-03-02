"""
Local disk cache for OpenAI responses.

Avoids re-paying for identical queries when re-running the pipeline.
Saves ~100% on repeated runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional


_CACHE_DIR = Path("response_cache")


def _cache_key(model: str, prompt: str, temperature: float, n: int) -> str:
    """Deterministic hash of request parameters."""
    blob = json.dumps({
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "n": n,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def get_cached(model: str, prompt: str, temperature: float, n: int) -> Optional[List[str]]:
    """Return cached responses if available, else None."""
    _CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(model, prompt, temperature, n)
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data["responses"]
    return None


def put_cache(model: str, prompt: str, temperature: float, n: int, responses: List[str]) -> None:
    """Store responses in disk cache."""
    _CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(model, prompt, temperature, n)
    path = _CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({
        "model": model,
        "temperature": temperature,
        "n": n,
        "responses": responses,
    }, indent=2))
