"""Local LLM provider backed by Ollama (https://ollama.com).

Talks to the Ollama HTTP API on localhost (or any host you point it at —
still your infrastructure, never a third-party API). Designed around
Qwen3-style models but works with anything Ollama serves.
"""
from __future__ import annotations

import re

import requests

from repobrain.config import LLMConfig
from repobrain.llm.base import LLMProvider, LLMProviderError
from repobrain.logging_setup import get_logger

logger = get_logger("llm.ollama")

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self._session = requests.Session()

    def is_available(self) -> bool:
        try:
            resp = self._session.get(f"{self.config.host}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Ollama unreachable at %s: %s", self.config.host, exc)
            return False

        models = {m.get("model") or m.get("name") for m in resp.json().get("models", [])}
        if self.config.model not in models:
            logger.warning(
                "Model '%s' not found in Ollama (available: %s). "
                "Pull it with: ollama pull %s",
                self.config.model, ", ".join(sorted(m for m in models if m)), self.config.model,
            )
            return False
        return True

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "think": not self.config.suppress_thinking,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
            },
        }
        if system:
            payload["system"] = system

        try:
            resp = self._session.post(
                f"{self.config.host}/api/generate",
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMProviderError(
                f"Failed to reach Ollama at {self.config.host} for model "
                f"'{self.config.model}': {exc}"
            ) from exc

        data = resp.json()
        text = data.get("response", "")
        return _strip_thinking(text).strip()


def _strip_thinking(text: str) -> str:
    """Defensively strip any <think>...</think> block that slips through,
    in case a given model/version ignores the `think` request option."""
    return _THINK_BLOCK_RE.sub("", text)
