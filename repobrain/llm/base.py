"""LLM provider interface.

RepoBrain talks to models purely through this interface so the
documentation pipeline never depends on a specific runtime. The default
(and currently only) implementation, `OllamaProvider`, talks to a local
Ollama daemon — no source code or generated prompts ever leave the
machine. Adding a new provider (another local runtime, or an opt-in
remote API) means implementing this class and registering it in
`repobrain.llm.registry`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot produce a completion."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's completion for `prompt`.

        Raises LLMProviderError on failure (connection issue, model not
        found, timeout, ...) so callers can decide how to degrade.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap reachability/health check, used for a fast CLI failure
        message instead of failing deep inside the doc pipeline."""
        raise NotImplementedError
