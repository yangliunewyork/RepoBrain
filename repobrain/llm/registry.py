"""Registry mapping provider names (from config `llm.provider`) to
implementations. Add a new provider by implementing `LLMProvider` and
registering a factory here.
"""
from __future__ import annotations

from typing import Callable

from repobrain.config import LLMConfig
from repobrain.llm.base import LLMProvider
from repobrain.llm.ollama_provider import OllamaProvider

_PROVIDER_FACTORIES: dict[str, Callable[[LLMConfig], LLMProvider]] = {
    "ollama": OllamaProvider,
}


def available_providers() -> list[str]:
    return list(_PROVIDER_FACTORIES)


def get_provider(config: LLMConfig) -> LLMProvider:
    try:
        factory = _PROVIDER_FACTORIES[config.provider]
    except KeyError as exc:
        raise ValueError(
            f"No LLM provider registered for '{config.provider}'. "
            f"Available: {', '.join(available_providers())}"
        ) from exc
    return factory(config)
