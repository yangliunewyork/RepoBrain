"""Configuration loading for RepoBrain.

Precedence (highest wins): CLI flags > project-local config file >
packaged default.yaml.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _default_config_dict() -> dict:
    default_path = resources.files("repobrain").joinpath("default_config.yaml")
    return yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen3:8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    num_ctx: int = 8192
    timeout_seconds: int = 600
    suppress_thinking: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str | None = None


@dataclass
class RepoBrainConfig:
    languages: list[str] = field(default_factory=lambda: ["java"])
    exclude_patterns: list[str] = field(default_factory=list)
    output_dir: str = "docs/generated"
    docs: list[str] = field(
        default_factory=lambda: [
            "README.md",
            "ARCHITECTURE.md",
            "SEQUENCE.md",
        ]
    )
    llm: LLMConfig = field(default_factory=LLMConfig)
    state_dir: str = ".repobrain"
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def load(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> "RepoBrainConfig":
        merged = _default_config_dict()
        if config_path is not None:
            project_cfg = _load_yaml(Path(config_path))
            merged = _deep_merge(merged, project_cfg)
        if overrides:
            merged = _deep_merge(merged, overrides)

        llm_data = merged.get("llm", {})
        logging_data = merged.get("logging", {})
        return RepoBrainConfig(
            languages=merged.get("languages", ["java"]),
            exclude_patterns=merged.get("exclude_patterns", []),
            output_dir=merged.get("output_dir", "docs/generated"),
            docs=merged.get("docs", ["README.md", "ARCHITECTURE.md", "SEQUENCE.md"]),
            llm=LLMConfig(**llm_data),
            state_dir=merged.get("state_dir", ".repobrain"),
            logging=LoggingConfig(**logging_data),
        )
