"""Registry mapping language names / file extensions to parser instances.

Adding a new language:
1. Implement a `LanguageParser` subclass (see `java_parser.py` for an example).
2. Add one line here: `_PARSERS["<name>"] = lambda: YourParser()`.
Nothing else in the pipeline, CLI, or docgen layer needs to change.
"""
from __future__ import annotations

from typing import Callable

from repobrain.parsing.base import LanguageParser
from repobrain.parsing.java_parser import JavaParser

_PARSER_FACTORIES: dict[str, Callable[[], LanguageParser]] = {
    "java": JavaParser,
}


def available_languages() -> list[str]:
    return list(_PARSER_FACTORIES)


def get_parser(language: str) -> LanguageParser:
    try:
        factory = _PARSER_FACTORIES[language]
    except KeyError as exc:
        raise ValueError(
            f"No parser registered for language '{language}'. "
            f"Available: {', '.join(available_languages())}"
        ) from exc
    return factory()


def extensions_for_languages(languages: list[str]) -> set[str]:
    exts: set[str] = set()
    for lang in languages:
        exts.update(get_parser(lang).file_extensions)
    return exts


def parser_for_extension(extension: str, languages: list[str]) -> LanguageParser | None:
    for lang in languages:
        parser = get_parser(lang)
        if extension in parser.file_extensions:
            return parser
    return None
