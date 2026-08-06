"""Parser interface every language plugin implements.

Adding a new language means: implement `LanguageParser`, then register it
(with its file extensions) in `repobrain.parsing.registry`. Nothing else
in the pipeline needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from repobrain.ir.models import FileIR


class LanguageParser(ABC):
    """Parses a single source file into the language-agnostic IR."""

    #: Short identifier, e.g. "java". Stored on the produced FileIR.
    language_name: str = "unknown"

    #: File extensions this parser handles, including the leading dot.
    file_extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, rel_path: str, source: bytes) -> FileIR:
        """Parse `source` (the raw bytes of `rel_path`) into a FileIR.

        Implementations should not raise on malformed source; instead
        return a FileIR with `parse_errors` populated so one bad file
        doesn't abort the whole repository scan.
        """
        raise NotImplementedError
