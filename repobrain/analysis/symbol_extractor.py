"""Flattens a RepoIR into lookup structures used by dependency analysis
and documentation generation: every class indexed by qualified name,
simple name, and containing package, plus the file it came from.
"""
from __future__ import annotations

from dataclasses import dataclass

from repobrain.ir.models import ClassInfo, RepoIR


@dataclass
class ClassEntry:
    class_info: ClassInfo
    file_path: str
    package: str


@dataclass
class SymbolIndex:
    by_qualified_name: dict[str, ClassEntry]
    by_simple_name: dict[str, list[ClassEntry]]
    by_package: dict[str, list[ClassEntry]]

    def resolve_simple_name(self, simple_name: str) -> ClassEntry | None:
        """Return the entry for `simple_name` if it uniquely identifies a class."""
        candidates = self.by_simple_name.get(simple_name)
        if candidates and len(candidates) == 1:
            return candidates[0]
        return None

    def public_api_entries(self) -> list[ClassEntry]:
        """Classes considered part of the public surface: those explicitly
        declared `public` (package-private/private helpers are excluded)."""
        return [e for e in self.by_qualified_name.values() if "public" in e.class_info.modifiers]


def build_symbol_index(repo_ir: RepoIR) -> SymbolIndex:
    by_qualified: dict[str, ClassEntry] = {}
    by_simple: dict[str, list[ClassEntry]] = {}
    by_package: dict[str, list[ClassEntry]] = {}

    for file_ir in repo_ir.files.values():
        package = file_ir.package or ""
        for class_info in file_ir.iter_classes():
            entry = ClassEntry(class_info=class_info, file_path=file_ir.path, package=package)
            by_qualified[class_info.qualified_name] = entry
            by_simple.setdefault(class_info.name, []).append(entry)
            by_package.setdefault(package, []).append(entry)

    return SymbolIndex(by_qualified_name=by_qualified, by_simple_name=by_simple, by_package=by_package)
