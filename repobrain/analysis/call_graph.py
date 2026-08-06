"""Resolves method-invocation call sites (`ir.models.MethodCall`) to their
target class, forming a call graph used to build sequence diagrams.

Resolution is intentionally conservative and local: a call's receiver is
looked up against the *caller's own* fields and the *calling method's*
parameters — the only declared types available without tracking local
variables — then resolved to a project class the same way dependency
edges are (`dependency_analyzer.resolve_type_reference`): via import,
same-package lookup, or an unambiguous simple-name match. Calls on a
local variable, a chained expression, or an external/JDK type resolve to
nothing and are dropped rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass

from repobrain.analysis.dependency_analyzer import resolve_type_reference
from repobrain.analysis.symbol_extractor import SymbolIndex
from repobrain.ir.models import ClassInfo, MethodInfo, RepoIR


@dataclass(frozen=True)
class CallEdge:
    caller_class: str
    caller_method: str
    callee_class: str
    callee_method: str


def _receiver_types(class_info: ClassInfo, method: MethodInfo) -> dict[str, str]:
    types = {f.name: f.type for f in class_info.fields}
    types.update({p.name: p.type for p in method.parameters})
    return types


def build_call_graph(repo_ir: RepoIR, index: SymbolIndex) -> list[CallEdge]:
    """Every resolvable call site in the repo, in the order encountered
    (source order within a method, file/class iteration order across
    methods)."""
    edges: list[CallEdge] = []
    for file_ir in repo_ir.files.values():
        for class_info in file_ir.iter_classes():
            for method in class_info.methods:
                receiver_types = _receiver_types(class_info, method)
                for call in method.calls:
                    if call.receiver is None or call.receiver == "this":
                        callee_class = class_info.qualified_name
                    else:
                        # Not a known field/parameter? Might still be a
                        # project class used for a static call, e.g.
                        # `WidgetFactory.create()` — try it as-is.
                        raw_type = receiver_types.get(call.receiver, call.receiver)
                        callee_class = resolve_type_reference(raw_type, file_ir, index)
                        if callee_class is None:
                            continue
                    edges.append(CallEdge(class_info.qualified_name, method.name, callee_class, call.method))
    return edges


def group_by_caller(edges: list[CallEdge]) -> dict[tuple[str, str], list[CallEdge]]:
    grouped: dict[tuple[str, str], list[CallEdge]] = {}
    for edge in edges:
        grouped.setdefault((edge.caller_class, edge.caller_method), []).append(edge)
    return grouped
