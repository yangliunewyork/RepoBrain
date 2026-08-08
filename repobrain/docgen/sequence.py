"""Selects a bounded set of "interesting" call flows and renders each as
a deterministic Mermaid `sequenceDiagram` block.

Diagrams are built entirely from resolved call edges (`analysis.call_graph`)
— never re-derived by the LLM — the same way `ARCHITECTURE.md`'s
dependency graph is: correctness of the diagram itself shouldn't depend
on an 8B model reproducing Mermaid syntax faithfully. The LLM's job is
only to describe, in prose, what each already-correct diagram shows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from repobrain.analysis.call_graph import CallEdge, build_call_graph, group_by_caller
from repobrain.analysis.layers import Layer, classify_layer, layer_rank
from repobrain.analysis.symbol_extractor import SymbolIndex
from repobrain.ir.models import RepoIR

#: How many flows to surface at all. Kept small — each becomes a full
#: diagram plus prose in the generated doc.
MAX_SEQUENCE_FLOWS = 8

#: Per-flow bounds so a deeply/broadly connected call graph still
#: produces a readable diagram rather than an unbounded trace.
MAX_STEPS_PER_FLOW = 12
MAX_DEPTH = 4

EntryKey = tuple[str, str]  # (caller_class qualified name, method name)

#: Method-level annotations marking a definite application entry point
#: (an HTTP route, RPC handler, ...) rather than just "public and nothing
#: else calls it." When present, these methods are preferred as flow
#: starting points over the generic heuristic below — the annotation
#: *is* the entry point, not a proxy for one.
_ROUTE_ANNOTATIONS = {
    "GetMapping", "PostMapping", "PutMapping", "DeleteMapping", "PatchMapping",
    "RequestMapping", "GET", "POST", "PUT", "DELETE", "PATCH",  # JAX-RS
}

#: Method-level annotations marking test code, excluded from entry-point
#: candidacy even if a test file slips past the default exclude_patterns
#: (non-standard layout, generated sources, ...). File-level exclusion is
#: the primary defense (see default_config.yaml); this is a second layer
#: — without it, a `@Test` method that calls several services and isn't
#: itself called by anything else looks exactly like a real entry point.
_TEST_ANNOTATIONS = {"Test", "ParameterizedTest", "RepeatedTest", "TestFactory"}


@dataclass
class SequenceStep:
    caller_class: str
    caller_method: str
    callee_class: str
    callee_method: str


@dataclass
class SequenceFlow:
    entry_class: str
    entry_method: str
    steps: list[SequenceStep] = field(default_factory=list)
    mermaid: str = ""


def _is_entry_candidate(class_info, method) -> bool:
    if method.is_constructor:
        return False
    if "public" not in method.modifiers or "public" not in class_info.modifiers:
        return False
    if set(method.annotations) & _TEST_ANNOTATIONS:
        return False
    return True


def _select_entry_points(repo_ir: RepoIR, grouped: dict[EntryKey, list[CallEdge]]) -> list[EntryKey]:
    called_targets = {(e.callee_class, e.callee_method) for edges in grouped.values() for e in edges}

    candidates: list[tuple[EntryKey, int, bool, bool]] = []
    for file_ir in repo_ir.files.values():
        for class_info in file_ir.iter_classes():
            for method in class_info.methods:
                if not _is_entry_candidate(class_info, method):
                    continue
                key: EntryKey = (class_info.qualified_name, method.name)
                outgoing = grouped.get(key, [])
                if not outgoing:
                    continue
                is_pure_entry = key not in called_targets
                has_route_annotation = bool(set(method.annotations) & _ROUTE_ANNOTATIONS)
                candidates.append((key, len(outgoing), is_pure_entry, has_route_annotation))

    # Prefer, in order: (1) a verified route/handler annotation -- not a
    # guess, an actual framework-recognized entry point; (2) methods
    # nothing else in the repo calls (likely real entry points otherwise:
    # CLI commands, public API surface); (3) most outgoing calls. Sort by
    # key too so selection is deterministic for fingerprinting.
    candidates.sort(key=lambda c: (not c[3], not c[2], -c[1], c[0]))
    return [c[0] for c in candidates[:MAX_SEQUENCE_FLOWS]]


def _trace_flow(entry: EntryKey, grouped: dict[EntryKey, list[CallEdge]]) -> list[SequenceStep]:
    steps: list[SequenceStep] = []
    visited: set[EntryKey] = set()

    def visit(key: EntryKey, depth: int) -> None:
        if depth > MAX_DEPTH or len(steps) >= MAX_STEPS_PER_FLOW or key in visited:
            return
        visited.add(key)
        for edge in grouped.get(key, []):
            if len(steps) >= MAX_STEPS_PER_FLOW:
                return
            steps.append(SequenceStep(edge.caller_class, edge.caller_method, edge.callee_class, edge.callee_method))
            visit((edge.callee_class, edge.callee_method), depth + 1)

    visit(entry, 0)
    return steps


def _sanitize_id(qualified_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", qualified_name) or "P"


def render_mermaid_sequence(entry: EntryKey, steps: list[SequenceStep], layer_of: dict[str, Layer | None]) -> str:
    if not steps:
        return ""

    participants: list[str] = []

    def add_participant(qname: str) -> None:
        if qname not in participants:
            participants.append(qname)

    add_participant(entry[0])
    for step in steps:
        add_participant(step.caller_class)
        add_participant(step.callee_class)

    # Left-to-right layout follows architectural layer (API -> service ->
    # data), not raw call order — e.g. a service that calls a repository
    # and then another service still renders with both services left of
    # the repository. Ties (same layer) keep call/first-seen order.
    first_seen_index = {p: i for i, p in enumerate(participants)}
    participants.sort(key=lambda p: (layer_rank(layer_of.get(p)), first_seen_index[p]))

    lines = ["sequenceDiagram"]
    for p in participants:
        lines.append(f'    participant {_sanitize_id(p)} as {p.rsplit(".", 1)[-1]}')
    for step in steps:
        lines.append(f"    {_sanitize_id(step.caller_class)}->>{_sanitize_id(step.callee_class)}: {step.callee_method}()")
    return "\n".join(lines)


def build_sequence_flows(repo_ir: RepoIR, index: SymbolIndex) -> list[SequenceFlow]:
    edges = build_call_graph(repo_ir, index)
    grouped = group_by_caller(edges)
    layer_of = {qname: classify_layer(entry.class_info) for qname, entry in index.by_qualified_name.items()}

    flows = []
    for entry in _select_entry_points(repo_ir, grouped):
        steps = _trace_flow(entry, grouped)
        mermaid = render_mermaid_sequence(entry, steps, layer_of)
        flows.append(SequenceFlow(entry_class=entry[0], entry_method=entry[1], steps=steps, mermaid=mermaid))
    return flows
