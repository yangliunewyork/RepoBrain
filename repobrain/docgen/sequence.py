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

#: Keyword-based layer classification, used only to order participants
#: left-to-right in the rendered diagram (API -> service -> data) —
#: matched against the lowercased package + class qualified name.
#: Checked in this order, so a name matching more than one tier resolves
#: to the earliest (leftmost) one.
_API_LAYER_KEYWORDS = ("controller", "endpoint", "resource", "router", "api", "rest", "web")
_DATA_LAYER_KEYWORDS = ("repository", "repo", "dao", "persistence", "database", "mapper", "entity", "store")
#: Everything else (services, domain/model classes, utilities, ...)
#: defaults to this middle rank rather than being pinned to an edge.
_DEFAULT_LAYER_RANK = 1


def _layer_rank(qualified_name: str) -> int:
    lowered = qualified_name.lower()
    if any(k in lowered for k in _API_LAYER_KEYWORDS):
        return 0
    if any(k in lowered for k in _DATA_LAYER_KEYWORDS):
        return 2
    return _DEFAULT_LAYER_RANK


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
    return (
        not method.is_constructor
        and "public" in method.modifiers
        and "public" in class_info.modifiers
    )


def _select_entry_points(repo_ir: RepoIR, grouped: dict[EntryKey, list[CallEdge]]) -> list[EntryKey]:
    called_targets = {(e.callee_class, e.callee_method) for edges in grouped.values() for e in edges}

    candidates: list[tuple[EntryKey, int, bool]] = []
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
                candidates.append((key, len(outgoing), is_pure_entry))

    # Prefer methods nothing else in the repo calls (likely real entry
    # points: controllers, CLI commands, public API surface) with the
    # most outgoing calls; sort by key too so selection is deterministic
    # for fingerprinting.
    candidates.sort(key=lambda c: (not c[2], -c[1], c[0]))
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


def render_mermaid_sequence(entry: EntryKey, steps: list[SequenceStep]) -> str:
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
    participants.sort(key=lambda p: (_layer_rank(p), first_seen_index[p]))

    lines = ["sequenceDiagram"]
    for p in participants:
        lines.append(f'    participant {_sanitize_id(p)} as {p.rsplit(".", 1)[-1]}')
    for step in steps:
        lines.append(f"    {_sanitize_id(step.caller_class)}->>{_sanitize_id(step.callee_class)}: {step.callee_method}()")
    return "\n".join(lines)


def build_sequence_flows(repo_ir: RepoIR, index: SymbolIndex) -> list[SequenceFlow]:
    edges = build_call_graph(repo_ir, index)
    grouped = group_by_caller(edges)

    flows = []
    for entry in _select_entry_points(repo_ir, grouped):
        steps = _trace_flow(entry, grouped)
        flows.append(SequenceFlow(entry_class=entry[0], entry_method=entry[1], steps=steps, mermaid=render_mermaid_sequence(entry, steps)))
    return flows
