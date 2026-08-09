"""Selects a bounded set of "interesting" call flows and renders each as
a deterministic, full-round-trip Mermaid `sequenceDiagram` block.

Diagrams are built entirely from resolved call edges (`analysis.call_graph`)
— never re-derived by the LLM — the same way `ARCHITECTURE.md`'s
component graph is: correctness of the diagram itself shouldn't depend
on an 8B model reproducing Mermaid syntax faithfully. The LLM's job is
only to describe, in prose, what each already-correct diagram shows.

Each diagram shows the full round trip a real request makes: a synthetic
`Client` participant issues the initiating call (labeled with the actual
HTTP method + path when the entry point carries a route annotation),
forward calls trace through the resolved call graph, dashed return
arrows unwind the stack in reverse using each call's real declared
return type, and — when the deepest resolved class in the chain imports
a recognized external-system library (a JDBC/JPA driver, jOOQ, a message
queue client, ...) — one more hop to that system is appended as the
visible sink, mirroring how a real request actually terminates at a
database or other external dependency rather than stopping at the last
line of application code RepoBrain can see.

Only *architecturally meaningful* calls make it into a diagram — one
component genuinely talking to another (api/service/data), or a class
delegating to its own other methods. A call on a request/response DTO
or a domain object resolves through the call graph exactly the same way
a call on a real collaborator does (both are project classes), but
`request.getStoreName()` or `Store.builder()` is the caller reading/
building its own local data, not an inter-component hop — showing it in
a "primary request flow" diagram misrepresents a data access as if it
were the controller calling out to that object for information. See
`_is_architecturally_meaningful_call`.

When a call's declared receiver type is an interface (the common
Spring pattern: `private final WidgetService service;` field typed as
the interface, injected with a concrete impl), the interface's own
method has no body to trace further calls from — interfaces don't have
one. If the project defines exactly one concrete class implementing
that interface, tracing continues through *that* class's real method
instead, so a flow doesn't go dark at the interface boundary (e.g.
Controller -> WidgetService stopping short of the repository call the
real implementation makes). See `_resolve_via_implementation`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from repobrain.analysis.call_graph import CallEdge, build_call_graph, group_by_caller
from repobrain.analysis.dependency_analyzer import resolve_type_reference
from repobrain.analysis.external_systems import classify_external_systems
from repobrain.analysis.layers import Layer, classify_layer, layer_rank
from repobrain.analysis.symbol_extractor import SymbolIndex
from repobrain.ir.models import MethodInfo, RepoIR

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

#: HTTP verb implied by a route annotation's own name — Spring's
#: per-verb annotations and JAX-RS's verb annotations both encode the
#: method in the annotation itself, no argument parsing required.
_ANNOTATION_HTTP_VERBS = {
    "GetMapping": "GET", "PostMapping": "POST", "PutMapping": "PUT",
    "DeleteMapping": "DELETE", "PatchMapping": "PATCH",
    "GET": "GET", "POST": "POST", "PUT": "PUT", "DELETE": "DELETE", "PATCH": "PATCH",
}

#: Method-level annotations marking test code, excluded from entry-point
#: candidacy even if a test file slips past the default exclude_patterns
#: (non-standard layout, generated sources, ...). File-level exclusion is
#: the primary defense (see default_config.yaml); this is a second layer
#: — without it, a `@Test` method that calls several services and isn't
#: itself called by anything else looks exactly like a real entry point.
_TEST_ANNOTATIONS = {"Test", "ParameterizedTest", "RepeatedTest", "TestFactory"}

#: The synthetic participant representing whatever originates a request
#: into the system — a browser, another service, a CLI invocation.
CLIENT_PARTICIPANT = "Client"

#: Rank sentinels so the synthetic Client and any external-system
#: participants sort to the far left/right regardless of `layer_rank`'s
#: normal api/service/domain/data ordering (they aren't project classes,
#: so they're never keys in `layer_of`).
_CLIENT_RANK = -1
_EXTERNAL_RANK = 99


@dataclass
class SequenceStep:
    caller_class: str
    caller_method: str
    callee_class: str
    callee_method: str
    #: The callee's actual declared return type, if not void — used to
    #: label the dashed return arrow. None means "no return arrow drawn
    #: for this step" (a void method has nothing to show returning).
    return_type: str | None = None


@dataclass
class SequenceFlow:
    entry_class: str
    entry_method: str
    steps: list[SequenceStep] = field(default_factory=list)
    mermaid: str = ""


def _find_method(qname: str, method_name: str, index: SymbolIndex) -> MethodInfo | None:
    entry = index.by_qualified_name.get(qname)
    if entry is None:
        return None
    return next((m for m in entry.class_info.methods if m.name == method_name), None)


def _resolve_return_type(qname: str, method_name: str, index: SymbolIndex) -> str | None:
    method = _find_method(qname, method_name, index)
    if method is None or method.return_type in ("", "void"):
        return None
    return method.return_type


def _route_label(qname: str, method_name: str, index: SymbolIndex) -> str | None:
    """HTTP-style label for an entry point, e.g. "POST /products" — the
    verb comes from the route annotation's own name, the path from that
    annotation's argument combined with a class-level base path when
    present (the common Spring pattern: `@RequestMapping("/products")`
    on the class, `@PostMapping` alone on the method). Returns None when
    the method carries no recognized route annotation at all.
    """
    entry = index.by_qualified_name.get(qname)
    method = _find_method(qname, method_name, index)
    if method is None:
        return None

    route_annotation = next((a for a in method.annotations if a in _ROUTE_ANNOTATIONS), None)
    if route_annotation is None:
        return None

    verb = _ANNOTATION_HTTP_VERBS.get(route_annotation)
    method_path = method.annotation_args.get(route_annotation, "")
    class_path = ""
    if entry is not None:
        class_annotation = next((a for a in entry.class_info.annotations if a in _ROUTE_ANNOTATIONS), None)
        if class_annotation is not None:
            class_path = entry.class_info.annotation_args.get(class_annotation, "")

    if class_path or method_path:
        path = (class_path.rstrip("/") + "/" + method_path.lstrip("/")).rstrip("/") or "/"
    else:
        path = "/"

    return f"{verb} {path}" if verb else path


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


def _is_architecturally_meaningful_call(edge: CallEdge, index: SymbolIndex) -> bool:
    """True unless this call is specifically on a **domain** object —
    i.e. a request/response DTO, entity, or other plain data holder
    (see `analysis.layers.classify_layer`) — in which case it's the
    caller reading/building its own local data, not one component
    talking to another.

    A call on a request DTO (`request.getStoreName()`) or a builder
    (`Store.builder()...build()`) resolves through the call graph the
    same way a call on a real injected collaborator does — the receiver
    is a project class either way — but it isn't a "the controller talks
    to the service" style interaction worth showing in a sequence
    diagram. This is deliberately a deny-list (drop only confirmed
    domain objects), not an allow-list requiring api/service/data: a
    great deal of real code has collaborator classes with no recognized
    annotation and no layer-suggestive name, and those must still show
    up as real interactions rather than being silently dropped for
    lacking a classification. Self-calls (a class calling its own other
    methods) are always kept regardless, since that's internal
    delegation within one component, not a cross-component hop the
    domain question doesn't apply to.
    """
    if edge.caller_class == edge.callee_class:
        return True
    entry = index.by_qualified_name.get(edge.callee_class)
    if entry is None:
        return True  # unresolvable to a class at all -- nothing to judge, don't drop it
    return classify_layer(entry.class_info) != "domain"


def _build_implementors_map(repo_ir: RepoIR, index: SymbolIndex) -> dict[str, list[str]]:
    """interface qualified name -> qualified names of project classes
    that implement it, via each class's own `implements` list resolved
    the same way dependency edges are (import / same-package / unique
    simple-name match)."""
    implementors: dict[str, list[str]] = {}
    for file_ir in repo_ir.files.values():
        for class_info in file_ir.iter_classes():
            if class_info.kind != "class":
                continue
            for raw in class_info.implements:
                resolved = resolve_type_reference(raw, file_ir, index)
                if resolved:
                    implementors.setdefault(resolved, []).append(class_info.qualified_name)
    return implementors


def _resolve_via_implementation(key: EntryKey, index: SymbolIndex, implementors: dict[str, list[str]]) -> EntryKey:
    """If `key`'s class is an interface (which never has a method body
    to trace further calls from) and the project defines exactly one
    concrete implementation, continue tracing through that
    implementation's real method instead — the call site recorded in
    the diagram still names the interface (that's what the caller
    actually depends on), but only a concrete class can say what
    happens next. Multiple implementations are left alone rather than
    guessing which one is actually injected at runtime.
    """
    qname, method_name = key
    entry = index.by_qualified_name.get(qname)
    if entry is None or entry.class_info.kind != "interface":
        return key
    impls = implementors.get(qname, [])
    if len(impls) != 1:
        return key
    return (impls[0], method_name)


def _trace_flow(
    entry: EntryKey, grouped: dict[EntryKey, list[CallEdge]], index: SymbolIndex, implementors: dict[str, list[str]]
) -> list[SequenceStep]:
    steps: list[SequenceStep] = []
    visited: set[EntryKey] = set()

    def visit(key: EntryKey, depth: int) -> None:
        if depth > MAX_DEPTH or len(steps) >= MAX_STEPS_PER_FLOW or key in visited:
            return
        visited.add(key)
        for edge in grouped.get(key, []):
            if len(steps) >= MAX_STEPS_PER_FLOW:
                return
            # Redirect *before* recording the step, not just when
            # continuing the trace — otherwise the step into the
            # interface names it as the callee, but the next arrow
            # originates from the implementation with no visible
            # transition between them, leaving a disconnected-looking
            # diagram even though the underlying trace is complete.
            callee_key = _resolve_via_implementation((edge.callee_class, edge.callee_method), index, implementors)
            return_type = _resolve_return_type(callee_key[0], callee_key[1], index)
            steps.append(SequenceStep(edge.caller_class, edge.caller_method, callee_key[0], callee_key[1], return_type))
            visit(callee_key, depth + 1)

    visit(entry, 0)
    return steps


def _class_external_categories(
    qname: str, repo_ir: RepoIR, index: SymbolIndex, external_systems: dict[str, list[str]]
) -> list[str]:
    entry = index.by_qualified_name.get(qname)
    if entry is None:
        return []
    file_ir = repo_ir.files.get(entry.file_path)
    if file_ir is None:
        return []
    matched: list[str] = []
    for imp in file_ir.imports:
        for category, prefixes in external_systems.items():
            if category not in matched and any(imp.path.startswith(p) for p in prefixes):
                matched.append(category)
    return matched


def _append_external_terminal(
    steps: list[SequenceStep], repo_ir: RepoIR, index: SymbolIndex, external_systems: dict[str, list[str]]
) -> list[SequenceStep]:
    """If the deepest class this flow actually reaches imports a
    recognized external-system library, append one more hop to that
    system as the visible sink — a real request doesn't stop at the last
    line of application code RepoBrain can see; it continues into
    whatever that code actually talks to. The return trip is labeled
    generically ("result"), since there's no Java method signature for
    an external system to read a real type from — everything else here
    still resolves return types from real declarations.
    """
    if not steps or not external_systems:
        return steps
    last = steps[-1]
    categories = _class_external_categories(last.callee_class, repo_ir, index, external_systems)
    if not categories:
        return steps
    category = categories[0]  # a class matching >1 category is rare; keep the diagram to one sink
    return steps + [SequenceStep(last.callee_class, last.callee_method, category, last.callee_method, return_type="result")]


def _sanitize_id(qualified_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", qualified_name) or "P"


def render_mermaid_sequence(
    entry: EntryKey,
    steps: list[SequenceStep],
    layer_of: dict[str, Layer | None],
    entry_label: str,
    entry_return_type: str | None,
    external_participants: set[str],
) -> str:
    if not steps:
        return ""

    participants: list[str] = [CLIENT_PARTICIPANT]

    def add_participant(qname: str) -> None:
        if qname not in participants:
            participants.append(qname)

    add_participant(entry[0])
    for step in steps:
        add_participant(step.caller_class)
        add_participant(step.callee_class)

    def rank(p: str) -> int:
        if p == CLIENT_PARTICIPANT:
            return _CLIENT_RANK
        if p in external_participants:
            return _EXTERNAL_RANK
        return layer_rank(layer_of.get(p))

    def label(p: str) -> str:
        if p == CLIENT_PARTICIPANT or p in external_participants:
            return p
        return p.rsplit(".", 1)[-1]

    # Left-to-right layout follows Client -> architectural layer (API ->
    # service -> domain/data) -> external system, not raw call order —
    # e.g. a service that touches its repository before calling another
    # service still renders with the repository to the right. Ties (same
    # rank) keep call/first-seen order.
    first_seen_index = {p: i for i, p in enumerate(participants)}
    participants.sort(key=lambda p: (rank(p), first_seen_index[p]))

    lines = ["sequenceDiagram"]
    for p in participants:
        lines.append(f"    participant {_sanitize_id(p)} as {label(p)}")

    lines.append(f"    {_sanitize_id(CLIENT_PARTICIPANT)}->>{_sanitize_id(entry[0])}: {entry_label}")
    for step in steps:
        lines.append(f"    {_sanitize_id(step.caller_class)}->>{_sanitize_id(step.callee_class)}: {step.callee_method}()")

    # Return arrows unwind the call stack in reverse, each labeled with
    # the callee's real declared return type; void calls draw no return
    # arrow rather than a fabricated generic "ok".
    for step in reversed(steps):
        if step.return_type:
            lines.append(f"    {_sanitize_id(step.callee_class)}-->>{_sanitize_id(step.caller_class)}: {step.return_type}")
    if entry_return_type:
        lines.append(f"    {_sanitize_id(entry[0])}-->>{_sanitize_id(CLIENT_PARTICIPANT)}: {entry_return_type}")

    return "\n".join(lines)


def build_sequence_flows(
    repo_ir: RepoIR, index: SymbolIndex, external_systems: dict[str, list[str]] | None = None
) -> list[SequenceFlow]:
    edges = build_call_graph(repo_ir, index)
    edges = [e for e in edges if _is_architecturally_meaningful_call(e, index)]
    grouped = group_by_caller(edges)
    implementors = _build_implementors_map(repo_ir, index)
    layer_of = {qname: classify_layer(entry.class_info) for qname, entry in index.by_qualified_name.items()}
    if external_systems is None:
        external_systems = classify_external_systems(repo_ir)

    flows = []
    for entry in _select_entry_points(repo_ir, grouped):
        steps = _trace_flow(entry, grouped, index, implementors)
        steps = _append_external_terminal(steps, repo_ir, index, external_systems)
        external_participants = {s.callee_class for s in steps if s.callee_class in external_systems}
        entry_label = _route_label(entry[0], entry[1], index) or f"{entry[1]}()"
        entry_return_type = _resolve_return_type(entry[0], entry[1], index)
        mermaid = render_mermaid_sequence(entry, steps, layer_of, entry_label, entry_return_type, external_participants)
        flows.append(SequenceFlow(entry_class=entry[0], entry_method=entry[1], steps=steps, mermaid=mermaid))
    return flows
