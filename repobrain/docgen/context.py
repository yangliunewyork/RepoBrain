"""Builds compact, structured summaries of the IR for LLM prompts.

This is the boundary between "structured software knowledge" and the
model: prompts are built from these summaries, never from raw source
text, so nothing beyond symbol names, signatures, and doc comments is
ever sent to the LLM (local or otherwise).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from repobrain.analysis.dependency_analyzer import DependencyGraph, external_package_usage, to_mermaid
from repobrain.analysis.external_systems import classify_external_systems
from repobrain.analysis.layers import classify_layer, component_graph, layer_breakdown
from repobrain.analysis.symbol_extractor import ClassEntry, SymbolIndex
from repobrain.docgen.sequence import SequenceFlow, build_sequence_flows, is_route_handler
from repobrain.ir.models import ClassInfo, RepoIR

#: Absolute hard cap on how many classes ever get full detail, regardless
#: of character budget. Mainly a defensive backstop for the pathological
#: case of many packages each with a single tiny class, where the
#: per-package "at least one class" guarantee could otherwise add up to
#: more than the intended budget.
MAX_CLASSES_IN_DETAIL = 150

#: Fallback character budget when no LLM context size is supplied. ~3.3
#: chars/token is a conservative estimate for English/code text.
DEFAULT_MAX_DETAIL_CHARS = 12_000

#: Per-class caps so a single unusually large class (e.g. a 100-method
#: God class) can't blow the whole prompt budget on its own.
MAX_METHODS_PER_CARD = 40
MAX_FIELDS_PER_CARD = 30


def char_budget_for_num_ctx(num_ctx: int) -> int:
    """Convert a model's context window (in tokens) into a character
    budget for the *class detail* portion of a prompt.

    Reserves roughly half of num_ctx for the system prompt, per-doc
    instructions, the dependency diagram, and the model's own output —
    all of which sit outside the class cards themselves — so the full
    assembled prompt still comfortably fits inside num_ctx.
    """
    usable_tokens = max(int(num_ctx * 0.5), 500)
    return usable_tokens * 3


def _first_line(doc: str | None) -> str | None:
    if not doc:
        return None
    text = doc.strip().lstrip("/*").rstrip("*/").strip()
    for line in text.splitlines():
        stripped = line.strip().lstrip("*").strip()
        if stripped and not stripped.startswith("@"):
            return stripped
    return None


@dataclass
class ClassCard:
    qualified_name: str
    kind: str
    modifiers: list[str]
    package: str
    summary: str | None
    extends: list[str]
    implements: list[str]
    fields: list[str]  # "type name"
    methods: list[str]  # signatures
    depends_on: list[str]
    depended_on_by: list[str]
    #: "api" | "service" | "data" | None, from `analysis.layers.classify_layer`
    #: — verified from framework annotations (or a name-keyword fallback),
    #: not left for the LLM to infer.
    layer: str | None = None
    #: How architecturally central this class is — see `_compute_importance`.
    #: Drives which classes survive truncation first, both in the primary
    #: per-layer selection below and in `docgen.prompts._render_component_classes`'s
    #: secondary, doc-specific pass: the classes dropped when a budget
    #: runs out should be the least important ones, not just whichever
    #: happened to sort last alphabetically.
    importance: float = 0.0

    def render(self, include_dependencies: bool = True) -> str:
        lines = [f"### {self.kind} {self.qualified_name}"]
        if self.layer:
            lines.append(f"- layer: {self.layer}")
        if self.modifiers:
            lines.append(f"- modifiers: {', '.join(self.modifiers)}")
        if self.extends:
            lines.append(f"- extends: {', '.join(self.extends)}")
        if self.implements:
            lines.append(f"- implements: {', '.join(self.implements)}")
        if self.summary:
            lines.append(f"- summary: {self.summary}")
        if self.fields:
            lines.append(f"- fields: {'; '.join(self.fields)}")
        if self.methods:
            lines.append(f"- methods:\n  - " + "\n  - ".join(self.methods))
        if include_dependencies and self.depends_on:
            lines.append(f"- depends on: {', '.join(self.depends_on)}")
        if include_dependencies and self.depended_on_by:
            lines.append(f"- used by: {', '.join(self.depended_on_by)}")
        return "\n".join(lines)


@dataclass
class PackageSummary:
    name: str
    class_count: int
    classes: list[ClassCard]


@dataclass
class ProjectContext:
    repo_name: str
    languages: list[str]
    file_count: int
    class_count: int
    package_summaries: list[PackageSummary]
    #: Mermaid `sequenceDiagram`-style component graph — API / Service /
    #: Domain / Data / Other plus any detected external-system category
    #: — from `analysis.layers.component_graph`. Deliberately *not* a
    #: per-package graph: package structure is an implementation detail,
    #: not an architectural one.
    component_mermaid: str
    external_packages: list[tuple[str, int]]
    #: External systems/integrations detected from real imports (JDBC/JPA,
    #: message queues, HTTP clients, cloud SDKs, ...), category -> the
    #: recognized library prefixes actually found. See
    #: `analysis.external_systems.classify_external_systems`.
    external_systems: dict[str, list[str]]
    sequence_flows: list[SequenceFlow]
    #: Qualified names grouped by "api"/"service"/"domain"/"data"/
    #: "unclassified", computed over *every* class regardless of prompt
    #: truncation — see `analysis.layers.layer_breakdown`.
    layer_breakdown: dict[str, list[str]]
    truncated: bool
    #: The same character budget used to bound the class-card listing
    #: above, exposed so prompt builders can share it across every
    #: variable-length section they add (layer breakdown, domain model,
    #: primary flow, ...) instead of each one stacking uncapped content
    #: on top of a budget that was only ever meant to bound class cards.
    max_detail_chars: int

    def all_class_cards(self) -> list[ClassCard]:
        return [c for pkg in self.package_summaries for c in pkg.classes]

    def classes_by_layer(self, layer: str) -> list[ClassCard]:
        return [c for c in self.all_class_cards() if c.layer == layer]

    @property
    def primary_request_flow(self) -> SequenceFlow | None:
        """The highest-confidence sequence flow (route-annotated entry
        points rank first, see `docgen.sequence._select_entry_points`),
        used as ARCHITECTURE.md's "how does a request actually flow
        through this system" anchor."""
        return self.sequence_flows[0] if self.sequence_flows else None


def _capped(items: list[str], limit: int) -> list[str]:
    if len(items) <= limit:
        return items
    return items[:limit] + [f"... ({len(items) - limit} more not shown)"]


#: Weights for `_compute_importance`. Fan-in ("how many things depend on
#: me") is weighted above fan-out ("how many things I depend on") since
#: it's the stronger signal of architectural centrality — a class lots of
#: other code relies on is more load-bearing than one that merely calls
#: a lot of things. The entry-point bonus is a flat addition rather than
#: another multiplier: entry points (controllers, handlers) often have
#: near-zero fan-in (nothing in the codebase calls them — the framework
#: does, from outside) but are still exactly the classes a reader needs
#: to see first, so their importance can't depend on fan-in alone.
_FAN_IN_WEIGHT = 3
_FAN_OUT_WEIGHT = 1
_ENTRY_POINT_BONUS = 15


def _compute_importance(info: ClassInfo, graph: DependencyGraph) -> float:
    """How architecturally central a class is, from signals that are
    computed, not guessed: fan-in/fan-out on the *raw* dependency graph
    (not the display-capped lists on `ClassCard` — a class with 200
    dependents must still outrank one with 2, which capping both to the
    same shown-list length would otherwise hide), plus a bonus for
    classes that verifiably host an entry point (a route/handler
    annotation — see `docgen.sequence.is_route_handler`).
    """
    fan_in = len(graph.dependents_of(info.qualified_name))
    fan_out = len(graph.dependencies_of(info.qualified_name))
    score = fan_in * _FAN_IN_WEIGHT + fan_out * _FAN_OUT_WEIGHT
    if is_route_handler(info):
        score += _ENTRY_POINT_BONUS
    return score


def select_with_group_coverage(
    groups: list[tuple[str, list[ClassCard]]], budget: int, max_count: int | None = None
) -> tuple[dict[str, list[ClassCard]], int]:
    """Selects which cards to keep across several groups (e.g. layers),
    guaranteeing every non-empty group at least one card — its most
    important, even if it alone exceeds that group's fair share — by
    giving each group its own even slice of `budget` up front, with
    unused slack from earlier groups carried forward to later ones.
    Groups are processed in the order given, so pass them pre-ordered if
    a particular tie-breaking order matters. Within each group, cards
    are ranked by `ClassCard.importance` descending.

    `max_count`, if given, is likewise split evenly across groups (with
    the same carry-forward), bounding total *class count* the way
    `budget` bounds total *character count* — a defensive backstop for
    many packages/layers each holding just one tiny class, where char
    budget alone wouldn't meaningfully limit how many classes pile up.

    Returns (kept_by_label, omitted_count) — a group with everything
    dropped is simply absent from `kept_by_label` rather than present
    with an empty list. This is the one selection mechanism shared by
    `build_project_context` (primary pass, across all docs) and
    `docgen.prompts._render_component_classes` (secondary, tighter pass
    specific to ARCHITECTURE.md) — using it in both places is what keeps
    a layer from being starved at one stage in a way the other stage can
    no longer recover from.
    """
    groups = [(label, cards) for label, cards in groups if cards]
    if not groups:
        return {}, 0

    per_group_budget = max(budget // len(groups), 1)
    per_group_count = max(max_count // len(groups), 1) if max_count is not None else None

    kept_by_label: dict[str, list[ClassCard]] = {}
    omitted = 0
    chars_leftover = 0
    count_leftover = 0
    for label, cards in groups:
        ranked = sorted(cards, key=lambda c: (-c.importance, c.qualified_name))
        group_budget = per_group_budget + chars_leftover
        group_count_limit = (per_group_count + count_leftover) if per_group_count is not None else None

        used = 0
        group_kept: list[ClassCard] = []
        for card in ranked:
            if group_count_limit is not None and len(group_kept) >= group_count_limit:
                break
            cost = len(card.render()) + 2
            if group_kept and used + cost > group_budget:
                break
            group_kept.append(card)
            used += cost

        if group_kept:
            kept_by_label[label] = group_kept
        omitted += len(cards) - len(group_kept)
        chars_leftover = max(group_budget - used, 0)
        if group_count_limit is not None:
            count_leftover = max(group_count_limit - len(group_kept), 0)

    return kept_by_label, omitted


def _build_class_card(entry: ClassEntry, graph: DependencyGraph) -> ClassCard:
    info: ClassInfo = entry.class_info
    fields = _capped([f"{f.type} {f.name}" for f in info.fields], MAX_FIELDS_PER_CARD)
    methods = _capped([m.signature for m in info.methods], MAX_METHODS_PER_CARD)
    depends_on = _capped(sorted(graph.dependencies_of(info.qualified_name).keys()), MAX_METHODS_PER_CARD)
    depended_on_by = _capped(sorted(graph.dependents_of(info.qualified_name)), MAX_METHODS_PER_CARD)
    return ClassCard(
        qualified_name=info.qualified_name,
        kind=info.kind,
        modifiers=info.modifiers,
        package=entry.package,
        summary=_first_line(info.doc_comment),
        extends=info.extends,
        implements=info.implements,
        fields=fields,
        methods=methods,
        depends_on=depends_on,
        depended_on_by=depended_on_by,
        layer=classify_layer(info),
        importance=_compute_importance(info, graph),
    )


_LAYER_GROUPS: tuple[tuple[str, str], ...] = (
    ("api", "API"), ("service", "Service"), ("domain", "Domain"), ("data", "Data"),
)


def build_project_context(
    repo_root_name: str,
    repo_ir: RepoIR,
    index: SymbolIndex,
    graph: DependencyGraph,
    languages: list[str],
    max_detail_chars: int = DEFAULT_MAX_DETAIL_CHARS,
) -> ProjectContext:
    total_classes = len(index.by_qualified_name)

    # Every class gets a full card up front — cheap (field/list
    # construction, no truncation decision yet) — so the primary
    # selection below can see the whole repository, not just whatever an
    # earlier package-level pass happened to keep. Package structure is
    # deliberately *not* the primary selection axis: a layer's classes
    # can be scattered thinly across many packages, and truncating by
    # package first can zero out an entire layer before layer-based
    # logic ever gets a chance to notice, let alone protect it.
    all_cards = [_build_class_card(entry, graph) for entry in index.by_qualified_name.values()]
    cards_by_layer: dict[str | None, list[ClassCard]] = {}
    for card in all_cards:
        cards_by_layer.setdefault(card.layer, []).append(card)

    groups = [(label, cards_by_layer.get(layer, [])) for layer, label in _LAYER_GROUPS]
    groups.append(("Unclassified", cards_by_layer.get(None, [])))

    kept_by_label, omitted = select_with_group_coverage(groups, max_detail_chars, max_count=MAX_CLASSES_IN_DETAIL)
    selected_by_qname = {c.qualified_name: c for cards in kept_by_label.values() for c in cards}

    package_summaries: list[PackageSummary] = []
    for package, entries in sorted(index.by_package.items()):
        classes = sorted(
            (selected_by_qname[e.class_info.qualified_name] for e in entries if e.class_info.qualified_name in selected_by_qname),
            key=lambda c: c.qualified_name,
        )
        package_summaries.append(PackageSummary(name=package or "(default package)", class_count=len(entries), classes=classes))

    truncated = omitted > 0

    external_systems = classify_external_systems(repo_ir)
    return ProjectContext(
        repo_name=repo_root_name,
        languages=languages,
        file_count=len(repo_ir.files),
        class_count=total_classes,
        package_summaries=package_summaries,
        component_mermaid=to_mermaid(component_graph(repo_ir, index, graph, external_systems)),
        external_packages=external_package_usage(repo_ir),
        external_systems=external_systems,
        sequence_flows=build_sequence_flows(repo_ir, index),
        layer_breakdown=layer_breakdown([e.class_info for e in index.by_qualified_name.values()]),
        truncated=truncated,
        max_detail_chars=max_detail_chars,
    )
