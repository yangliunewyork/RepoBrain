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
from repobrain.docgen.sequence import SequenceFlow, build_sequence_flows
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

    # Split the budget evenly across packages up front, rather than
    # first-come-first-served, so a handful of huge early packages can't
    # starve every package that sorts after them — while still keeping the
    # *total* bounded to roughly max_detail_chars regardless of package count.
    num_packages = max(len(index.by_package), 1)
    per_package_budget = max(max_detail_chars // num_packages, 1)

    package_summaries: list[PackageSummary] = []
    detailed_count = 0
    for package, entries in sorted(index.by_package.items()):
        cards = []
        remaining_chars = per_package_budget
        for entry in sorted(entries, key=lambda e: e.class_info.qualified_name):
            if detailed_count >= MAX_CLASSES_IN_DETAIL:
                break
            # Always include at least one class per package so every
            # package gets some representation, even if that one class
            # alone exceeds this package's share of the budget.
            if cards and remaining_chars <= 0:
                break
            card = _build_class_card(entry, graph)
            rendered_len = len(card.render())
            if cards and rendered_len > remaining_chars:
                break
            cards.append(card)
            remaining_chars -= rendered_len
            detailed_count += 1
        package_summaries.append(PackageSummary(name=package or "(default package)", class_count=len(entries), classes=cards))

    truncated = detailed_count < total_classes

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
    )
