"""Prompt templates for each generated document.

Every prompt is built entirely from the structured `ProjectContext` —
class cards, signatures, doc-comment summaries, dependency edges — never
from raw source. That's what keeps this pipeline local-first-safe: the
LLM only ever sees the shape of the code, not its implementation.
"""
from __future__ import annotations

from repobrain.docgen.context import ClassCard, ProjectContext

SYSTEM_PROMPT = (
    "You are a senior software engineer writing technical documentation "
    "for a Java codebase. You are given a structured summary of the "
    "code's classes, signatures, and dependencies — not the raw source. "
    "Write clear, accurate, professional Markdown. Never invent classes, "
    "methods, or behavior that is not present in the provided context. "
    "Do not include a title heading matching the document type unless "
    "asked to; do not wrap the output in code fences."
)


def _package_overview(ctx: ProjectContext) -> str:
    lines = []
    for pkg in ctx.package_summaries:
        lines.append(f"- `{pkg.name}` ({pkg.class_count} types)")
    return "\n".join(lines)


def _render_cards(cards: list[ClassCard], include_dependencies: bool = True) -> str:
    return "\n\n".join(c.render(include_dependencies=include_dependencies) for c in cards)


def _truncated_list(names: list[str], limit: int = 30) -> str:
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", and {len(names) - limit} more"


def _render_layer_breakdown(ctx: ProjectContext) -> str:
    api = ctx.layer_breakdown.get("api", [])
    service = ctx.layer_breakdown.get("service", [])
    domain = ctx.layer_breakdown.get("domain", [])
    data = ctx.layer_breakdown.get("data", [])
    unclassified = ctx.layer_breakdown.get("unclassified", [])

    if not api and not service and not data and not domain:
        return (
            "VERIFIED FACT: no framework annotations (e.g. @RestController/@Service/"
            "@Repository/@Entity) or layer-suggestive naming/shape were found anywhere in "
            "this codebase. Do not assert a layered (API/service/data) architecture unless "
            "the dependency graph clearly shows one on its own."
        )

    lines = [
        "VERIFIED FACT — Runtime component / layer breakdown (from actual framework "
        "annotations: @RestController/@Controller/@*Mapping -> API, @Service -> Service, "
        "@Entity/@Table/@Document -> Domain, @Repository/@Dao -> Data; a name-keyword or "
        "data-shape fallback applies only where no annotation was present). Treat this as "
        "ground truth, not something to re-infer:"
    ]
    if api:
        lines.append(f"- API layer (handles incoming requests): {_truncated_list(api)}")
    if service:
        lines.append(f"- Service layer (business logic): {_truncated_list(service)}")
    if domain:
        lines.append(f"- Domain layer (business data/model types): {_truncated_list(domain)}")
    if data:
        lines.append(f"- Data layer (persistence/data access): {_truncated_list(data)}")
    if unclassified:
        lines.append(
            f"- Unclassified (no annotation/keyword/shape match — likely utility, "
            f"configuration, or infrastructure code): {len(unclassified)} types"
        )
    return "\n".join(lines)


def _render_external_systems(ctx: ProjectContext) -> str:
    if not ctx.external_systems:
        return (
            "VERIFIED FACT: no recognized external-system integration was found in this "
            "codebase's imports. This may mean the system is self-contained, or that it "
            "integrates with something RepoBrain doesn't yet recognize — do not guess "
            "which; say plainly that none were detected."
        )
    lines = ["VERIFIED FACT — external systems/integrations, detected from actual imports:"]
    for category, prefixes in sorted(ctx.external_systems.items()):
        lines.append(f"- {category} (via {', '.join(prefixes)})")
    return "\n".join(lines)


def _render_domain_model(ctx: ProjectContext) -> str:
    domain_cards = ctx.classes_by_layer("domain")
    if not domain_cards:
        return (
            "No domain/model types were identified — either this codebase has none, or "
            "none matched a recognized annotation (@Entity/@Table/@Document/@Embeddable) "
            "or data-carrying shape (a class/record with instance fields, or an enum)."
        )
    lines = ["Business/domain concepts (the nouns this system deals with), with their fields:"]
    shown = domain_cards[:30]
    for card in shown:
        field_list = "; ".join(card.fields) if card.fields else "no fields"
        summary = f" — {card.summary}" if card.summary else ""
        lines.append(f"- **{card.qualified_name}** ({card.kind}): {field_list}{summary}")
    if len(domain_cards) > len(shown):
        lines.append(f"- ...and {len(domain_cards) - len(shown)} more domain types not shown in detail")
    lines.append(
        "(These are the fields/constants actually declared — nothing here says how or when "
        "any of them are set. Do not invent a state-transition story, e.g. for an enum like "
        "OrderStatus, unless you can point to an actual method elsewhere in this context that "
        "performs that transition.)"
    )
    return "\n".join(lines)


def _render_component_classes(ctx: ProjectContext) -> str:
    """Per-class supporting detail, grouped by verified layer rather than
    by package — the narrative sections this feeds should be organized
    around responsibility the same way, treating this as backing
    evidence rather than a structure to reproduce."""
    sections = []
    for layer, label in (("api", "API"), ("service", "Service"), ("domain", "Domain"), ("data", "Data")):
        cards = ctx.classes_by_layer(layer)
        if cards:
            sections.append(f"#### {label} layer\n\n{_render_cards(cards)}")
    other = [c for c in ctx.all_class_cards() if c.layer is None]
    if other:
        sections.append(f"#### Unclassified\n\n{_render_cards(other)}")
    return "\n\n".join(sections)


def build_readme_prompt(ctx: ProjectContext) -> str:
    top_classes = [c for c in ctx.all_class_cards() if "public" in c.modifiers][:25]
    external = ", ".join(name for name, _ in ctx.external_packages[:10]) or "none detected"
    truncation_note = (
        "\nNote: this is a large codebase; only a representative subset of classes is listed below."
        if ctx.truncated
        else ""
    )
    return f"""Generate the content of a project README.md for the repository "{ctx.repo_name}".

Project facts:
- Languages analyzed: {', '.join(ctx.languages)}
- Files analyzed: {ctx.file_count}
- Types (classes/interfaces/enums/records) discovered: {ctx.class_count}
- Packages:
{_package_overview(ctx)}
- Key external dependencies (by import frequency): {external}
{truncation_note}

Representative public types:

{_render_cards(top_classes, include_dependencies=False)}

Write a README.md with these sections: a one-paragraph project overview
(infer the project's purpose from package and class names — be honest
that this is inferred), Features/Capabilities (bullet list grounded in
the classes above), Project Structure (the package list), Getting
Started (generic build/run guidance appropriate for a Java project —
do not invent specific build commands that aren't evidenced), and a
short Key Components section referencing the most central classes.
Keep it concise and skimmable."""


def build_architecture_prompt(ctx: ProjectContext) -> str:
    return f"""Generate the content of ARCHITECTURE.md for the repository "{ctx.repo_name}".

Below are several VERIFIED structural facts (from static analysis, not inference), followed
by supporting per-class detail. Treat every fact marked VERIFIED FACT as ground truth —
never hedge on it or re-derive it yourself. Everything else (business capability names,
prose descriptions of what a component "does") is your inference from the evidence and
must read as such, not as another verified fact.

{_render_layer_breakdown(ctx)}

{_render_external_systems(ctx)}

Domain model — the business nouns this system deals with:
{_render_domain_model(ctx)}

Primary request flow — the highest-confidence entry point, already traced through the
call graph as a correct Mermaid `sequenceDiagram`:
{_render_primary_flow(ctx)}

VERIFIED FACT — component-level dependency graph (Mermaid; nodes are architectural
components and external systems, NOT individual packages or classes — a `-->` edge means
the source component depends on / calls into the target):
```mermaid
{ctx.component_mermaid or 'graph LR'}
```

Supporting per-class detail, grouped by the verified layer classification above (use this
as backing evidence for your prose — do not restructure your document around packages):

{_render_component_classes(ctx)}

---

Write ARCHITECTURE.md organized entirely around responsibility and runtime behavior —
never structure a section around the Java package layout. Include these sections, in
this order, each answering its question directly:

1. **Overview** — What does this service actually do? Ground this in the domain model,
   the service-layer method names, and the primary request flow above; do not just
   restate file/class counts.
2. **Business Capabilities** — What are its major business capabilities? Phrase as
   capabilities ("manages widget inventory", "processes payments"), inferred from
   service-layer method names and the domain model — not a class listing.
3. **Runtime Components** — What are its major runtime components? Describe what each
   verified layer (API/Service/Domain/Data) is responsible for in this specific codebase.
4. **External Systems** — What external systems does it communicate with? State the
   verified external-systems fact above; if none were detected, say so plainly.
5. **Data Entry and Exit** — Where does data enter and leave the system? Identify this
   from the API layer (entry) and the data/external-systems layer (exit/persistence),
   grounded in the component graph's edges.
6. **Primary Request Flow** — Walk through the primary request flow above in 2-4
   sentences of prose, then reproduce its Mermaid diagram exactly as given.
7. **Component Dependency Graph** — Reproduce the component-level Mermaid diagram above
   exactly, with a short explanation of the major dependency directions.
8. **Design Observations** — Any notable coupling, cycles, or architectural patterns
   (e.g. repository pattern, dependency injection via constructors) visible in the
   component graph or class shapes.

Be precise and avoid speculation not grounded in the evidence above."""


def _render_one_flow(flow) -> str:
    entry = f"{flow.entry_class}.{flow.entry_method}()"
    steps = "\n".join(
        f"  {i + 1}. {s.caller_class}.{s.caller_method}() calls {s.callee_class}.{s.callee_method}()"
        for i, s in enumerate(flow.steps)
    )
    return f"Flow starting at {entry}:\n{steps}\n```mermaid\n{flow.mermaid}\n```"


def _render_sequence_flows(ctx: ProjectContext) -> str:
    if not ctx.sequence_flows:
        return "(No resolvable multi-class call flows were found — the codebase may be too simple, or calls go through receivers that can't be statically resolved to a project class, such as local variables.)"
    return "\n\n".join(_render_one_flow(flow) for flow in ctx.sequence_flows)


def _render_primary_flow(ctx: ProjectContext) -> str:
    flow = ctx.primary_request_flow
    if flow is None:
        return "(No primary request flow could be identified from resolvable call chains.)"
    return _render_one_flow(flow)


def build_sequence_prompt(ctx: ProjectContext) -> str:
    return f"""Generate the content of SEQUENCE.md for the repository "{ctx.repo_name}": a set of
sequence diagrams for its most significant call flows.

Below are {len(ctx.sequence_flows)} call flows, each already rendered as a correct Mermaid
`sequenceDiagram` block from statically resolved method calls, along with the same
information as a plain numbered step list. Entry points were chosen as public methods
that aren't themselves called elsewhere in the codebase and that make the most outgoing
calls — a heuristic for "likely meaningful workflow", not a guarantee.

{_render_sequence_flows(ctx)}

Write SEQUENCE.md with one section per flow (heading: the entry point's class and method).
For each section: reproduce the Mermaid code block exactly as given (do not alter arrows,
participant names, or add/remove steps), followed by 1-3 sentences of plain-English
description of what the flow does, grounded only in the class/method names shown — do not
invent behavior the names don't support. If there are no flows, say plainly that no
resolvable cross-class call chains were found rather than fabricating one."""
