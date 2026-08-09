"""Prompt templates for each generated document.

Every prompt is built entirely from the structured `ProjectContext` —
class cards, signatures, doc-comment summaries, dependency edges — never
from raw source. That's what keeps this pipeline local-first-safe: the
LLM only ever sees the shape of the code, not its implementation.
"""
from __future__ import annotations

from repobrain.docgen.context import ClassCard, ProjectContext, select_with_group_coverage

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


#: Floor so ARCHITECTURE.md's per-class detail section never gets
#: squeezed to nothing even if the other "verified fact" sections
#: (layer breakdown, domain model, primary flow) happen to be unusually
#: large for a given repo — better a small, correctly-bounded amount of
#: class detail than none at all.
MIN_CLASS_DETAIL_CHARS = 1500


def _fit_items_to_budget(items: list[str], budget: int, force_first: bool = True) -> tuple[str, int, int]:
    """Joins pre-rendered `items` in order, keeping as many as fit within
    `budget` characters. Returns (rendered_text, chars_used, num_kept);
    the caller compares `num_kept` against `len(items)` to know how many
    (if any) were omitted, and should note that in the surrounding text.

    `force_first`, when true, always keeps the first item even if it
    alone exceeds `budget`, so a single oversized entry doesn't wipe out
    the whole section. Pass `False` when calling this repeatedly across
    several independent groups that share one running budget (see
    `_render_component_classes`) — force-including a first item in
    *every* group can compound well past the intended total, since each
    group's "at least one" guarantee is applied independently of how
    much budget earlier groups already used.
    """
    kept: list[str] = []
    used = 0
    for item in items:
        cost = len(item) + 2  # + blank-line separator
        fits = used + cost <= budget
        if not fits and not (force_first and not kept):
            break
        kept.append(item)
        used += cost
    return "\n\n".join(kept), used, len(kept)


def _truncated_list(names: list[str], limit: int = 15) -> str:
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


#: Kept deliberately small and fixed rather than scaled to the model's
#: context window: this section is meant to be a compact "here are the
#: business nouns" fact block, not an exhaustive dump — the per-class
#: detail section below already carries the full, budget-aware listing
#: for every *other* layer. Without a firm cap here, a domain-heavy repo
#: (e.g. many JPA entities with dozens of columns) can blow the prompt
#: size out from this one section alone, regardless of `num_ctx`.
_MAX_DOMAIN_CLASSES_SHOWN = 12
_MAX_DOMAIN_FIELDS_SHOWN = 6


def _render_domain_model(ctx: ProjectContext) -> str:
    domain_cards = ctx.classes_by_layer("domain")
    if not domain_cards:
        return (
            "No domain/model types were identified — either this codebase has none, or "
            "none matched a recognized annotation (@Entity/@Table/@Document/@Embeddable) "
            "or data-carrying shape (a class/record with instance fields, or an enum)."
        )
    lines = ["Business/domain concepts (the nouns this system deals with), with their fields:"]
    ranked = sorted(domain_cards, key=lambda c: (-c.importance, c.qualified_name))
    shown = ranked[:_MAX_DOMAIN_CLASSES_SHOWN]
    for card in shown:
        fields = card.fields[:_MAX_DOMAIN_FIELDS_SHOWN]
        field_list = "; ".join(fields) if fields else "no fields"
        if len(card.fields) > len(fields):
            field_list += f"; and {len(card.fields) - len(fields)} more fields"
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


def _render_component_classes(ctx: ProjectContext, budget: int) -> str:
    """Per-class supporting detail, grouped by verified layer rather than
    by package — the narrative sections this feeds should be organized
    around responsibility the same way, treating this as backing
    evidence rather than a structure to reproduce.

    Domain-layer classes are intentionally excluded here: they're
    already covered, compactly, by `_render_domain_model` — a full
    `ClassCard.render()` (fields, dependencies, ...) for every domain
    class as well would just duplicate that section, and domain types
    are typically the most numerous class of type in any real codebase.

    Selection is `select_with_group_coverage` — the same mechanism
    `build_project_context` uses for its own, earlier, wider-budget pass
    — so a layer that already survived that first pass can't be silently
    zeroed out by this second, tighter one specific to ARCHITECTURE.md
    (see `build_architecture_prompt`, which carves `budget` out of
    `ctx.max_detail_chars` after accounting for every other "verified
    fact" section). Both passes being layer-coverage-aware is what
    actually closes the gap: a single layer-blind pass anywhere in the
    pipeline can starve a layer in a way no later pass can recover from.
    """
    groups = [(label, ctx.classes_by_layer(layer)) for layer, label in (("api", "API"), ("service", "Service"), ("data", "Data"))]
    groups.append(("Unclassified", [c for c in ctx.all_class_cards() if c.layer is None]))

    kept_by_label, omitted_total = select_with_group_coverage(groups, budget)

    sections = [f"#### {label} layer\n\n{_render_cards(cards)}" for label, cards in kept_by_label.items()]
    result = "\n\n".join(sections)
    if omitted_total:
        result += f"\n\n... and {omitted_total} more classes not shown in detail due to prompt size limits"
    return result


def build_readme_prompt(ctx: ProjectContext) -> str:
    # A fixed count cap (25 classes) isn't enough on its own -- a repo
    # whose public classes happen to carry many fields/methods each can
    # still produce an oversized prompt at only 25 entries. Bound the
    # rendered *characters* too, the same way ARCHITECTURE.md's sections
    # are, using a fraction of the shared detail budget (README's other
    # sections are small and fixed, so this doesn't need the same
    # extras-first accounting ARCHITECTURE.md's prompt does).
    public_classes = [c for c in ctx.all_class_cards() if "public" in c.modifiers]
    top_classes = sorted(public_classes, key=lambda c: (-c.importance, c.qualified_name))[:25]
    rendered = [c.render(include_dependencies=False) for c in top_classes]
    class_text, _, kept_count = _fit_items_to_budget(rendered, ctx.max_detail_chars // 2)
    if len(rendered) > kept_count:
        class_text += f"\n\n... and {len(rendered) - kept_count} more not shown in detail"

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

{class_text}

Write a README.md with these sections: a one-paragraph project overview
(infer the project's purpose from package and class names — be honest
that this is inferred), Features/Capabilities (bullet list grounded in
the classes above), Project Structure (the package list), Getting
Started (generic build/run guidance appropriate for a Java project —
do not invent specific build commands that aren't evidenced), and a
short Key Components section referencing the most central classes.
Keep it concise and skimmable."""


def build_architecture_prompt(ctx: ProjectContext) -> str:
    # `ctx.max_detail_chars` is the same budget context.py used to decide
    # how many classes made it into `ctx` at all — but every section below
    # besides the per-class listing (layer breakdown, external systems,
    # domain model, primary flow, component graph) was, until this fix,
    # stacked *on top of* that budget rather than sharing it, so a real
    # repo with many classes and long package/class names could blow the
    # assembled prompt out to ~2-3x the intended size regardless of
    # num_ctx. Computing the "extras" first and carving the per-class
    # section's budget out of what's left keeps the total bounded.
    layer_breakdown_text = _render_layer_breakdown(ctx)
    external_systems_text = _render_external_systems(ctx)
    domain_model_text = _render_domain_model(ctx)
    primary_flow_text = _render_primary_flow(ctx)
    component_mermaid_text = ctx.component_mermaid or "graph LR"

    extras_chars = sum(
        len(t) for t in (layer_breakdown_text, external_systems_text, domain_model_text, primary_flow_text, component_mermaid_text)
    )
    class_detail_budget = max(ctx.max_detail_chars - extras_chars, MIN_CLASS_DETAIL_CHARS)
    component_classes_text = _render_component_classes(ctx, class_detail_budget)

    return f"""Generate the content of ARCHITECTURE.md for the repository "{ctx.repo_name}".

Below are several VERIFIED structural facts (from static analysis, not inference), followed
by supporting per-class detail. Treat every fact marked VERIFIED FACT as ground truth —
never hedge on it or re-derive it yourself. Everything else (business capability names,
prose descriptions of what a component "does") is your inference from the evidence and
must read as such, not as another verified fact.

{layer_breakdown_text}

{external_systems_text}

Domain model — the business nouns this system deals with:
{domain_model_text}

Primary request flow — the highest-confidence entry point, already traced through the
call graph as a correct Mermaid `sequenceDiagram`:
{primary_flow_text}

VERIFIED FACT — component-level dependency graph (Mermaid; nodes are architectural
components and external systems, NOT individual packages or classes — a `-->` edge means
the source component depends on / calls into the target):
```mermaid
{component_mermaid_text}
```

Supporting per-class detail, grouped by the verified layer classification above (use this
as backing evidence for your prose — do not restructure your document around packages):

{component_classes_text}

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
    lines = [f"  1. Client calls {entry}"]
    for i, s in enumerate(flow.steps, start=2):
        lines.append(f"  {i}. {s.caller_class}.{s.caller_method}() calls {s.callee_class}.{s.callee_method}()")
    steps = "\n".join(lines)
    return (
        f"Flow starting when a client calls {entry}:\n{steps}\n"
        "(the Mermaid diagram below also shows the full return trip back to Client, "
        "including any external system reached at the end — reproduce it exactly)\n"
        f"```mermaid\n{flow.mermaid}\n```"
    )


def _render_sequence_flows(ctx: ProjectContext) -> str:
    if not ctx.sequence_flows:
        return "(No resolvable multi-class call flows were found — the codebase may be too simple, or calls go through receivers that can't be statically resolved to a project class, such as local variables.)"
    # Bounded the same way ARCHITECTURE.md's class-card section is: a
    # repo with long package/class names and deep call chains can make
    # even a handful of flows (each already capped by MAX_STEPS_PER_FLOW)
    # add up to a prompt far larger than `max_detail_chars` intends,
    # since that cap is on step *count*, not rendered character length.
    rendered = [_render_one_flow(flow) for flow in ctx.sequence_flows]
    text, _, kept_count = _fit_items_to_budget(rendered, ctx.max_detail_chars)
    omitted = len(rendered) - kept_count
    if omitted:
        text += f"\n\n... and {omitted} more flow(s) not shown due to prompt size limits"
    return text


def _render_primary_flow(ctx: ProjectContext) -> str:
    flow = ctx.primary_request_flow
    if flow is None:
        return "(No primary request flow could be identified from resolvable call chains.)"
    return _render_one_flow(flow)


def build_sequence_prompt(ctx: ProjectContext) -> str:
    return f"""Generate the content of SEQUENCE.md for the repository "{ctx.repo_name}": a set of
sequence diagrams for its most significant call flows.

Below are this repository's most significant call flows, each already rendered as a correct
Mermaid `sequenceDiagram` block from statically resolved method calls, along with the same
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
