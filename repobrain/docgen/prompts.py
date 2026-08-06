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

Package structure:
{_package_overview(ctx)}

Package-level dependency graph (Mermaid, `A --> B` means A depends on B):
```mermaid
{ctx.dependency_mermaid or 'graph LR'}
```

Representative types grouped by package:

{_render_cards(ctx.all_class_cards())}

Write ARCHITECTURE.md with: an Overview of the system's structure and
likely layering (e.g. model/service/repository/controller if that
pattern is evident from the package names and dependency directions —
do not assert a layered architecture if the evidence doesn't support
it), a Module/Package Breakdown describing each package's
responsibility as inferred from its classes, a Dependency Graph section
that reproduces the Mermaid diagram above, and a Design Observations
section noting any notable coupling, cycles, or architectural patterns
(e.g. repository pattern, dependency injection via constructors) visible
in the class shapes. Be precise and avoid speculation not grounded in
the provided structure."""


def _render_sequence_flows(ctx: ProjectContext) -> str:
    if not ctx.sequence_flows:
        return "(No resolvable multi-class call flows were found — the codebase may be too simple, or calls go through receivers that can't be statically resolved to a project class, such as local variables.)"

    blocks = []
    for flow in ctx.sequence_flows:
        entry = f"{flow.entry_class}.{flow.entry_method}()"
        steps = "\n".join(
            f"  {i + 1}. {s.caller_class}.{s.caller_method}() calls {s.callee_class}.{s.callee_method}()"
            for i, s in enumerate(flow.steps)
        )
        blocks.append(f"Flow starting at {entry}:\n{steps}\n```mermaid\n{flow.mermaid}\n```")
    return "\n\n".join(blocks)


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
