"""Builds a class- and package-level dependency graph from a RepoIR.

Edges are resolved conservatively: a raw type reference (a field type, a
method parameter, `extends`/`implements`, ...) becomes an edge only when
it can be tied to a class actually defined in the repository — via an
explicit import, same-package lookup, or an unambiguous simple-name
match. Everything else (JDK types, third-party libraries) is dropped
from the graph, though import package roots are still surfaced
separately for a "tech stack" summary.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from repobrain.analysis.symbol_extractor import SymbolIndex
from repobrain.ir.models import ClassInfo, FileIR, RepoIR

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_JAVA_LANG_BUILTINS = {
    "String", "Object", "Integer", "Long", "Double", "Float", "Boolean",
    "Byte", "Short", "Character", "Void", "int", "long", "double", "float",
    "boolean", "byte", "short", "char", "void", "var",
}


@dataclass
class DependencyEdge:
    target: str
    kinds: set[str] = field(default_factory=set)


@dataclass
class DependencyGraph:
    # qualified_name -> {target_qualified_name: DependencyEdge}
    edges: dict[str, dict[str, DependencyEdge]] = field(default_factory=dict)
    reverse_edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_edge(self, source: str, target: str, kind: str) -> None:
        if source == target:
            return
        bucket = self.edges.setdefault(source, {})
        edge = bucket.setdefault(target, DependencyEdge(target=target))
        edge.kinds.add(kind)
        self.reverse_edges[target].add(source)

    def dependencies_of(self, qualified_name: str) -> dict[str, DependencyEdge]:
        return self.edges.get(qualified_name, {})

    def dependents_of(self, qualified_name: str) -> set[str]:
        return self.reverse_edges.get(qualified_name, set())

    def package_graph(self, index: SymbolIndex) -> dict[str, set[str]]:
        pkg_edges: dict[str, set[str]] = defaultdict(set)
        for source, targets in self.edges.items():
            src_entry = index.by_qualified_name.get(source)
            if src_entry is None:
                continue
            for target in targets:
                tgt_entry = index.by_qualified_name.get(target)
                if tgt_entry is None:
                    continue
                if src_entry.package != tgt_entry.package:
                    pkg_edges[src_entry.package].add(tgt_entry.package)
        return pkg_edges


def _extract_simple_names(raw_type: str) -> list[str]:
    """Pull out identifier-looking tokens from a type string, e.g.
    "Map<String, List<Widget>>" -> ["Map", "String", "List", "Widget"]."""
    if not raw_type:
        return []
    cleaned = raw_type.replace("...", "").replace("[]", "")
    names = _IDENTIFIER_RE.findall(cleaned)
    return [n for n in names if n not in _JAVA_LANG_BUILTINS]


def resolve_type_reference(simple_or_qualified: str, file_ir: FileIR, index: SymbolIndex) -> str | None:
    """Resolve a raw type name (as it appears in source: a simple name, a
    qualified name, or generic/collection syntax that happens to reduce
    to one) to a project class's qualified name, via explicit import,
    same-package lookup, or an unambiguous simple-name match — or None if
    it's external (JDK/third-party) or ambiguous. Shared by dependency
    graph construction and call-graph resolution (`analysis/call_graph.py`).
    """
    if simple_or_qualified in index.by_qualified_name:
        return simple_or_qualified

    simple_name = simple_or_qualified.rsplit(".", 1)[-1]

    for imp in file_ir.imports:
        if imp.is_wildcard or imp.is_static:
            continue
        if imp.path.rsplit(".", 1)[-1] == simple_name and imp.path in index.by_qualified_name:
            return imp.path

    if file_ir.package:
        candidate = f"{file_ir.package}.{simple_name}"
        if candidate in index.by_qualified_name:
            return candidate

    unique_match = index.resolve_simple_name(simple_name)
    return unique_match.class_info.qualified_name if unique_match else None


def _addresolve_type_referenced_edges(graph: DependencyGraph, source_qname: str, raw_types: list[str], kind: str, file_ir: FileIR, index: SymbolIndex) -> None:
    for raw in raw_types:
        for simple in _extract_simple_names(raw):
            resolved = resolve_type_reference(simple, file_ir, index)
            if resolved:
                graph.add_edge(source_qname, resolved, kind)


def build_dependency_graph(repo_ir: RepoIR, index: SymbolIndex) -> DependencyGraph:
    graph = DependencyGraph()

    for file_ir in repo_ir.files.values():
        for class_info in file_ir.iter_classes():
            qname = class_info.qualified_name
            _addresolve_type_referenced_edges(graph, qname, class_info.extends, "extends", file_ir, index)
            _addresolve_type_referenced_edges(graph, qname, class_info.implements, "implements", file_ir, index)
            _addresolve_type_referenced_edges(graph, qname, [f.type for f in class_info.fields], "field", file_ir, index)
            for method in class_info.methods:
                _addresolve_type_referenced_edges(graph, qname, [p.type for p in method.parameters], "param", file_ir, index)
                _addresolve_type_referenced_edges(graph, qname, [method.return_type], "return", file_ir, index)
                _addresolve_type_referenced_edges(graph, qname, method.referenced_types, "creates", file_ir, index)

    return graph


def external_package_usage(repo_ir: RepoIR, top_n: int = 15) -> list[tuple[str, int]]:
    """Most-imported external (non-project) package roots, for a quick
    "technology stack" summary. Project-internal imports are excluded by
    comparing against the set of packages actually defined in the repo.
    """
    internal_packages = {f.package for f in repo_ir.files.values() if f.package}
    counter: Counter[str] = Counter()
    for file_ir in repo_ir.files.values():
        for imp in file_ir.imports:
            root = ".".join(imp.path.split(".")[:2])
            if any(pkg == root or pkg.startswith(root + ".") for pkg in internal_packages):
                continue
            counter[root] += 1
    return counter.most_common(top_n)


def to_mermaid(pkg_edges: dict[str, set[str]], orientation: str = "TD") -> str:
    """Renders a Mermaid `graph` block from an edge dict. Defaults to
    top-down (`TD`) — matches the conventional "request flows downward
    through layers" reading of an architecture diagram; pass `"LR"` for
    a left-to-right layout instead."""
    lines = [f"graph {orientation}"]
    seen_nodes: set[str] = set()

    def node_id(pkg: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", pkg) or "root"

    for src, targets in sorted(pkg_edges.items()):
        if src not in seen_nodes:
            lines.append(f'    {node_id(src)}["{src}"]')
            seen_nodes.add(src)
        for tgt in sorted(targets):
            if tgt not in seen_nodes:
                lines.append(f'    {node_id(tgt)}["{tgt}"]')
                seen_nodes.add(tgt)
            lines.append(f"    {node_id(src)} --> {node_id(tgt)}")
    return "\n".join(lines)
