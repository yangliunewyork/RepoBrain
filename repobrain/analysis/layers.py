"""Architectural layer classification for a class.

Used to order sequence-diagram participants left-to-right (API ->
service -> domain -> data), to aggregate the class-level dependency
graph into a small, legible component graph (see `component_graph`
below), and to give ARCHITECTURE.md a verified layer breakdown instead
of leaving an 8B model to guess purely from package/class names.

Classification prefers framework annotations — a far more reliable
signal than name substrings, since `@RestController`/`@Service`/
`@Repository`/`@Entity` *are* the layer, not a naming convention someone
might or might not have followed. Keyword matching on the qualified name
is the next fallback, then a structural heuristic for domain/model types
(see `_looks_like_domain_model`). Classes matching none of these stay
unclassified (`None`) rather than being forced into a guessed bucket.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from repobrain.ir.models import ClassInfo

if TYPE_CHECKING:
    from repobrain.analysis.dependency_analyzer import DependencyGraph
    from repobrain.analysis.symbol_extractor import SymbolIndex
    from repobrain.ir.models import RepoIR

Layer = Literal["api", "service", "domain", "data"]

#: Checked in this order: these four annotation sets are mutually
#: exclusive stereotypes, so the order between them doesn't matter much
#: — "Service" is checked last only because it's the most generic name
#: and least likely to collide with the other three anyway.
_API_ANNOTATIONS = {
    "RestController", "Controller", "RequestMapping", "GetMapping", "PostMapping",
    "PutMapping", "DeleteMapping", "PatchMapping",
    "Path", "GET", "POST", "PUT", "DELETE",  # JAX-RS
}
#: Domain/model stereotypes: these mark a class as a data *structure*
#: (a business noun), not a runtime component that does something.
_DOMAIN_ANNOTATIONS = {"Entity", "Table", "Document", "Embeddable", "MappedSuperclass"}
#: Data-access stereotypes: a runtime component that reads/writes domain
#: objects, distinct from the domain objects themselves.
_DATA_ANNOTATIONS = {"Repository", "Dao"}
_SERVICE_ANNOTATIONS = {"Service"}

#: Fallback for code with no recognized annotations at all.
_API_LAYER_KEYWORDS = ("controller", "endpoint", "resource", "router", "api", "rest", "web")
_DATA_LAYER_KEYWORDS = ("repository", "repo", "dao", "persistence", "database", "mapper", "store")

_LAYER_RANK: dict[Layer | None, int] = {"api": 0, "service": 1, "domain": 2, "data": 3}
#: Truly unclassified code (utility/config/helper classes with no
#: annotation, keyword, or domain-shape signal) sits alongside "domain"
#: rather than being pinned to either edge.
_DEFAULT_RANK = 2

_LAYER_LABELS: dict[Layer | None, str] = {"api": "API", "service": "Service", "domain": "Domain", "data": "Data"}
_OTHER_LABEL = "Other"


def _looks_like_domain_model(class_info: ClassInfo) -> bool:
    """Structural fallback for domain/model types with no framework
    annotation at all (plain POJOs, Java records used as value objects).
    An enum is treated as domain-ish unconditionally (its constants are
    themselves a business vocabulary, e.g. OrderStatus.SHIPPED); classes
    and records qualify if they carry at least one *instance* field —
    static-only fields are how RepoBrain's IR represents a "constants"
    utility class or a class made entirely of static helpers, neither of
    which is a domain object.
    """
    if class_info.kind == "enum":
        return True
    return any("static" not in f.modifiers for f in class_info.fields)


def classify_layer(class_info: ClassInfo) -> Layer | None:
    """Best-effort architectural layer for one class, or None if nothing
    — annotation, keyword, or domain-shape — gives a signal."""
    annotations = set(class_info.annotations)
    for method in class_info.methods:
        # Catches a plain class whose routes are annotated at the method
        # level (`@GetMapping` on a method) without a class-level
        # `@RestController`/`@Controller` — less common, but free to check.
        annotations.update(method.annotations)

    if annotations & _API_ANNOTATIONS:
        return "api"
    if annotations & _DOMAIN_ANNOTATIONS:
        return "domain"
    if annotations & _DATA_ANNOTATIONS:
        return "data"
    if annotations & _SERVICE_ANNOTATIONS:
        return "service"

    lowered = class_info.qualified_name.lower()
    if any(k in lowered for k in _API_LAYER_KEYWORDS):
        return "api"
    if any(k in lowered for k in _DATA_LAYER_KEYWORDS):
        return "data"

    if _looks_like_domain_model(class_info):
        return "domain"
    return None


def layer_rank(layer: Layer | None) -> int:
    """Sort key for left-to-right diagram placement: api < service < domain/other < data."""
    return _LAYER_RANK.get(layer, _DEFAULT_RANK)


def layer_breakdown(class_infos: list[ClassInfo]) -> dict[str, list[str]]:
    """Groups qualified names by layer, for surfacing a verified
    "here's what we found" fact block in ARCHITECTURE.md. Keys are
    "api", "service", "domain", "data", "unclassified"."""
    groups: dict[str, list[str]] = {"api": [], "service": [], "domain": [], "data": [], "unclassified": []}
    for info in class_infos:
        layer = classify_layer(info)
        groups[layer or "unclassified"].append(info.qualified_name)
    return groups


def component_graph(
    repo_ir: "RepoIR",
    index: "SymbolIndex",
    graph: "DependencyGraph",
    external_systems: dict[str, list[str]] | None = None,
) -> dict[str, set[str]]:
    """Aggregates the class-level dependency graph into a small
    component-level graph — API / Service / Domain / Data / Other, plus
    a node per external-system category actually detected in this repo's
    imports (see `external_systems.classify_external_systems`).

    This is a deliberately coarser, more "architectural" view than
    `DependencyGraph.package_graph`: a repo's package structure often has
    far more nodes than it has meaningfully distinct *responsibilities*,
    and package names alone say nothing about what's actually external
    to the system (a database, a message queue, a third-party API).
    """
    def bucket(qname: str) -> str:
        entry = index.by_qualified_name.get(qname)
        if entry is None:
            return _OTHER_LABEL
        return _LAYER_LABELS.get(classify_layer(entry.class_info), _OTHER_LABEL)

    edges: dict[str, set[str]] = defaultdict(set)
    for source, targets in graph.edges.items():
        source_bucket = bucket(source)
        for target in targets:
            target_bucket = bucket(target)
            if source_bucket != target_bucket:
                edges[source_bucket].add(target_bucket)

    if external_systems:
        prefix_to_category = {prefix: category for category, prefixes in external_systems.items() for prefix in prefixes}
        for qname, entry in index.by_qualified_name.items():
            file_ir = repo_ir.files.get(entry.file_path)
            if file_ir is None:
                continue
            source_bucket = bucket(qname)
            for imp in file_ir.imports:
                for prefix, category in prefix_to_category.items():
                    if imp.path.startswith(prefix):
                        edges[source_bucket].add(category)

    return dict(edges)
