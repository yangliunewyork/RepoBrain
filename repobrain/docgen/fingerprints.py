"""Structural fingerprints used to decide which documents actually need
regenerating after an incremental update.

Re-parsing changed files is cheap; calling the LLM is not. So after a
`repobrain update`, we recompute one fingerprint per document type from
the *merged* IR and compare it against the fingerprint stored from the
last run. A document is only regenerated if its fingerprint changed —
e.g. editing a method body without changing which methods it calls
doesn't change SEQUENCE.md's fingerprint, so that doc is skipped.
"""
from __future__ import annotations

import hashlib
import json

from repobrain.analysis.dependency_analyzer import DependencyGraph, external_package_usage
from repobrain.analysis.external_systems import classify_external_systems
from repobrain.analysis.layers import layer_breakdown
from repobrain.analysis.symbol_extractor import ClassEntry, SymbolIndex
from repobrain.docgen.sequence import SequenceFlow, build_sequence_flows
from repobrain.ir.models import RepoIR


def _hash(obj) -> str:
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _class_shape(entry: ClassEntry, graph: DependencyGraph) -> dict:
    """Everything about one class that ARCHITECTURE.md's per-class card
    (`ClassCard.render`) actually shows, so any change visible in that
    rendered card — a new field, a new dependency edge, a newly added
    class entirely — is guaranteed to invalidate the fingerprint. Package-
    level edges alone under-tracked this: a same-package dependency or a
    brand-new class with no cross-package impact could change the
    prompt's "Representative types" section while leaving `package_edges`
    untouched.
    """
    info = entry.class_info
    return {
        "qualified_name": info.qualified_name,
        "kind": info.kind,
        "modifiers": sorted(info.modifiers),
        "annotations": sorted(info.annotations),
        "extends": sorted(info.extends),
        "implements": sorted(info.implements),
        "fields": sorted(f"{f.type} {f.name}" for f in info.fields),
        "methods": sorted(m.signature for m in info.methods),
        "depends_on": sorted(graph.dependencies_of(info.qualified_name).keys()),
        "depended_on_by": sorted(graph.dependents_of(info.qualified_name)),
    }


def _flow_shape(flow: SequenceFlow) -> str:
    # The rendered Mermaid text already reflects everything that could
    # possibly change about a flow — call chain, route label, resolved
    # return types, an appended external-system hop — so hashing it
    # directly stays automatically correct as `sequence.py`'s rendering
    # evolves, rather than needing a parallel shape kept in sync by hand.
    return flow.mermaid


def compute_fingerprints(repo_ir: RepoIR, index: SymbolIndex, graph: DependencyGraph) -> dict[str, str]:
    all_classes = sorted(index.by_qualified_name.values(), key=lambda e: e.class_info.qualified_name)
    public_classes = [e for e in all_classes if "public" in e.class_info.modifiers]
    sequence_flows = build_sequence_flows(repo_ir, index)

    package_names = sorted(index.by_package.keys())
    breakdown = layer_breakdown([e.class_info for e in all_classes])
    architecture_payload = {
        "packages": package_names,
        "package_edges": sorted((s, t) for s, targets in graph.package_graph(index).items() for t in targets),
        "layer_breakdown": {layer: sorted(names) for layer, names in breakdown.items()},
        "classes": [_class_shape(e, graph) for e in all_classes],
        # Component graph and "primary request flow" both render directly
        # into ARCHITECTURE.md's prompt; external_systems affects the
        # component graph's external-system nodes and isn't otherwise
        # captured above (it comes from imports, not the type-dependency
        # graph), and the primary flow depends on the *call* graph, which
        # is a different graph than the type-dependency edges already
        # hashed in `classes` above.
        "external_systems": classify_external_systems(repo_ir),
        "primary_flow": _flow_shape(sequence_flows[0]) if sequence_flows else None,
    }
    readme_payload = {
        "packages": package_names,
        "class_count": len(all_classes),
        "public_class_names": sorted(e.class_info.qualified_name for e in public_classes),
        "external_packages": external_package_usage(repo_ir),
    }

    return {
        "README.md": _hash(readme_payload),
        "ARCHITECTURE.md": _hash(architecture_payload),
        "SEQUENCE.md": _hash([_flow_shape(f) for f in sequence_flows]),
    }
