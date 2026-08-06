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
from repobrain.analysis.symbol_extractor import SymbolIndex
from repobrain.docgen.sequence import build_sequence_flows
from repobrain.ir.models import RepoIR


def _hash(obj) -> str:
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sequence_shape(repo_ir: RepoIR, index: SymbolIndex) -> list[dict]:
    flows = build_sequence_flows(repo_ir, index)
    return [
        {
            "entry": f"{flow.entry_class}.{flow.entry_method}",
            "steps": [
                f"{s.caller_class}.{s.caller_method}->{s.callee_class}.{s.callee_method}" for s in flow.steps
            ],
        }
        for flow in flows
    ]


def compute_fingerprints(repo_ir: RepoIR, index: SymbolIndex, graph: DependencyGraph) -> dict[str, str]:
    all_classes = sorted(index.by_qualified_name.values(), key=lambda e: e.class_info.qualified_name)
    public_classes = [e for e in all_classes if "public" in e.class_info.modifiers]

    package_names = sorted(index.by_package.keys())
    architecture_payload = {
        "packages": package_names,
        "package_edges": sorted((s, t) for s, targets in graph.package_graph(index).items() for t in targets),
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
        "SEQUENCE.md": _hash(_sequence_shape(repo_ir, index)),
    }
