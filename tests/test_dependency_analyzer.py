from datetime import datetime, timezone
from pathlib import Path

from repobrain.analysis.dependency_analyzer import build_dependency_graph, external_package_usage, to_mermaid
from repobrain.analysis.symbol_extractor import build_symbol_index
from repobrain.ir.models import RepoIR
from repobrain.parsing.java_parser import JavaParser

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_java_repo"


def _build_fixture_repo_ir() -> RepoIR:
    parser = JavaParser()
    files = {}
    for f in FIXTURE_DIR.rglob("*.java"):
        rel = str(f.relative_to(FIXTURE_DIR))
        files[rel] = parser.parse(rel, f.read_bytes())
    return RepoIR(repo_root=str(FIXTURE_DIR), generated_at=datetime.now(timezone.utc).isoformat(), files=files)


def test_service_depends_on_repository_and_model():
    repo_ir = _build_fixture_repo_ir()
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)

    deps = graph.dependencies_of("com.example.service.WidgetService")
    assert "com.example.repo.WidgetRepository" in deps
    assert "com.example.model.Widget" in deps
    assert "field" in deps["com.example.repo.WidgetRepository"].kinds


def test_in_memory_repository_implements_interface():
    repo_ir = _build_fixture_repo_ir()
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)

    deps = graph.dependencies_of("com.example.repo.InMemoryWidgetRepository")
    assert "implements" in deps["com.example.repo.WidgetRepository"].kinds


def test_widget_model_has_no_internal_dependencies():
    repo_ir = _build_fixture_repo_ir()
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    assert graph.dependencies_of("com.example.model.Widget") == {}


def test_dependents_of_widget_include_repository_and_service():
    repo_ir = _build_fixture_repo_ir()
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    dependents = graph.dependents_of("com.example.model.Widget")
    assert "com.example.service.WidgetService" in dependents
    assert "com.example.repo.WidgetRepository" in dependents


def test_package_graph_aggregates_cross_package_edges():
    repo_ir = _build_fixture_repo_ir()
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    pkg_graph = graph.package_graph(index)
    assert "com.example.model" in pkg_graph["com.example.repo"]
    assert "com.example.model" in pkg_graph["com.example.service"]
    assert "com.example.repo" in pkg_graph["com.example.service"]


def test_external_package_usage_excludes_internal_packages():
    repo_ir = _build_fixture_repo_ir()
    usage = dict(external_package_usage(repo_ir))
    assert "java.util" in usage
    assert not any(pkg.startswith("com.example") for pkg in usage)


def test_to_mermaid_produces_valid_looking_graph():
    repo_ir = _build_fixture_repo_ir()
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    mermaid = to_mermaid(graph.package_graph(index))
    assert mermaid.startswith("graph TD")
    assert "-->" in mermaid


def test_to_mermaid_orientation_is_overridable():
    repo_ir = _build_fixture_repo_ir()
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    mermaid = to_mermaid(graph.package_graph(index), orientation="LR")
    assert mermaid.startswith("graph LR")
    assert "-->" in mermaid
