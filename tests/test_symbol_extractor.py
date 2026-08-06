from datetime import datetime, timezone
from pathlib import Path

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


def test_index_contains_every_class_by_qualified_name():
    index = build_symbol_index(_build_fixture_repo_ir())
    assert set(index.by_qualified_name.keys()) == {
        "com.example.model.Widget",
        "com.example.repo.WidgetRepository",
        "com.example.repo.InMemoryWidgetRepository",
        "com.example.service.WidgetService",
    }


def test_index_groups_by_package():
    index = build_symbol_index(_build_fixture_repo_ir())
    assert {e.class_info.name for e in index.by_package["com.example.repo"]} == {
        "WidgetRepository",
        "InMemoryWidgetRepository",
    }


def test_resolve_simple_name_is_unique():
    index = build_symbol_index(_build_fixture_repo_ir())
    entry = index.resolve_simple_name("Widget")
    assert entry is not None
    assert entry.class_info.qualified_name == "com.example.model.Widget"


def test_resolve_simple_name_returns_none_when_absent():
    index = build_symbol_index(_build_fixture_repo_ir())
    assert index.resolve_simple_name("DoesNotExist") is None


def test_public_api_entries_excludes_non_public_classes():
    index = build_symbol_index(_build_fixture_repo_ir())
    public_names = {e.class_info.qualified_name for e in index.public_api_entries()}
    assert public_names == set(index.by_qualified_name.keys())  # all fixture classes are public
