from datetime import datetime, timezone

from repobrain.analysis.dependency_analyzer import build_dependency_graph
from repobrain.analysis.symbol_extractor import build_symbol_index
from repobrain.docgen.context import build_project_context, char_budget_for_num_ctx
from repobrain.ir.models import RepoIR
from repobrain.parsing.java_parser import JavaParser


def _repo_ir(num_classes: int, methods_per_class: int = 5) -> RepoIR:
    parser = JavaParser()
    files = {}
    for i in range(num_classes):
        methods = "\n".join(f"public void method{m}() {{}}" for m in range(methods_per_class))
        src = f"package p; public class Class{i} {{ {methods} }}"
        path = f"Class{i}.java"
        files[path] = parser.parse(path, src.encode("utf-8"))
    return RepoIR(repo_root="/repo", generated_at=datetime.now(timezone.utc).isoformat(), files=files)


def _context(num_classes, methods_per_class=5, max_detail_chars=12_000):
    repo_ir = _repo_ir(num_classes, methods_per_class)
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    return build_project_context("repo", repo_ir, index, graph, ["java"], max_detail_chars=max_detail_chars)


def test_small_repo_is_not_truncated():
    ctx = _context(num_classes=5)
    assert not ctx.truncated
    assert len(ctx.all_class_cards()) == 5


def test_large_repo_is_truncated_to_fit_char_budget():
    ctx = _context(num_classes=200, methods_per_class=20, max_detail_chars=5_000)
    assert ctx.truncated
    assert ctx.class_count == 200
    assert len(ctx.all_class_cards()) < 200
    total_rendered_chars = sum(len(c.render()) for c in ctx.all_class_cards())
    assert total_rendered_chars < 20_000  # bounded, not proportional to 200 classes


def test_every_package_gets_at_least_one_class_even_under_tight_budget():
    repo_ir_a = _repo_ir(1, methods_per_class=200)  # one huge class
    parser = JavaParser()
    src_b = "package q; public class Small {}"
    repo_ir_a.files["Small.java"] = parser.parse("Small.java", src_b.encode("utf-8"))

    index = build_symbol_index(repo_ir_a)
    graph = build_dependency_graph(repo_ir_a, index)
    ctx = build_project_context("repo", repo_ir_a, index, graph, ["java"], max_detail_chars=200)

    assert all(pkg.classes for pkg in ctx.package_summaries)


def test_methods_capped_on_a_single_oversized_class():
    ctx = _context(num_classes=1, methods_per_class=100, max_detail_chars=1_000_000)
    card = ctx.all_class_cards()[0]
    assert len(card.methods) <= 41  # capped list + one "more not shown" marker
    assert "more not shown" in card.methods[-1]


def test_char_budget_scales_with_num_ctx():
    small = char_budget_for_num_ctx(2048)
    large = char_budget_for_num_ctx(32768)
    assert large > small
    assert small > 0
