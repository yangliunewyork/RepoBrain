from datetime import datetime, timezone

from repobrain.analysis.call_graph import build_call_graph, group_by_caller
from repobrain.analysis.symbol_extractor import build_symbol_index
from repobrain.ir.models import RepoIR
from repobrain.parsing.java_parser import JavaParser


def _repo_ir(sources: dict[str, str]) -> RepoIR:
    parser = JavaParser()
    files = {path: parser.parse(path, src.encode("utf-8")) for path, src in sources.items()}
    return RepoIR(repo_root="/repo", generated_at=datetime.now(timezone.utc).isoformat(), files=files)


def _edges(sources):
    repo_ir = _repo_ir(sources)
    index = build_symbol_index(repo_ir)
    return build_call_graph(repo_ir, index)


def test_resolves_call_on_a_field():
    edges = _edges(
        {
            "A.java": "public class A { private B b; public void run() { b.step(); } }",
            "B.java": "public class B { public void step() {} }",
        }
    )
    assert len(edges) == 1
    edge = edges[0]
    assert (edge.caller_class, edge.caller_method) == ("A", "run")
    assert (edge.callee_class, edge.callee_method) == ("B", "step")


def test_resolves_call_on_a_parameter():
    edges = _edges(
        {
            "A.java": "public class A { public void run(B b) { b.step(); } }",
            "B.java": "public class B { public void step() {} }",
        }
    )
    assert len(edges) == 1
    assert edges[0].callee_class == "B"


def test_implicit_self_call_resolves_to_own_class():
    edges = _edges({"A.java": "public class A { public void run() { helper(); } private void helper() {} }"})
    assert len(edges) == 1
    assert edges[0].caller_class == edges[0].callee_class == "A"
    assert edges[0].callee_method == "helper"


def test_this_call_resolves_to_own_class():
    edges = _edges({"A.java": "public class A { public void run() { this.helper(); } private void helper() {} }"})
    assert len(edges) == 1
    assert edges[0].callee_class == "A"


def test_call_on_unresolvable_local_variable_is_dropped():
    edges = _edges(
        {
            "A.java": "public class A { public void run() { B b = make(); b.step(); } private B make() { return null; } }",
            "B.java": "public class B { public void step() {} }",
        }
    )
    # `make()` resolves (implicit self-call); `b.step()` doesn't, since `b`
    # is a local variable, not a field or parameter, and isn't tracked.
    assert all(e.callee_method != "step" for e in edges)


def test_call_on_external_type_is_dropped():
    edges = _edges({"A.java": "public class A { public void run() { java.util.Collections.emptyList(); } }"})
    assert edges == []


def test_static_call_on_project_class_resolves():
    edges = _edges(
        {
            "A.java": "public class A { public void run() { B.create(); } }",
            "B.java": "public class B { public static void create() {} }",
        }
    )
    assert len(edges) == 1
    assert edges[0].callee_class == "B"
    assert edges[0].callee_method == "create"


def test_group_by_caller():
    edges = _edges(
        {
            "A.java": "public class A { private B b; public void run() { b.one(); b.two(); } }",
            "B.java": "public class B { public void one() {} public void two() {} }",
        }
    )
    grouped = group_by_caller(edges)
    assert list(grouped.keys()) == [("A", "run")]
    assert [e.callee_method for e in grouped[("A", "run")]] == ["one", "two"]
