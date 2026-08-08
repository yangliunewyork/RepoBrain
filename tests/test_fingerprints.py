from datetime import datetime, timezone
from pathlib import Path

from repobrain.analysis.dependency_analyzer import build_dependency_graph
from repobrain.analysis.symbol_extractor import build_symbol_index
from repobrain.docgen.fingerprints import compute_fingerprints
from repobrain.ir.models import RepoIR
from repobrain.parsing.java_parser import JavaParser

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_java_repo"


def _repo_ir_from_sources(sources: dict[str, str]) -> RepoIR:
    parser = JavaParser()
    files = {path: parser.parse(path, src.encode("utf-8")) for path, src in sources.items()}
    return RepoIR(repo_root="/repo", generated_at=datetime.now(timezone.utc).isoformat(), files=files)


def _fingerprints(sources: dict[str, str]) -> dict[str, str]:
    repo_ir = _repo_ir_from_sources(sources)
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    return compute_fingerprints(repo_ir, index, graph)


def test_identical_sources_produce_identical_fingerprints():
    src = {"A.java": "public class A { public void run() {} }"}
    assert _fingerprints(src) == _fingerprints(dict(src))


def test_comment_only_change_does_not_affect_any_fingerprint():
    before = {"A.java": "public class A { public void run() { int x = 1; } }"}
    after = {"A.java": "public class A { public void run() { /* recomputed */ int x = 1; } }"}
    assert _fingerprints(before) == _fingerprints(after)


def test_new_resolvable_call_changes_sequence_and_architecture_but_not_readme():
    before = {
        "A.java": "public class A { private B b; public void run() {} }",
        "B.java": "public class B { public void step() {} }",
    }
    after = {
        "A.java": "public class A { private B b; public void run() { b.step(); } }",
        "B.java": "public class B { public void step() {} }",
    }

    fp_before, fp_after = _fingerprints(before), _fingerprints(after)
    assert fp_before["SEQUENCE.md"] != fp_after["SEQUENCE.md"]
    # ARCHITECTURE.md's prompt now includes the primary request flow
    # (derived from the same call graph SEQUENCE.md uses), so a new
    # resolvable call invalidates it too, even with no new type-level
    # dependency edge.
    assert fp_before["ARCHITECTURE.md"] != fp_after["ARCHITECTURE.md"]
    assert fp_before["README.md"] == fp_after["README.md"]


def test_new_public_method_with_no_calls_does_not_change_sequence_fingerprint():
    before = {"A.java": "public class A { public void run() {} }"}
    after = {"A.java": "public class A { public void run() {} public void stop() {} }"}

    fp_before, fp_after = _fingerprints(before), _fingerprints(after)
    assert fp_before["SEQUENCE.md"] == fp_after["SEQUENCE.md"]  # neither method calls anything resolvable
    # ARCHITECTURE.md's per-class card lists every method signature, so a
    # new method does invalidate it even with no new calls/dependencies.
    assert fp_before["ARCHITECTURE.md"] != fp_after["ARCHITECTURE.md"]


def test_new_cross_package_dependency_changes_architecture_fingerprint():
    before = {
        "a/A.java": "package a; public class A {}",
        "b/B.java": "package b; public class B {}",
    }
    after = {
        "a/A.java": "package a; public class A {}",
        "b/B.java": "package b; import a.A; public class B { A field; }",
    }
    fp_before, fp_after = _fingerprints(before), _fingerprints(after)
    assert fp_before["ARCHITECTURE.md"] != fp_after["ARCHITECTURE.md"]
    assert fp_before["README.md"] != fp_after["README.md"]  # class_count / shape changed too


def test_new_class_changes_readme_fingerprint():
    before = {"a/A.java": "package a; public class A {}"}
    after = {
        "a/A.java": "package a; public class A {}",
        "a/B.java": "package a; public class B {}",
    }
    assert _fingerprints(before)["README.md"] != _fingerprints(after)["README.md"]


def test_adding_a_layer_annotation_changes_architecture_fingerprint():
    """Adding @Service doesn't touch package structure or dependency
    edges, but it does change the verified layer breakdown that
    ARCHITECTURE.md's prompt is built from -- the fingerprint must catch
    this or a doc regeneration would be wrongly skipped."""
    before = {"a/Widget.java": "package a; public class Widget {}"}
    after = {"a/Widget.java": "package a; @Service public class Widget {}"}

    fp_before, fp_after = _fingerprints(before), _fingerprints(after)
    assert fp_before["ARCHITECTURE.md"] != fp_after["ARCHITECTURE.md"]


def test_adding_test_annotation_to_a_method_changes_sequence_fingerprint():
    """Marking a method @Test removes it from entry-point candidacy,
    which changes which flows SEQUENCE.md selects -- the fingerprint
    must catch this even though the method's signature is unchanged."""
    before = {
        "A.java": "public class A { private B b; public void run() { b.step(); } }",
        "B.java": "public class B { public void step() {} }",
    }
    after = {
        "A.java": "public class A { private B b; @Test public void run() { b.step(); } }",
        "B.java": "public class B { public void step() {} }",
    }
    fp_before, fp_after = _fingerprints(before), _fingerprints(after)
    assert fp_before["SEQUENCE.md"] != fp_after["SEQUENCE.md"]


def test_new_external_system_import_changes_architecture_fingerprint():
    """A new @Repository importing a JDBC/JPA package doesn't change any
    type-level dependency edge, but it does add an external-system node
    to ARCHITECTURE.md's component graph -- the fingerprint must catch
    this since it isn't otherwise reflected in per-class shape/deps."""
    before = {"A.java": "@Repository public class A { public void save() {} }"}
    after = {
        "A.java": "import javax.persistence.EntityManager;\n@Repository public class A { public void save() {} }"
    }
    fp_before, fp_after = _fingerprints(before), _fingerprints(after)
    assert fp_before["ARCHITECTURE.md"] != fp_after["ARCHITECTURE.md"]
