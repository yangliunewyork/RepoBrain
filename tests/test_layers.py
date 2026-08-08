from datetime import datetime, timezone

from repobrain.analysis.dependency_analyzer import build_dependency_graph
from repobrain.analysis.layers import classify_layer, component_graph, layer_breakdown, layer_rank
from repobrain.analysis.symbol_extractor import build_symbol_index
from repobrain.ir.models import RepoIR
from repobrain.parsing.java_parser import JavaParser


def _repo_ir(sources: dict[str, str]) -> RepoIR:
    parser = JavaParser()
    files = {path: parser.parse(path, src.encode("utf-8")) for path, src in sources.items()}
    return RepoIR(repo_root="/repo", generated_at=datetime.now(timezone.utc).isoformat(), files=files)


def _class_info(source: str):
    ir = JavaParser().parse("A.java", source.encode("utf-8"))
    return ir.classes[0]


def test_rest_controller_annotation_classifies_as_api():
    info = _class_info("@RestController public class WidgetApi {}")
    assert classify_layer(info) == "api"


def test_service_annotation_classifies_as_service():
    info = _class_info("@Service public class WidgetBrain {}")
    assert classify_layer(info) == "service"


def test_repository_annotation_classifies_as_data():
    info = _class_info("@Repository public class WidgetVault {}")
    assert classify_layer(info) == "data"


def test_entity_annotation_classifies_as_domain():
    """@Entity marks a business noun (a data structure), not a runtime
    data-access component -- distinct from @Repository, which is "data"."""
    info = _class_info("@Entity public class WidgetRow { private String name; }")
    assert classify_layer(info) == "domain"


def test_table_document_embeddable_annotations_classify_as_domain():
    assert classify_layer(_class_info("@Table public class A { private int x; }")) == "domain"
    assert classify_layer(_class_info("@Document public class A { private int x; }")) == "domain"
    assert classify_layer(_class_info("@Embeddable public class A { private int x; }")) == "domain"


def test_plain_class_with_instance_field_classifies_as_domain():
    info = _class_info("public class Money { private long cents; }")
    assert classify_layer(info) == "domain"


def test_record_classifies_as_domain():
    info = _class_info("public record Point(int x, int y) {}")
    assert classify_layer(info) == "domain"


def test_enum_classifies_as_domain_even_without_fields():
    info = _class_info("public enum Status { ACTIVE, INACTIVE }")
    assert classify_layer(info) == "domain"


def test_static_only_class_is_not_domain():
    """A class made entirely of static constants/helpers isn't a domain
    object -- it has no instance state."""
    info = _class_info("public class Constants { public static final int MAX = 10; }")
    assert classify_layer(info) is None


def test_class_with_no_fields_or_annotations_is_unclassified():
    info = _class_info("public class Utils { public static int add(int a, int b) { return a + b; } }")
    assert classify_layer(info) is None


def test_keyword_match_wins_over_domain_shape_heuristic():
    """A class named "...Repository" with an instance field should still
    classify as data via the keyword, not domain via the field-shape
    fallback -- keyword matching is checked first."""
    info = _class_info("public class WidgetRepository { private int cacheSize; }")
    assert classify_layer(info) == "data"


def test_method_level_mapping_annotation_classifies_class_as_api():
    info = _class_info("public class Handler { @GetMapping(\"/x\") public void get() {} }")
    assert classify_layer(info) == "api"


def test_annotation_wins_over_conflicting_name_keyword():
    """A class literally named "...Repository" but annotated @Service
    should be classified by the annotation, not the name."""
    info = _class_info("@Service public class WidgetRepository {}")
    assert classify_layer(info) == "service"


def test_keyword_fallback_when_no_annotation_present():
    info = _class_info("public class WidgetController {}")
    assert classify_layer(info) == "api"

    info = _class_info("public class WidgetRepository {}")
    assert classify_layer(info) == "data"


def test_unclassified_when_no_annotation_or_keyword_matches():
    info = _class_info("public class Widget {}")
    assert classify_layer(info) is None


def test_layer_rank_orders_api_service_domain_data():
    assert layer_rank("api") < layer_rank("service") < layer_rank("domain") < layer_rank("data")


def test_layer_rank_treats_unclassified_as_middle_tier():
    assert layer_rank("api") < layer_rank(None) < layer_rank("data")


def test_layer_breakdown_groups_by_classification():
    infos = [
        _class_info("@RestController public class A {}"),
        _class_info("@Service public class B {}"),
        _class_info("@Repository public class C {}"),
        _class_info("@Entity public class E { private int x; }"),
        _class_info("public class D {}"),
    ]
    breakdown = layer_breakdown(infos)
    assert breakdown["api"] == ["A"]
    assert breakdown["service"] == ["B"]
    assert breakdown["data"] == ["C"]
    assert breakdown["domain"] == ["E"]
    assert breakdown["unclassified"] == ["D"]


def test_component_graph_aggregates_by_layer_not_package():
    repo_ir = _repo_ir(
        {
            "a/WidgetController.java": (
                "package a; @RestController public class WidgetController { private WidgetBrain b; "
                "public void get() { b.find(); } }"
            ),
            "b/WidgetBrain.java": (
                "package b; @Service public class WidgetBrain { private WidgetVault v; "
                "public Widget find() { return v.load(); } }"
            ),
            "c/WidgetVault.java": (
                "package c; @Repository public class WidgetVault { public Widget load() { return new Widget(); } } "
            ),
            "d/Widget.java": "package d; @Entity public class Widget { private String name; }",
        }
    )
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    edges = component_graph(repo_ir, index, graph)

    assert edges["API"] == {"Service"}
    assert edges["Service"] == {"Data", "Domain"}
    assert edges["Data"] == {"Domain"}
    # Four distinct packages collapsed into four component nodes, not
    # four package nodes with the same shape -- the point of the test.
    assert set(edges.keys()) <= {"API", "Service", "Domain", "Data"}


def test_component_graph_adds_external_system_nodes():
    repo_ir = _repo_ir(
        {
            "WidgetVault.java": (
                "import javax.persistence.EntityManager;\n"
                "@Repository public class WidgetVault { public void save() {} }"
            ),
        }
    )
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    external = {"Relational Database": ["javax.persistence"]}
    edges = component_graph(repo_ir, index, graph, external)

    assert edges["Data"] == {"Relational Database"}


def test_component_graph_has_no_self_loops():
    repo_ir = _repo_ir(
        {
            "A.java": "@Service public class A { @Service public class Unused {} public void run() {} }",
            "B.java": "@Service public class B { private A a; public void run() { a.run(); } }",
        }
    )
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    edges = component_graph(repo_ir, index, graph)
    # Both A and B are "Service" -- same bucket, so no edge should be recorded.
    assert edges == {}
