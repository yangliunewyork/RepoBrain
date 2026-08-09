from datetime import datetime, timezone

from repobrain.analysis import build_dependency_graph, build_symbol_index
from repobrain.docgen.context import build_project_context, char_budget_for_num_ctx
from repobrain.docgen.prompts import _fit_items_to_budget, build_architecture_prompt
from repobrain.ir.models import RepoIR
from repobrain.parsing.java_parser import JavaParser


def _repo_ir(sources: dict[str, str]) -> RepoIR:
    parser = JavaParser()
    files = {path: parser.parse(path, src.encode("utf-8")) for path, src in sources.items()}
    return RepoIR(repo_root="/repo", generated_at=datetime.now(timezone.utc).isoformat(), files=files)


def _context(sources: dict[str, str], max_detail_chars: int = 12_000):
    repo_ir = _repo_ir(sources)
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)
    return build_project_context("repo", repo_ir, index, graph, ["java"], max_detail_chars=max_detail_chars)


def test_architecture_prompt_stays_within_budget_for_domain_heavy_repo():
    """Regression test: a repo with many JPA-style entities (each with
    many fields) used to blow the prompt size out to ~2.5x the intended
    budget regardless of num_ctx, because the domain-model fact block
    wasn't capped and duplicated the same classes' full detail a second
    time in the per-layer class-card dump. This is the exact shape that
    caused a real production timeout against Ollama."""
    sources = {}
    for i in range(80):
        fields = "\n".join(f"private String field{j};" for j in range(20))
        sources[f"Entity{i}.java"] = f"@Entity public class Entity{i} {{ {fields} }}"
    sources["Controller.java"] = "@RestController public class Controller { private Service s; public void get() { s.run(); } }"
    sources["Service.java"] = "@Service public class Service { private Repo r; public void run() { r.save(); } }"
    sources["Repo.java"] = "@Repository public class Repo { public void save() {} }"

    budget = char_budget_for_num_ctx(8192)
    ctx = _context(sources, max_detail_chars=budget)
    prompt = build_architecture_prompt(ctx)

    # A generous multiple of the class-card budget, to allow room for the
    # layer breakdown / external systems / primary flow / component graph
    # sections without re-introducing an effectively unbounded prompt.
    assert len(prompt) < budget * 2


def test_domain_classes_are_not_duplicated_in_architecture_prompt():
    sources = {
        "Widget.java": "@Entity public class Widget { private String uniqueMarkerField; }",
        "Controller.java": "@RestController public class Controller { public void get() {} }",
    }
    ctx = _context(sources)
    prompt = build_architecture_prompt(ctx)
    assert prompt.count("uniqueMarkerField") == 1


def test_domain_model_section_caps_classes_and_fields_shown():
    sources = {}
    for i in range(20):
        fields = "\n".join(f"private String field{j};" for j in range(10))
        sources[f"Entity{i}.java"] = f"@Entity public class Entity{i} {{ {fields} }}"
    ctx = _context(sources)
    prompt = build_architecture_prompt(ctx)

    assert "more domain types not shown in detail" in prompt
    assert "more fields" in prompt


def test_layer_breakdown_name_list_is_capped():
    sources = {f"Service{i}.java": f"@Service public class Service{i} {{ public void run() {{}} }}" for i in range(50)}
    ctx = _context(sources)
    prompt = build_architecture_prompt(ctx)
    assert ", and " in prompt and " more" in prompt


def test_fit_items_to_budget_force_first_only_applies_when_requested():
    """Direct test of the exact mechanism behind the compounding bug:
    calling `_fit_items_to_budget` with `force_first=True` against an
    already-exhausted budget (simulating a second/third layer group
    after earlier groups used up the shared budget) must NOT add
    anything, whereas `force_first=True` against a *fresh* budget must
    still guarantee at least one item, even an oversized one."""
    big_item = "x" * 1000

    # Fresh budget, force_first=True: the oversized item is still added
    # (this is what guarantees a section is never left completely empty).
    text, used, kept = _fit_items_to_budget([big_item], budget=10, force_first=True)
    assert kept == 1
    assert text == big_item

    # Exhausted/negative budget, force_first=False: nothing gets added --
    # this is the case that must hold for every group after the first
    # one sharing a single running budget, or their "at least one"
    # guarantees compound past the intended total.
    text, used, kept = _fit_items_to_budget([big_item], budget=-500, force_first=False)
    assert kept == 0
    assert text == ""
    assert used == 0


def test_component_classes_budget_is_not_compounded_across_layer_groups():
    """Regression test: `_render_component_classes` used to guarantee
    "at least one card" independently for *each* of API/Service/Data/
    Unclassified, so on a real repo with substantial classes in every
    layer, each group force-included its own oversized first card
    regardless of how little budget earlier groups had already used --
    compounding to ~2x the intended total on a real 94-file repository.
    A single class here is deliberately large (many fields) relative to
    a tight budget, and appears in three different layers, to reproduce
    that compounding shape directly.
    """
    def big_class(name: str, annotation: str) -> str:
        fields = "\n".join(f"private String field{j};" for j in range(15))
        return f"{annotation} public class {name} {{ {fields} }}"

    sources = {
        "Api1.java": big_class("Api1", "@RestController"),
        "Api2.java": big_class("Api2", "@RestController"),
        "Svc1.java": big_class("Svc1", "@Service"),
        "Svc2.java": big_class("Svc2", "@Service"),
        "Data1.java": big_class("Data1", "@Repository"),
        "Data2.java": big_class("Data2", "@Repository"),
    }
    tight_budget = 1200  # smaller than a single one of these classes' rendered card
    ctx = _context(sources, max_detail_chars=tight_budget)
    prompt = build_architecture_prompt(ctx)

    # Extract just the "Supporting per-class detail" section (it runs up
    # to the "---" separator before the instructions) to measure its
    # size in isolation from the rest of the prompt.
    section = prompt.split("Supporting per-class detail")[1].split("\n---")[0]
    # With the fix, at most one card total gets force-included past the
    # tight budget; without it, up to three groups (API/Service/Data)
    # would each force one in, multiplying the overshoot.
    assert len(section) < tight_budget * 2
