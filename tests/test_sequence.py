from datetime import datetime, timezone

from repobrain.analysis.symbol_extractor import build_symbol_index
from repobrain.docgen.sequence import MAX_DEPTH, MAX_SEQUENCE_FLOWS, MAX_STEPS_PER_FLOW, build_sequence_flows
from repobrain.ir.models import RepoIR
from repobrain.parsing.java_parser import JavaParser


def _repo_ir(sources: dict[str, str]) -> RepoIR:
    parser = JavaParser()
    files = {path: parser.parse(path, src.encode("utf-8")) for path, src in sources.items()}
    return RepoIR(repo_root="/repo", generated_at=datetime.now(timezone.utc).isoformat(), files=files)


def _flows(sources):
    repo_ir = _repo_ir(sources)
    index = build_symbol_index(repo_ir)
    return build_sequence_flows(repo_ir, index)


def test_simple_flow_is_traced_and_rendered():
    flows = _flows(
        {
            "A.java": "public class A { private B b; public void run() { b.step(); } }",
            "B.java": "public class B { public void step() {} }",
        }
    )
    assert len(flows) == 1
    flow = flows[0]
    assert (flow.entry_class, flow.entry_method) == ("A", "run")
    assert len(flow.steps) == 1
    assert flow.steps[0].callee_method == "step"
    assert flow.mermaid.startswith("sequenceDiagram")
    assert "A->>B: step()" in flow.mermaid


def test_method_with_no_outgoing_calls_is_not_a_flow():
    flows = _flows({"A.java": "public class A { public void run() {} }"})
    assert flows == []


def test_private_or_non_public_methods_are_not_entry_points():
    flows = _flows(
        {
            "A.java": "public class A { private B b; void run() { b.step(); } }",
            "B.java": "public class B { public void step() {} }",
        }
    )
    assert flows == []  # `run` is package-private, not a candidate entry point


def test_method_called_by_another_project_method_is_deprioritized_when_over_cap():
    """When candidates exceed MAX_SEQUENCE_FLOWS, methods nothing else
    calls (likely real entry points) are kept over ones that are
    themselves called by other project code — here `inner`."""
    sources = {
        "A.java": (
            "public class A { private B b; "
            "public void outer() { inner(); } "
            "public void inner() { b.step(); } }"
        ),
        "B.java": "public class B { public void step() {} }",
    }
    for i in range(MAX_SEQUENCE_FLOWS):
        sources[f"P{i}.java"] = f"public class P{i} {{ private B b; public void run{i}() {{ b.step(); }} }}"

    entries = {(f.entry_class, f.entry_method) for f in _flows(sources)}
    assert ("A", "inner") not in entries  # dropped: called by A.outer, and candidates exceed the cap
    assert ("A", "outer") in entries


def test_call_chain_traced_transitively_across_classes():
    flows = _flows(
        {
            "A.java": "public class A { private B b; public void run() { b.step(); } }",
            # step() is package-private, so it isn't itself an entry
            # candidate — it should only appear as a traced step of A.run.
            "B.java": "public class B { private C c; void step() { c.finish(); } }",
            "C.java": "public class C { public void finish() {} }",
        }
    )
    assert len(flows) == 1
    steps = flows[0].steps
    assert len(steps) == 2
    assert steps[0].callee_class == "B" and steps[0].callee_method == "step"
    assert steps[1].callee_class == "C" and steps[1].callee_method == "finish"


def test_recursive_calls_do_not_infinite_loop():
    flows = _flows(
        {
            "A.java": "public class A { private B b; public void run() { b.step(); } }",
            "B.java": "public class B { private A a; public void step() { a.run(); } }",
        }
    )
    # Neither method is a "nothing else calls this" entry point (each
    # calls the other), so both remain candidates; the important thing is
    # that tracing either one terminates rather than looping forever.
    assert 1 <= len(flows) <= 2
    for flow in flows:
        assert len(flow.steps) <= MAX_STEPS_PER_FLOW
        assert len(flow.steps) <= MAX_DEPTH + 1


def _participant_order(mermaid: str) -> list[str]:
    return [
        line.split(" as ")[1].strip()
        for line in mermaid.splitlines()
        if line.strip().startswith("participant")
    ]


def test_participants_ordered_left_to_right_by_api_service_data_layer():
    flows = _flows(
        {
            "OrderController.java": (
                "public class OrderController { private OrderService s; "
                "public void run() { s.place(); } }"
            ),
            "OrderService.java": (
                "public class OrderService { private OrderRepository r; "
                "void place() { r.save(); } }"  # package-private: not its own entry candidate
            ),
            "OrderRepository.java": "public class OrderRepository { public void save() {} }",
        }
    )
    assert len(flows) == 1
    order = _participant_order(flows[0].mermaid)
    assert order == ["OrderController", "OrderService", "OrderRepository"]


def test_layer_order_wins_over_raw_call_order():
    """OrderService calls its repository *before* it calls a second
    service, so naive first-seen order would put the repository in the
    middle — the layer rule should still push it to the right."""
    flows = _flows(
        {
            "OrderService.java": (
                "public class OrderService { private OrderRepository r; private PricingService p; "
                "public void run() { r.save(); p.compute(); } }"
            ),
            "OrderRepository.java": "public class OrderRepository { public void save() {} }",
            "PricingService.java": "public class PricingService { public void compute() {} }",
        }
    )
    assert len(flows) == 1
    order = _participant_order(flows[0].mermaid)
    assert order.index("OrderService") < order.index("PricingService") < order.index("OrderRepository")


def test_flow_count_is_capped():
    sources = {"B.java": "public class B { public void step() {} }"}
    for i in range(MAX_SEQUENCE_FLOWS + 5):
        sources[f"A{i}.java"] = f"public class A{i} {{ private B b; public void run{i}() {{ b.step(); }} }}"
    flows = _flows(sources)
    assert len(flows) == MAX_SEQUENCE_FLOWS


def test_empty_mermaid_for_entry_with_no_steps_is_not_produced():
    # sanity: build_sequence_flows never emits a flow with zero steps,
    # since entry selection requires at least one outgoing call
    flows = _flows({"A.java": "public class A { public void run() {} }"})
    assert all(f.steps for f in flows)
