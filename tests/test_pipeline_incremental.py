from repobrain.config import RepoBrainConfig
from repobrain.pipeline import Pipeline

from tests.conftest import git_commit_all


def _pipeline(repo_path, fake_llm) -> Pipeline:
    config = RepoBrainConfig.load(overrides={"output_dir": "docs/generated"})
    return Pipeline(repo_path, config, llm=fake_llm)


def test_run_full_generates_every_configured_doc(git_repo, fake_llm):
    pipeline = _pipeline(git_repo, fake_llm)
    summary = pipeline.run_full()

    assert summary.mode == "full"
    assert summary.files_parsed == 4
    written = {r.filename for r in summary.doc_results if r.written}
    assert written == {"README.md", "ARCHITECTURE.md", "SEQUENCE.md"}
    for name in written:
        assert (git_repo / "docs/generated" / name).is_file()
    assert (git_repo / ".repobrain" / "state.json").is_file()


def test_update_with_no_changes_is_a_no_op(git_repo, fake_llm):
    pipeline = _pipeline(git_repo, fake_llm)
    pipeline.run_full()
    calls_after_full = len(fake_llm.calls)

    summary = pipeline.run_update()
    assert summary.message == "No file changes detected since last run; nothing to do."
    assert summary.doc_results == []
    assert len(fake_llm.calls) == calls_after_full  # no new LLM calls


def test_update_skips_docs_when_only_method_body_changes(git_repo, fake_llm):
    pipeline = _pipeline(git_repo, fake_llm)
    pipeline.run_full()

    target = git_repo / "src/main/java/com/example/model/Widget.java"
    target.write_text(target.read_text().replace("return name;", "return name; // noop"))

    summary = pipeline.run_update()
    assert summary.files_parsed >= 1
    assert all(not r.written for r in summary.doc_results)
    assert all(r.reason == "unchanged" for r in summary.doc_results)


def test_update_regenerates_docs_when_a_new_class_is_added(git_repo, fake_llm):
    pipeline = _pipeline(git_repo, fake_llm)
    pipeline.run_full()

    new_file = git_repo / "src/main/java/com/example/service/PricingService.java"
    new_file.write_text(
        "package com.example.service;\n"
        "import com.example.model.Widget;\n"
        "public class PricingService {\n"
        "    public double discount(Widget w) { return w.getPrice(); }\n"
        "}\n"
    )

    summary = pipeline.run_update()
    written = {r.filename for r in summary.doc_results if r.written}
    assert "README.md" in written  # a new public class was added
    assert "SEQUENCE.md" in written  # discount() -> Widget.getPrice() is a new resolvable call
    # ARCHITECTURE.md's prompt lists every class as a card, so a brand-new
    # class regenerates it too even without a new cross-package edge.
    assert "ARCHITECTURE.md" in written


def test_update_handles_deleted_file(git_repo, fake_llm):
    pipeline = _pipeline(git_repo, fake_llm)
    pipeline.run_full()

    (git_repo / "src/main/java/com/example/service/WidgetService.java").unlink()
    git_commit_all(git_repo, "remove WidgetService")

    summary = pipeline.run_update()
    assert summary.files_removed == 1
    written = {r.filename for r in summary.doc_results if r.written}
    # WidgetService was the only class in com.example.service, and the
    # only source of its call flows — removing it changes package
    # structure, dependency edges, and call flows all at once.
    assert written == {"README.md", "ARCHITECTURE.md", "SEQUENCE.md"}


def test_update_falls_back_to_full_without_prior_state(git_repo, fake_llm):
    pipeline = _pipeline(git_repo, fake_llm)
    summary = pipeline.run_update()
    assert "full" in summary.mode
    assert summary.files_parsed == 4


def test_force_regenerates_unchanged_docs(git_repo, fake_llm):
    pipeline = _pipeline(git_repo, fake_llm)
    pipeline.run_full()

    summary = pipeline.run_update(force_docs=True)
    written = {r.filename for r in summary.doc_results if r.written}
    assert written == {"README.md", "ARCHITECTURE.md", "SEQUENCE.md"}
    assert all(r.reason == "forced" for r in summary.doc_results)
