"""RepoBrain command-line interface.

    repobrain generate <repo>   Full analysis + documentation generation.
    repobrain update <repo>     Incremental analysis via Git diff.
    repobrain scan <repo>       Parse-only dry run (no LLM calls); useful
                                 to sanity-check parsing before spending
                                 time on generation.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from repobrain import __version__
from repobrain.analysis import build_dependency_graph, build_symbol_index
from repobrain.config import RepoBrainConfig
from repobrain.ir.models import RepoIR
from repobrain.llm import get_provider
from repobrain.logging_setup import configure_logging, get_logger
from repobrain.parsing import extensions_for_languages
from repobrain.pipeline import Pipeline, parse_files
from repobrain.scanner import NotAGitRepositoryError, RepoScanner

logger = get_logger("cli")


def _build_config(config_path, language, model, host, output_dir, log_level) -> RepoBrainConfig:
    overrides: dict = {}
    if language:
        overrides["languages"] = list(language)
    if output_dir:
        overrides["output_dir"] = output_dir
    if log_level:
        overrides.setdefault("logging", {})["level"] = log_level
    if model or host:
        llm_overrides = {}
        if model:
            llm_overrides["model"] = model
        if host:
            llm_overrides["host"] = host
        overrides["llm"] = llm_overrides

    config = RepoBrainConfig.load(config_path=config_path, overrides=overrides)
    configure_logging(config.logging)
    return config


def _common_options(f):
    f = click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None, help="Path to a project-local YAML config overriding the defaults.")(f)
    f = click.option("--language", multiple=True, help="Restrict analysis to these languages (repeatable). Default: from config.")(f)
    f = click.option("--model", default=None, help="Ollama model name, e.g. qwen3:8b.")(f)
    f = click.option("--host", default=None, help="Ollama host URL. Default: http://localhost:11434.")(f)
    f = click.option("--output-dir", default=None, help="Where generated docs are written, relative to the repo root.")(f)
    f = click.option("--log-level", default=None, help="DEBUG, INFO, WARNING, or ERROR.")(f)
    return f


@click.group()
@click.version_option(__version__, prog_name="repobrain")
def main():
    """RepoBrain: local-first repository intelligence and documentation generation."""


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--force", is_flag=True, help="Regenerate every configured document even if unchanged.")
@_common_options
def generate(repo_path, force, config_path, language, model, host, output_dir, log_level):
    """Run a full analysis of REPO_PATH and generate all configured docs."""
    config = _build_config(config_path, language, model, host, output_dir, log_level)
    _run(repo_path, config, mode="full", force=force)


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--force", is_flag=True, help="Regenerate every configured document even if its fingerprint is unchanged.")
@_common_options
def update(repo_path, force, config_path, language, model, host, output_dir, log_level):
    """Incrementally re-analyze REPO_PATH using Git diff and regenerate
    only the documents whose underlying structure changed."""
    config = _build_config(config_path, language, model, host, output_dir, log_level)
    _run(repo_path, config, mode="update", force=force)


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@_common_options
def scan(repo_path, config_path, language, model, host, output_dir, log_level):
    """Parse REPO_PATH and print a structural summary. No LLM calls."""
    config = _build_config(config_path, language, model, host, output_dir, log_level)
    repo_root = Path(repo_path).resolve()

    try:
        scanner = RepoScanner(repo_root, config.exclude_patterns)
    except NotAGitRepositoryError as exc:
        raise click.ClickException(str(exc))

    all_files = scanner.scan()
    extensions = extensions_for_languages(config.languages)
    source_files = scanner.filter_by_extensions(all_files, extensions)

    file_irs = parse_files(source_files, config)
    repo_ir = RepoIR(repo_root=str(repo_root), generated_at=datetime.now(timezone.utc).isoformat(), files=file_irs)
    index = build_symbol_index(repo_ir)
    graph = build_dependency_graph(repo_ir, index)

    click.echo(f"Files scanned:      {len(all_files)}")
    click.echo(f"Source files parsed: {len(file_irs)}")
    click.echo(f"Packages:           {len(index.by_package)}")
    click.echo(f"Types discovered:   {len(index.by_qualified_name)}")
    error_files = [p for p, f in file_irs.items() if f.parse_errors]
    if error_files:
        click.secho(f"Files with parse errors: {len(error_files)}", fg="yellow")
        for p in error_files[:10]:
            click.echo(f"  - {p}: {file_irs[p].parse_errors}")

    click.echo()
    for package, entries in sorted(index.by_package.items()):
        click.echo(f"{package or '(default package)'} ({len(entries)} types)")
        for entry in entries:
            deps = len(graph.dependencies_of(entry.class_info.qualified_name))
            click.echo(f"  - {entry.class_info.kind} {entry.class_info.name} ({len(entry.class_info.methods)} methods, {deps} internal deps)")


def _run(repo_path: str, config: RepoBrainConfig, mode: str, force: bool) -> None:
    repo_root = Path(repo_path).resolve()

    try:
        scanner = RepoScanner(repo_root, config.exclude_patterns)
    except NotAGitRepositoryError as exc:
        raise click.ClickException(str(exc))
    del scanner  # constructed only to validate the repo up front

    llm = get_provider(config.llm)
    if not llm.is_available():
        raise click.ClickException(
            f"Ollama model '{config.llm.model}' is not reachable/available at {config.llm.host}. "
            f"Start Ollama and run `ollama pull {config.llm.model}`, or pass --model/--host."
        )

    pipeline = Pipeline(repo_root, config, llm=llm)
    click.echo(f"Analyzing {repo_root} ({mode})...")

    summary = pipeline.run_full(force_docs=force) if mode == "full" else pipeline.run_update(force_docs=force)

    click.echo(f"Mode: {summary.mode}")
    click.echo(f"Files scanned: {summary.files_scanned}  parsed: {summary.files_parsed}  removed: {summary.files_removed}  reused: {summary.files_reused}")
    if summary.message:
        click.echo(summary.message)
    for result in summary.doc_results:
        color = "green" if result.written else "bright_black"
        status = "written" if result.written else "skipped"
        click.secho(f"  [{status}] {result.filename} ({result.reason})", fg=color)

    output_dir = repo_root / config.output_dir
    if any(r.written for r in summary.doc_results):
        click.echo(f"\nDocumentation written to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
