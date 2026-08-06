"""Ties context building, fingerprinting, and generation together into
one call: given the analyzed repository, produce (and know whether to
skip) each configured document.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repobrain.analysis.dependency_analyzer import DependencyGraph
from repobrain.analysis.symbol_extractor import SymbolIndex
from repobrain.config import RepoBrainConfig
from repobrain.docgen.context import build_project_context, char_budget_for_num_ctx
from repobrain.docgen.fingerprints import compute_fingerprints
from repobrain.docgen.generators import ALL_GENERATORS, DocGenerator
from repobrain.ir.models import RepoIR
from repobrain.llm.base import LLMProvider
from repobrain.logging_setup import get_logger

logger = get_logger("docgen.pipeline")


@dataclass
class DocResult:
    filename: str
    written: bool
    reason: str


def _generators_for(config: RepoBrainConfig) -> list[DocGenerator]:
    wanted = set(config.docs)
    return [g for g in ALL_GENERATORS if g.filename in wanted]


def generate_documentation(
    repo_root: Path,
    repo_name: str,
    repo_ir: RepoIR,
    index: SymbolIndex,
    graph: DependencyGraph,
    config: RepoBrainConfig,
    llm: LLMProvider,
    previous_fingerprints: dict[str, str] | None = None,
    force: bool = False,
) -> tuple[dict[str, str], list[DocResult]]:
    """Generate configured docs, skipping any whose structural fingerprint
    is unchanged from `previous_fingerprints` (unless `force`).

    Returns (new_fingerprints, results). Files are written under
    `config.output_dir` relative to `repo_root`.
    """
    fingerprints = compute_fingerprints(repo_ir, index, graph)
    max_detail_chars = char_budget_for_num_ctx(config.llm.num_ctx)
    ctx = build_project_context(repo_name, repo_ir, index, graph, config.languages, max_detail_chars=max_detail_chars)
    if ctx.truncated:
        logger.warning(
            "Repository has more class detail than fits in num_ctx=%d; some classes are "
            "summarized by package count only. Increase llm.num_ctx to include more detail.",
            config.llm.num_ctx,
        )

    output_dir = repo_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[DocResult] = []
    for generator in _generators_for(config):
        old_fp = (previous_fingerprints or {}).get(generator.filename)
        new_fp = fingerprints[generator.filename]

        if not force and old_fp is not None and old_fp == new_fp:
            results.append(DocResult(generator.filename, written=False, reason="unchanged"))
            logger.info("Skipping %s: no structural changes detected", generator.filename)
            continue

        content = generator.generate(ctx, llm)
        (output_dir / generator.filename).write_text(content, encoding="utf-8")
        reason = "forced" if force else ("initial" if old_fp is None else "changed")
        results.append(DocResult(generator.filename, written=True, reason=reason))
        logger.info("Wrote %s (%s)", generator.filename, reason)

    return fingerprints, results
