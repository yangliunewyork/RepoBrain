"""Top-level orchestrator: scan -> parse -> analyze -> generate docs.

`run_full` always re-parses every file. `run_update` uses Git to find
what changed since the last run, re-parses only that subset, merges the
result into the cached repository IR, and regenerates only the
documents whose structural fingerprint actually moved (see
`docgen.fingerprints`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from repobrain.analysis import build_dependency_graph, build_symbol_index
from repobrain.cache import RepoBrainState, StateStore
from repobrain.config import RepoBrainConfig
from repobrain.docgen import DocResult, generate_documentation
from repobrain.ir.models import FileIR, RepoIR
from repobrain.llm import get_provider
from repobrain.llm.base import LLMProvider
from repobrain.logging_setup import get_logger
from repobrain.parsing import extensions_for_languages, parser_for_extension
from repobrain.scanner import RepoScanner, current_commit, diff_since

logger = get_logger("pipeline")


@dataclass
class RunSummary:
    mode: str  # "full" | "incremental"
    files_scanned: int = 0
    files_parsed: int = 0
    files_removed: int = 0
    files_reused: int = 0
    doc_results: list[DocResult] = field(default_factory=list)
    message: str | None = None


def parse_files(
    scanner_files, config: RepoBrainConfig, cached_files: dict[str, FileIR] | None = None
) -> dict[str, FileIR]:
    """Parse each file, reusing a cached FileIR when its content hash is
    unchanged. Git may flag a file as dirty relative to a stale base
    commit (e.g. edited again before the previous edit was committed);
    hashing the actual bytes catches that cheaply, without re-running
    tree-sitter on unchanged content."""
    cached_files = cached_files or {}
    parsed: dict[str, FileIR] = {}
    for f in scanner_files:
        parser = parser_for_extension(Path(f.rel_path).suffix, config.languages)
        if parser is None:
            continue
        content = f.abs_path.read_bytes()
        cached = cached_files.get(f.rel_path)
        if cached is not None and cached.content_hash == hashlib.sha256(content).hexdigest():
            parsed[f.rel_path] = cached
            continue
        parsed[f.rel_path] = parser.parse(f.rel_path, content)
    return parsed


class Pipeline:
    def __init__(self, repo_root: Path, config: RepoBrainConfig, llm: LLMProvider | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        self.llm = llm or get_provider(config.llm)
        self.state_store = StateStore(self.repo_root, config.state_dir)

    def _repo_name(self) -> str:
        return self.repo_root.name

    def run_full(self, force_docs: bool = False) -> RunSummary:
        scanner = RepoScanner(self.repo_root, self.config.exclude_patterns)
        all_files = scanner.scan()
        extensions = extensions_for_languages(self.config.languages)
        source_files = scanner.filter_by_extensions(all_files, extensions)

        file_irs = parse_files(source_files, self.config)
        repo_ir = RepoIR(
            repo_root=str(self.repo_root),
            generated_at=datetime.now(timezone.utc).isoformat(),
            files=file_irs,
        )

        summary = RunSummary(mode="full", files_scanned=len(source_files), files_parsed=len(file_irs))
        summary.doc_results = self._generate_and_save(repo_ir, previous_fingerprints=None, force_docs=force_docs)
        return summary

    def run_update(self, force_docs: bool = False) -> RunSummary:
        state = self.state_store.load()
        if state is None or state.repo_ir is None:
            logger.info("No previous state found; running a full analysis instead.")
            summary = self.run_full(force_docs=force_docs)
            summary.mode = "full (no prior state)"
            return summary

        scanner = RepoScanner(self.repo_root, self.config.exclude_patterns)
        all_files = scanner.scan()
        extensions = extensions_for_languages(self.config.languages)
        source_files = scanner.filter_by_extensions(all_files, extensions)
        current_paths = {f.rel_path for f in source_files}
        cached_paths = set(state.repo_ir.files.keys())

        changeset = diff_since(self.repo_root, state.last_commit)
        # git diff only sees tracked history and never reports untracked
        # files, so brand-new untracked files (present now, absent from the
        # cache) are added on top of whatever the commit-based diff found.
        changed_paths = (changeset.all_changed_paths & current_paths) | (current_paths - cached_paths)
        removed_paths = (changeset.all_removed_paths | cached_paths) - current_paths

        files_by_path = {f.rel_path: f for f in source_files}
        to_reparse = [files_by_path[p] for p in changed_paths if p in files_by_path]

        reparsed = parse_files(to_reparse, self.config, cached_files=state.repo_ir.files)

        merged_files = dict(state.repo_ir.files)
        for path in removed_paths:
            merged_files.pop(path, None)
        merged_files.update(reparsed)

        repo_ir = RepoIR(
            repo_root=str(self.repo_root),
            generated_at=datetime.now(timezone.utc).isoformat(),
            files=merged_files,
        )

        summary = RunSummary(
            mode="incremental",
            files_scanned=len(source_files),
            files_parsed=len(reparsed),
            files_removed=len(removed_paths),
            files_reused=len(merged_files) - len(reparsed),
        )

        old_hashes = {p: fir.content_hash for p, fir in state.repo_ir.files.items()}
        new_hashes = {p: fir.content_hash for p, fir in merged_files.items()}
        if old_hashes == new_hashes and not force_docs:
            summary.message = "No file changes detected since last run; nothing to do."
            summary.doc_results = []
            return summary

        summary.doc_results = self._generate_and_save(repo_ir, previous_fingerprints=state.doc_fingerprints, force_docs=force_docs)
        return summary

    def _generate_and_save(self, repo_ir: RepoIR, previous_fingerprints: dict[str, str] | None, force_docs: bool) -> list[DocResult]:
        index = build_symbol_index(repo_ir)
        graph = build_dependency_graph(repo_ir, index)

        fingerprints, results = generate_documentation(
            repo_root=self.repo_root,
            repo_name=self._repo_name(),
            repo_ir=repo_ir,
            index=index,
            graph=graph,
            config=self.config,
            llm=self.llm,
            previous_fingerprints=previous_fingerprints,
            force=force_docs,
        )

        new_state = RepoBrainState(
            last_commit=current_commit(self.repo_root),
            repo_ir=repo_ir,
            doc_fingerprints=fingerprints,
        )
        self.state_store.save(new_state)
        return results
