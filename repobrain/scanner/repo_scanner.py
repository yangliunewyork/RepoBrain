"""Repository file discovery.

Lists the files RepoBrain should consider for analysis. Delegates
.gitignore handling to `git ls-files` (the repository is assumed to be a
Git working tree — this is a hard requirement of RepoBrain, since
incremental updates rely on Git history too) and applies a small set of
additional glob excludes on top for things users don't want tracked in
`.gitignore` itself (build output, vendored dirs, etc.).
"""
from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path

from repobrain.logging_setup import get_logger

logger = get_logger("scanner")


class NotAGitRepositoryError(RuntimeError):
    pass


@dataclass
class ScannedFile:
    abs_path: Path
    rel_path: str  # POSIX-style, relative to repo root


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def is_git_repository(repo_root: Path) -> bool:
    try:
        out = _run_git(repo_root, "rev-parse", "--is-inside-work-tree")
        return out.strip() == "true"
    except RuntimeError:
        return False


def _is_excluded(rel_path: str, exclude_patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_patterns)


class RepoScanner:
    """Discovers candidate source files in a Git repository."""

    def __init__(self, repo_root: Path, exclude_patterns: list[str] | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.exclude_patterns = exclude_patterns or []
        if not is_git_repository(self.repo_root):
            raise NotAGitRepositoryError(
                f"{self.repo_root} is not a Git repository. RepoBrain requires "
                "the target to be a Git working tree (used for .gitignore "
                "handling and incremental updates)."
            )

    def scan(self) -> list[ScannedFile]:
        """Return every non-ignored, non-excluded file in the repo."""
        # -c: cached (tracked), -o: others (untracked), --exclude-standard
        # applies .gitignore / .git/info/exclude / global excludes.
        raw = _run_git(
            self.repo_root, "ls-files", "-c", "-o", "--exclude-standard", "-z"
        )
        rel_paths = [p for p in raw.split("\0") if p]

        files: list[ScannedFile] = []
        for rel in rel_paths:
            if _is_excluded(rel, self.exclude_patterns):
                continue
            abs_path = self.repo_root / rel
            if not abs_path.is_file():
                continue
            files.append(ScannedFile(abs_path=abs_path, rel_path=rel))

        logger.info(
            "Scanned %s: %d candidate files (%d excluded by pattern)",
            self.repo_root,
            len(files),
            len(rel_paths) - len(files),
        )
        return files

    def filter_by_extensions(
        self, files: list[ScannedFile], extensions: set[str]
    ) -> list[ScannedFile]:
        return [f for f in files if Path(f.rel_path).suffix in extensions]
