"""Git-based change detection, used to drive incremental documentation
updates: only files that changed since the last processed commit need to
be re-parsed, and only docs whose underlying symbols changed need to be
regenerated.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from repobrain.logging_setup import get_logger

logger = get_logger("git_diff")


@dataclass
class ChangeSet:
    added: list[str]
    modified: list[str]
    deleted: list[str]
    renamed: list[tuple[str, str]]  # (old_path, new_path)

    @property
    def all_changed_paths(self) -> set[str]:
        """Paths that need re-parsing: added, modified, and the new side of renames."""
        changed = set(self.added) | set(self.modified)
        changed |= {new for _, new in self.renamed}
        return changed

    @property
    def all_removed_paths(self) -> set[str]:
        """Paths that must be dropped from cached IR."""
        removed = set(self.deleted)
        removed |= {old for old, _ in self.renamed}
        return removed

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.modified or self.deleted or self.renamed)


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def current_commit(repo_root: Path) -> str:
    """HEAD commit hash, or an empty string for a repo with no commits yet."""
    try:
        return _run_git(repo_root, "rev-parse", "HEAD").strip()
    except RuntimeError:
        return ""


def has_uncommitted_changes(repo_root: Path) -> bool:
    out = _run_git(repo_root, "status", "--porcelain")
    return bool(out.strip())


def diff_since(repo_root: Path, since_commit: str) -> ChangeSet:
    """Compute the change set between `since_commit` and the current
    working tree (including uncommitted changes), so `repobrain update`
    always reflects what's actually on disk.
    """
    if not since_commit:
        return ChangeSet(added=[], modified=[], deleted=[], renamed=[])

    out = _run_git(
        repo_root,
        "diff",
        "--name-status",
        "-M",  # detect renames
        "-z",
        since_commit,
        "--",
    )
    return _parse_name_status(out, since_commit)


def _parse_name_status(raw: str, since_commit: str = "") -> ChangeSet:
    tokens = [t for t in raw.split("\0") if t]
    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []
    renamed: list[tuple[str, str]] = []

    i = 0
    while i < len(tokens):
        status = tokens[i]
        code = status[0]
        if code == "A":
            added.append(tokens[i + 1])
            i += 2
        elif code == "M":
            modified.append(tokens[i + 1])
            i += 2
        elif code == "D":
            deleted.append(tokens[i + 1])
            i += 2
        elif code == "R":
            old, new = tokens[i + 1], tokens[i + 2]
            renamed.append((old, new))
            i += 3
        else:
            # Covers copy (C) and any other status by treating as modified.
            modified.append(tokens[i + 1])
            i += 2

    changeset = ChangeSet(added=added, modified=modified, deleted=deleted, renamed=renamed)
    logger.info(
        "Diff since %s: +%d ~%d -%d renamed=%d",
        since_commit[:8],
        len(added),
        len(modified),
        len(deleted),
        len(renamed),
    )
    return changeset
