"""Persists everything needed to make `repobrain update` incremental:
the last processed commit, the merged RepoIR, and the per-document
fingerprints used to decide which docs to regenerate.

State lives at `<repo_root>/<state_dir>/state.json` inside the analyzed
repository (default `.repobrain/`), not inside RepoBrain itself — each
target repo tracks its own analysis state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from repobrain.ir.models import RepoIR

_STATE_FILENAME = "state.json"


@dataclass
class RepoBrainState:
    last_commit: str = ""
    repo_ir: RepoIR | None = None
    doc_fingerprints: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "last_commit": self.last_commit,
            "repo_ir": self.repo_ir.to_dict() if self.repo_ir else None,
            "doc_fingerprints": self.doc_fingerprints,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def from_dict(data: dict) -> "RepoBrainState":
        repo_ir = RepoIR.from_dict(data["repo_ir"]) if data.get("repo_ir") else None
        return RepoBrainState(
            last_commit=data.get("last_commit", ""),
            repo_ir=repo_ir,
            doc_fingerprints=data.get("doc_fingerprints", {}),
        )


class StateStore:
    def __init__(self, repo_root: Path, state_dir: str):
        self.state_path = Path(repo_root) / state_dir / _STATE_FILENAME

    def load(self) -> RepoBrainState | None:
        if not self.state_path.is_file():
            return None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return RepoBrainState.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def save(self, state: RepoBrainState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
