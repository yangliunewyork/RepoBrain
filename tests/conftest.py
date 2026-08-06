import shutil
import subprocess
from pathlib import Path

import pytest

from repobrain.llm.base import LLMProvider

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_java_repo"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path):
    """A throwaway copy of the sample Java fixture, initialized as a Git
    repo with one commit, so tests can freely mutate files and history."""
    dest = tmp_path / "repo"
    shutil.copytree(FIXTURE_DIR, dest)
    _git(dest, "init", "-q")
    _git(dest, "config", "user.email", "test@example.com")
    _git(dest, "config", "user.name", "Test")
    _git(dest, "add", "-A")
    _git(dest, "commit", "-q", "-m", "initial commit")
    return dest


def git_commit_all(repo_path, message="update"):
    _git(repo_path, "add", "-A")
    _git(repo_path, "commit", "-q", "-m", message)


class FakeLLMProvider(LLMProvider):
    """Deterministic stand-in for a real LLM, so tests don't depend on a
    running Ollama daemon. Returns a short, stable string derived from the
    prompt length so different prompts are distinguishable in assertions."""

    def __init__(self):
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append(prompt)
        return f"Generated content ({len(prompt)} char prompt, call #{len(self.calls)})"


@pytest.fixture
def fake_llm():
    return FakeLLMProvider()
