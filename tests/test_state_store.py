from datetime import datetime, timezone

from repobrain.cache.state_store import RepoBrainState, StateStore
from repobrain.ir.models import FileIR, RepoIR


def _sample_state() -> RepoBrainState:
    file_ir = FileIR(path="A.java", language="java", content_hash="deadbeef", package="p")
    repo_ir = RepoIR(repo_root="/repo", generated_at=datetime.now(timezone.utc).isoformat(), files={"A.java": file_ir})
    return RepoBrainState(last_commit="abc123", repo_ir=repo_ir, doc_fingerprints={"README.md": "hash1"})


def test_load_returns_none_when_no_state_file(tmp_path):
    store = StateStore(tmp_path, ".repobrain")
    assert store.load() is None


def test_save_and_load_round_trip(tmp_path):
    store = StateStore(tmp_path, ".repobrain")
    original = _sample_state()
    store.save(original)

    loaded = store.load()
    assert loaded is not None
    assert loaded.last_commit == "abc123"
    assert loaded.doc_fingerprints == {"README.md": "hash1"}
    assert loaded.repo_ir.files["A.java"].content_hash == "deadbeef"


def test_state_file_written_under_configured_state_dir(tmp_path):
    store = StateStore(tmp_path, "custom_state")
    store.save(_sample_state())
    assert (tmp_path / "custom_state" / "state.json").is_file()


def test_load_returns_none_for_corrupt_state_file(tmp_path):
    state_dir = tmp_path / ".repobrain"
    state_dir.mkdir()
    (state_dir / "state.json").write_text("not valid json{")

    store = StateStore(tmp_path, ".repobrain")
    assert store.load() is None
