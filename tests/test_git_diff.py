from repobrain.scanner.git_diff import current_commit, diff_since, has_uncommitted_changes

from tests.conftest import git_commit_all


def test_current_commit_returns_a_hash(git_repo):
    commit = current_commit(git_repo)
    assert len(commit) == 40


def test_no_diff_right_after_commit(git_repo):
    commit = current_commit(git_repo)
    changeset = diff_since(git_repo, commit)
    assert changeset.is_empty


def test_diff_detects_modified_file(git_repo):
    commit = current_commit(git_repo)
    target = git_repo / "src/main/java/com/example/model/Widget.java"
    target.write_text(target.read_text() + "\n// trailing comment\n")

    changeset = diff_since(git_repo, commit)
    assert "src/main/java/com/example/model/Widget.java" in changeset.modified
    assert "src/main/java/com/example/model/Widget.java" in changeset.all_changed_paths


def test_diff_detects_deleted_file(git_repo):
    commit = current_commit(git_repo)
    target = git_repo / "src/main/java/com/example/model/Widget.java"
    target.unlink()

    changeset = diff_since(git_repo, commit)
    assert "src/main/java/com/example/model/Widget.java" in changeset.deleted
    assert "src/main/java/com/example/model/Widget.java" in changeset.all_removed_paths


def test_diff_ignores_new_untracked_files(git_repo):
    """git diff never reports untracked files — the pipeline layer must
    handle brand-new files separately by comparing against the cache."""
    commit = current_commit(git_repo)
    (git_repo / "src/main/java/com/example/model/NewThing.java").write_text("class NewThing {}\n")

    changeset = diff_since(git_repo, commit)
    assert changeset.is_empty


def test_diff_detects_added_file_once_committed(git_repo):
    commit = current_commit(git_repo)
    (git_repo / "src/main/java/com/example/model/NewThing.java").write_text("class NewThing {}\n")
    git_commit_all(git_repo, "add NewThing")

    changeset = diff_since(git_repo, commit)
    assert "src/main/java/com/example/model/NewThing.java" in changeset.added


def test_has_uncommitted_changes(git_repo):
    assert not has_uncommitted_changes(git_repo)
    (git_repo / "src/main/java/com/example/model/Widget.java").write_text("// dirty\n")
    assert has_uncommitted_changes(git_repo)


def test_diff_since_empty_commit_returns_empty_changeset(git_repo):
    changeset = diff_since(git_repo, "")
    assert changeset.is_empty
