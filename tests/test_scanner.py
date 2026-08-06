import pytest

from repobrain.scanner import NotAGitRepositoryError, RepoScanner


def test_scan_finds_all_java_files(git_repo):
    scanner = RepoScanner(git_repo)
    files = scanner.scan()
    rel_paths = {f.rel_path for f in files}
    assert rel_paths == {
        "src/main/java/com/example/model/Widget.java",
        "src/main/java/com/example/repo/WidgetRepository.java",
        "src/main/java/com/example/repo/InMemoryWidgetRepository.java",
        "src/main/java/com/example/service/WidgetService.java",
    }


def test_scan_includes_untracked_files(git_repo):
    (git_repo / "src/main/java/com/example/model/NewThing.java").write_text("class NewThing {}\n")
    scanner = RepoScanner(git_repo)
    rel_paths = {f.rel_path for f in scanner.scan()}
    assert "src/main/java/com/example/model/NewThing.java" in rel_paths


def test_scan_respects_gitignore(git_repo):
    (git_repo / ".gitignore").write_text("ignored/\n")
    ignored_dir = git_repo / "ignored"
    ignored_dir.mkdir()
    (ignored_dir / "Ignored.java").write_text("class Ignored {}\n")

    scanner = RepoScanner(git_repo)
    rel_paths = {f.rel_path for f in scanner.scan()}
    assert not any("ignored/" in p for p in rel_paths)


def test_scan_applies_custom_exclude_patterns(git_repo):
    scanner = RepoScanner(git_repo, exclude_patterns=["**/repo/**"])
    rel_paths = {f.rel_path for f in scanner.scan()}
    assert not any("/repo/" in p for p in rel_paths)
    assert "src/main/java/com/example/model/Widget.java" in rel_paths


def test_filter_by_extensions(git_repo):
    scanner = RepoScanner(git_repo)
    files = scanner.scan()
    java_only = scanner.filter_by_extensions(files, {".java"})
    assert len(java_only) == len(files)
    none_matching = scanner.filter_by_extensions(files, {".py"})
    assert none_matching == []


def test_non_git_directory_raises(tmp_path):
    plain_dir = tmp_path / "not_a_repo"
    plain_dir.mkdir()
    with pytest.raises(NotAGitRepositoryError):
        RepoScanner(plain_dir)
