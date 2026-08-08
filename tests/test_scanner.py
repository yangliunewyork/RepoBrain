import pytest

from repobrain.config import RepoBrainConfig
from repobrain.scanner import NotAGitRepositoryError, RepoScanner
from repobrain.scanner.repo_scanner import _is_excluded


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


def test_leading_double_star_pattern_matches_top_level_directory():
    """Regression test: fnmatch's naive glob translation requires a
    literal "/" before the rest of a "**/X" pattern, so it used to miss
    "build/" when that directory sat at the repo root with nothing
    ahead of it — the single most common case for build/target dirs."""
    assert _is_excluded("build/output.txt", ["**/build/**"])
    assert _is_excluded("target/classes/Foo.class", ["**/target/**"])
    assert _is_excluded("src/test/java/Foo.java", ["**/src/test/**"])
    assert not _is_excluded("src/main/java/Foo.java", ["**/src/test/**"])


def test_leading_double_star_pattern_still_matches_nested_directory():
    assert _is_excluded("module/build/output.txt", ["**/build/**"])


def test_default_config_excludes_maven_style_test_sources(git_repo):
    (git_repo / "src/test/java/com/example").mkdir(parents=True)
    (git_repo / "src/test/java/com/example/WidgetServiceTest.java").write_text("class WidgetServiceTest {}\n")

    config = RepoBrainConfig.load()
    scanner = RepoScanner(git_repo, config.exclude_patterns)
    rel_paths = {f.rel_path for f in scanner.scan()}

    assert "src/test/java/com/example/WidgetServiceTest.java" not in rel_paths
    assert "src/main/java/com/example/model/Widget.java" in rel_paths


def test_default_config_excludes_test_suffixed_files_outside_test_dir(git_repo):
    """Some codebases put *Test.java / *IT.java files alongside main
    sources rather than under src/test — the suffix-based patterns
    should still catch these even outside a conventional test directory."""
    (git_repo / "src/main/java/com/example/model/WidgetIT.java").write_text("class WidgetIT {}\n")

    config = RepoBrainConfig.load()
    scanner = RepoScanner(git_repo, config.exclude_patterns)
    rel_paths = {f.rel_path for f in scanner.scan()}

    assert "src/main/java/com/example/model/WidgetIT.java" not in rel_paths
