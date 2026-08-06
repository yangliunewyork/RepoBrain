from click.testing import CliRunner

from repobrain import cli


def test_scan_command_reports_structure(git_repo):
    runner = CliRunner()
    result = runner.invoke(cli.main, ["scan", str(git_repo)])

    assert result.exit_code == 0, result.output
    assert "Types discovered:   4" in result.output
    assert "com.example.model" in result.output
    assert "com.example.repo" in result.output


def test_scan_command_on_non_git_directory_fails_cleanly(tmp_path):
    plain_dir = tmp_path / "not_a_repo"
    plain_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(cli.main, ["scan", str(plain_dir)])

    assert result.exit_code != 0
    assert "not a Git repository" in result.output or "Git repository" in result.output


def test_generate_command_with_fake_llm(monkeypatch, git_repo, fake_llm):
    monkeypatch.setattr(cli, "get_provider", lambda llm_config: fake_llm)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["generate", str(git_repo)])

    assert result.exit_code == 0, result.output
    assert "README.md" in result.output
    assert (git_repo / "docs/generated/README.md").is_file()


def test_update_command_reports_no_op_on_second_run(monkeypatch, git_repo, fake_llm):
    monkeypatch.setattr(cli, "get_provider", lambda llm_config: fake_llm)

    runner = CliRunner()
    runner.invoke(cli.main, ["generate", str(git_repo)])
    result = runner.invoke(cli.main, ["update", str(git_repo)])

    assert result.exit_code == 0, result.output
    assert "nothing to do" in result.output


def test_generate_fails_cleanly_when_llm_unavailable(monkeypatch, git_repo, fake_llm):
    fake_llm.is_available = lambda: False
    monkeypatch.setattr(cli, "get_provider", lambda llm_config: fake_llm)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["generate", str(git_repo)])

    assert result.exit_code != 0
    assert "not reachable" in result.output
