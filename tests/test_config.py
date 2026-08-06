from repobrain.config import RepoBrainConfig


def test_defaults_load_without_a_project_config():
    config = RepoBrainConfig.load()
    assert config.languages == ["java"]
    assert config.llm.provider == "ollama"
    assert config.llm.model == "qwen3:8b"
    assert "README.md" in config.docs


def test_overrides_take_precedence_over_defaults():
    config = RepoBrainConfig.load(overrides={"llm": {"model": "custom:1b"}, "output_dir": "generated-docs"})
    assert config.llm.model == "custom:1b"
    assert config.output_dir == "generated-docs"
    # unrelated defaults survive the merge
    assert config.llm.provider == "ollama"


def test_project_config_file_overrides_defaults(tmp_path):
    config_file = tmp_path / "repobrain.yaml"
    config_file.write_text("output_dir: custom-docs\nllm:\n  model: mymodel\n")

    config = RepoBrainConfig.load(config_path=config_file)
    assert config.output_dir == "custom-docs"
    assert config.llm.model == "mymodel"


def test_cli_overrides_win_over_project_config_file(tmp_path):
    config_file = tmp_path / "repobrain.yaml"
    config_file.write_text("llm:\n  model: from-file\n")

    config = RepoBrainConfig.load(config_path=config_file, overrides={"llm": {"model": "from-cli"}})
    assert config.llm.model == "from-cli"
