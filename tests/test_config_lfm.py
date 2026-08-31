from second_brain.config import DEFAULT_AGENT_ID, DEFAULT_AGENT_LOCAL_DIR, load_config


def test_default_config_uses_lfm_as_local_agent(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"source_roots = ['{source.as_posix()}']\n"
        "[models]\n"
        "agent_enabled = true\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.models.agent_id == DEFAULT_AGENT_ID
    assert config.models.agent_local_dir == DEFAULT_AGENT_LOCAL_DIR
    assert config.agent_model_path == tmp_path / "models" / DEFAULT_AGENT_LOCAL_DIR
