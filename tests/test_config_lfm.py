from second_brain.config import (
    DEFAULT_AGENT_ID,
    DEFAULT_AGENT_LOCAL_DIR,
    DEFAULT_AGENT_OPENVINO_ID,
    load_config,
)


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
    assert config.models.agent_openvino_id == DEFAULT_AGENT_OPENVINO_ID
    assert config.models.agent_local_dir == DEFAULT_AGENT_LOCAL_DIR
    assert config.agent_model_path == tmp_path / "models" / DEFAULT_AGENT_LOCAL_DIR
    assert config.models.reasoner_id == "OpenVINO/Qwen3.5-4B-int4-ov"
    assert config.models.fast_context_tokens == 4096
    assert config.models.deep_context_tokens == 8192
    assert config.models.deep_max_new_tokens == 256
    assert config.models.controller_device == "CPU"
