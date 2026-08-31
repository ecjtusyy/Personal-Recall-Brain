from pathlib import Path

from second_brain.config import AppConfig, IngestionConfig, ModelConfig
from second_brain.model_download import _export_lfm_agent


def test_lfm_is_exported_from_official_source_as_openvino_int4(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = AppConfig(
        config_path=tmp_path / "config.toml",
        study_year=2026,
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        device="CPU",
        source_roots=(source,),
        ingestion=IngestionConfig(),
        models=ModelConfig(),
    )
    config.ensure_runtime_dirs()
    optimum_cli = tmp_path / "optimum-cli.exe"
    optimum_cli.write_bytes(b"test")
    calls = []

    def fake_runner(command, check):
        calls.append(command)
        assert check is True
        output = Path(command[-1])
        output.mkdir(parents=True)
        (output / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")

    result = _export_lfm_agent(config, runner=fake_runner, optimum_cli=optimum_cli)
    assert result == config.agent_model_path
    assert (result / "openvino_model.xml").exists()
    command = calls[0]
    assert command[command.index("--model") + 1] == "LiquidAI/LFM2.5-2.6B"
    assert command[command.index("--weight-format") + 1] == "int4"
    assert command[command.index("--group-size") + 1] == "128"
    assert "--sym" in command
    assert command[command.index("--backup-precision") + 1] == "int8_sym"
