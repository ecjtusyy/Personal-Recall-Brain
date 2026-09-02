from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import load_config


LFM_OPENVINO_FILES = (
    "LICENSE",
    "NOTICE",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "openvino_config.json",
    "openvino_detokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_model.bin",
    "openvino_model.xml",
    "openvino_tokenizer.bin",
    "openvino_tokenizer.xml",
    "tokenizer.json",
    "tokenizer_config.json",
)


def profile_models(config, profile: str) -> list[str]:
    mapping = {
        "core": [config.models.agent_id],
        "audio": [config.models.asr_id],
        "semantic": [config.models.embedding_id],
        "vision": [config.models.reasoner_id],
        "deep": [config.models.reasoner_id],
        "all": [config.models.agent_id, config.models.asr_id, config.models.embedding_id, config.models.reasoner_id],
    }
    return mapping[profile]


def _export_lfm_agent(config, runner=None, optimum_cli: Path | None = None) -> Path:
    target = config.agent_model_path
    if (target / "openvino_model.xml").exists():
        return target
    optimum_cli = optimum_cli or Path(sys.executable).with_name(
        "optimum-cli.exe" if sys.platform == "win32" else "optimum-cli"
    )
    if not optimum_cli.exists():
        raise RuntimeError("缺少模型转换组件，请重新运行“安装与下载模型.bat”")
    if target.exists():
        resolved_target = target.resolve(strict=False)
        model_root = config.model_dir.resolve(strict=False)
        if model_root not in resolved_target.parents:
            raise RuntimeError("拒绝清理项目模型目录之外的不完整模型")
        shutil.rmtree(target)
    partial = target.with_name(f"{target.name}.converting")
    if partial.exists():
        resolved = partial.resolve(strict=False)
        model_root = config.model_dir.resolve(strict=False)
        if model_root not in resolved.parents:
            raise RuntimeError("拒绝清理项目模型目录之外的转换缓存")
        shutil.rmtree(partial)
    command = [
        str(optimum_cli), "export", "openvino",
        "--model", config.models.agent_id,
        "--trust-remote-code",
        "--weight-format", "int4",
        "--sym",
        "--group-size", "128",
        "--backup-precision", "int8_sym",
        "--cache_dir", str(config.model_dir / ".hf-cache"),
        str(partial),
    ]
    (runner or subprocess.run)(command, check=True)
    if not (partial / "openvino_model.xml").exists():
        raise RuntimeError("LFM 转换结束但未生成 openvino_model.xml")
    partial.replace(target)
    return target


def _prepare_lfm_agent(config) -> Path:
    target = config.agent_model_path
    if (target / "openvino_model.xml").exists():
        return target
    if config.models.agent_openvino_id:
        snapshot_download(
            repo_id=config.models.agent_openvino_id,
            local_dir=target,
            allow_patterns=LFM_OPENVINO_FILES,
        )
        if not (target / "openvino_model.xml").exists():
            raise RuntimeError("下载结束但未找到 LFM OpenVINO IR")
        return target
    return _export_lfm_agent(config)


def download_models(config_path: str | Path, profile: str = "core") -> list[Path]:
    config = load_config(config_path)
    downloaded: list[Path] = []
    for model_id in profile_models(config, profile):
        if model_id == config.models.agent_id and not model_id.startswith("OpenVINO/"):
            target = _prepare_lfm_agent(config)
        else:
            target = config.model_dir / model_id.split("/")[-1]
            snapshot_download(repo_id=model_id, local_dir=target)
        downloaded.append(target)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Intel OpenVINO 官方优化模型")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--profile", choices=("core", "audio", "semantic", "deep", "vision", "all"), default="core")
    args = parser.parse_args()
    for path in download_models(args.config, args.profile):
        print(f"已准备 OpenVINO 模型：{path}")


if __name__ == "__main__":
    main()

