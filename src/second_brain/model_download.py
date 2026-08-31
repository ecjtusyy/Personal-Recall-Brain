from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import load_config


def profile_models(config, profile: str) -> list[str]:
    mapping = {
        "core": [config.models.agent_id],
        "audio": [config.models.asr_id],
        "semantic": [config.models.embedding_id],
        "vision": [config.models.vision_id],
        "all": [config.models.agent_id, config.models.asr_id, config.models.embedding_id, config.models.vision_id],
    }
    return mapping[profile]


def download_models(config_path: str | Path, profile: str = "core") -> list[Path]:
    config = load_config(config_path)
    downloaded: list[Path] = []
    for model_id in profile_models(config, profile):
        target = config.model_dir / model_id.split("/")[-1]
        snapshot_download(repo_id=model_id, local_dir=target)
        downloaded.append(target)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Intel OpenVINO 官方优化模型")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--profile", choices=("core", "audio", "semantic", "vision", "all"), default="core")
    args = parser.parse_args()
    for path in download_models(args.config, args.profile):
        print(f"已准备 OpenVINO 模型：{path}")


if __name__ == "__main__":
    main()

