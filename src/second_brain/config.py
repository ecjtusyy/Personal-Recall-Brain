from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_EXTENSIONS = (
    ".docx", ".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".webp",
    ".wav", ".mp3", ".m4a", ".flac",
)


@dataclass(frozen=True)
class IngestionConfig:
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    extract_docx_images: bool = True
    enable_ocr: bool = True
    enable_asr: bool = True
    max_file_mb: int = 500


@dataclass(frozen=True)
class ModelConfig:
    agent_id: str = "OpenVINO/Qwen3-1.7B-int4-ov"
    embedding_id: str = "OpenVINO/Qwen3-Embedding-0.6B-int4-cw-ov"
    asr_id: str = "OpenVINO/whisper-small-int8-ov"
    vision_id: str = "OpenVINO/Qwen3.5-4B-int8-ov"
    agent_enabled: bool = True
    semantic_enabled: bool = False
    vision_enabled: bool = False
    max_new_tokens: int = 512


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    study_year: int
    data_dir: Path
    model_dir: Path
    device: str
    source_roots: tuple[Path, ...]
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "brain.db"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "extracted_assets"

    def ensure_runtime_dirs(self) -> None:
        source_resolved = {root.resolve(strict=False) for root in self.source_roots}
        for runtime_dir in (self.data_dir, self.model_dir, self.assets_dir):
            resolved = runtime_dir.resolve(strict=False)
            if resolved in source_resolved or any(root in resolved.parents for root in source_resolved):
                raise ValueError(f"运行目录不得位于资料源目录内: {runtime_dir}")
            runtime_dir.mkdir(parents=True, exist_ok=True)


def _resolve(base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path = "config.toml") -> AppConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        example = config_path.with_name("config.example.toml")
        if not example.exists():
            raise FileNotFoundError(f"找不到配置文件: {config_path}")
        shutil.copyfile(example, config_path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    ingestion_raw = raw.get("ingestion", {})
    models_raw = raw.get("models", {})
    config = AppConfig(
        config_path=config_path,
        study_year=int(raw.get("study_year", 2026)),
        data_dir=_resolve(base, raw.get("data_dir", "data")),
        model_dir=_resolve(base, raw.get("model_dir", "models")),
        device=str(raw.get("device", "CPU")).upper(),
        source_roots=tuple(Path(item).resolve(strict=False) for item in raw.get("source_roots", [])),
        ingestion=IngestionConfig(
            extensions=tuple(str(x).lower() for x in ingestion_raw.get("extensions", DEFAULT_EXTENSIONS)),
            extract_docx_images=bool(ingestion_raw.get("extract_docx_images", True)),
            enable_ocr=bool(ingestion_raw.get("enable_ocr", True)),
            enable_asr=bool(ingestion_raw.get("enable_asr", True)),
            max_file_mb=int(ingestion_raw.get("max_file_mb", 500)),
        ),
        models=ModelConfig(
            agent_id=str(models_raw.get("agent_id", ModelConfig.agent_id)),
            embedding_id=str(models_raw.get("embedding_id", ModelConfig.embedding_id)),
            asr_id=str(models_raw.get("asr_id", ModelConfig.asr_id)),
            vision_id=str(models_raw.get("vision_id", ModelConfig.vision_id)),
            agent_enabled=bool(models_raw.get("agent_enabled", True)),
            semantic_enabled=bool(models_raw.get("semantic_enabled", False)),
            vision_enabled=bool(models_raw.get("vision_enabled", False)),
            max_new_tokens=int(models_raw.get("max_new_tokens", 512)),
        ),
    )
    if not config.source_roots:
        raise ValueError("source_roots 不能为空")
    config.ensure_runtime_dirs()
    return config

