from __future__ import annotations

import re
from pathlib import Path


class OpenVINOLFMModel:
    """Resident LiquidAI LFM2.5-2.6B model running through OpenVINO GenAI."""

    def __init__(self, model_path: Path, device: str = "CPU", max_new_tokens: int = 512) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    @property
    def available(self) -> bool:
        return (self.model_path / "openvino_model.xml").exists()

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def _load(self):
        if self._pipeline is None:
            import openvino_genai as ov_genai

            self._pipeline = ov_genai.LLMPipeline(
                str(self.model_path),
                self.device,
                CACHE_DIR=str(self.model_path / ".ov_cache"),
            )
        return self._pipeline

    def warmup(self) -> None:
        if not self.available:
            raise FileNotFoundError(f"找不到 OpenVINO LFM 模型：{self.model_path}")
        self._load()

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        result = self._load().generate(
            prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.1,
            rng_seed=42,
        )
        text = str(result).strip()
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def unload(self) -> None:
        self._pipeline = None
