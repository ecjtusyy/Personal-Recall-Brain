from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image


class OpenVINOQwenReasoner:
    """On-demand Qwen3.5 multimodal reasoner for deep recall tasks."""

    def __init__(self, model_path: Path, device: str = "CPU", max_new_tokens: int = 768) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._lock = RLock()

    @property
    def available(self) -> bool:
        return (
            (self.model_path / "openvino_language_model.xml").exists()
            and (self.model_path / "openvino_tokenizer.xml").exists()
        )

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def _load(self):
        with self._lock:
            if self._pipeline is None:
                import openvino_genai as ov_genai

                self._pipeline = ov_genai.VLMPipeline(
                    str(self.model_path),
                    self.device,
                    CACHE_DIR=str(self.model_path / ".ov_cache"),
                )
            return self._pipeline

    def generate(self, prompt: str, max_new_tokens: int | None = None, image_paths: list[Path] | None = None) -> str:
        kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.08,
        }
        if image_paths:
            import openvino as ov

            kwargs["images"] = [
                ov.Tensor(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
                for path in image_paths
            ]
        with self._lock:
            pipeline = self._load()
            if not image_paths:
                prompt = pipeline.get_tokenizer().apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    extra_context={"enable_thinking": False},
                )
                kwargs["apply_chat_template"] = False
            result = pipeline.generate(prompt, **kwargs)
        text = str(result).strip()
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
        return text.strip()

    def analyze_image(self, image_path: Path) -> dict[str, object]:
        prompt = (
            "请用中文分析这张学习资料图片。依次给出：可见文字、学科与知识点、"
            "学习者可能的卡点。看不清的内容明确标注不确定，不得猜测。"
        )
        return {
            "description": self.generate(prompt, 512, [image_path]),
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "model": self.model_path.name,
        }

    def unload(self) -> None:
        with self._lock:
            self._pipeline = None
