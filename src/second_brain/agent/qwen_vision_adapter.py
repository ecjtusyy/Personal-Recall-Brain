from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


class OpenVINOVisionEngine:
    def __init__(self, model_path: Path, device: str = "CPU") -> None:
        self.model_path = model_path
        self.device = device
        self._pipeline = None

    @property
    def available(self) -> bool:
        return (self.model_path / "openvino_model.xml").exists()

    def _load(self):
        if self._pipeline is None:
            import openvino_genai as ov_genai

            self._pipeline = ov_genai.VLMPipeline(str(self.model_path), self.device)
        return self._pipeline

    def analyze(self, image_path: Path) -> dict[str, object]:
        import openvino as ov

        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        tensor = ov.Tensor(rgb)
        prompt = "请用中文简洁分析这张学习资料图片：提取可见文字、主题、关键知识点；不确定内容不要猜测。"
        result = self._load().generate(prompt, images=[tensor], max_new_tokens=384, do_sample=False)
        return {
            "description": str(result).strip(),
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "model": self.model_path.name,
        }

    def unload(self) -> None:
        self._pipeline = None

