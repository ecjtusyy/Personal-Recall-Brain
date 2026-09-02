from __future__ import annotations

from pathlib import Path

from .qwen_reasoner import OpenVINOQwenReasoner


class OpenVINOVisionEngine(OpenVINOQwenReasoner):
    def __init__(self, model_path: Path, device: str = "CPU") -> None:
        super().__init__(model_path, device, 512)

    def analyze(self, image_path: Path) -> dict[str, object]:
        return super().analyze_image(image_path)
