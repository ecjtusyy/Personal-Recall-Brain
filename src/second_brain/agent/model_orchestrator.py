from __future__ import annotations

import gc
from pathlib import Path
from threading import RLock

from .lfm_adapter import OpenVINOLFMModel
from .qwen_reasoner import OpenVINOQwenReasoner


class ModelOrchestrator:
    """Keep LFM resident and guarantee that only one large model is active."""

    def __init__(
        self,
        controller: OpenVINOLFMModel,
        reasoner: OpenVINOQwenReasoner,
        restore_controller: bool = True,
    ) -> None:
        self.controller = controller
        self.reasoner = reasoner
        self.restore_controller = restore_controller
        self._lock = RLock()

    @property
    def available(self) -> bool:
        return self.controller.available

    @property
    def loaded(self) -> bool:
        return self.controller.loaded

    @property
    def reasoner_available(self) -> bool:
        return self.reasoner.available

    def warmup(self) -> None:
        with self._lock:
            self.reasoner.unload()
            self.controller.warmup()

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        with self._lock:
            if self.reasoner.loaded:
                self.reasoner.unload()
                gc.collect()
            return self.controller.generate(prompt, max_new_tokens)

    def analyze_deep(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        image_paths: list[Path] | None = None,
    ) -> str:
        with self._lock:
            self.controller.unload()
            gc.collect()
            try:
                return self.reasoner.generate(prompt, max_new_tokens, image_paths)
            finally:
                self.reasoner.unload()
                gc.collect()
                if self.restore_controller and self.controller.available:
                    self.controller.warmup()

    def unload(self) -> None:
        with self._lock:
            self.controller.unload()
            self.reasoner.unload()
            gc.collect()

