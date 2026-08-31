from __future__ import annotations

from pathlib import Path

from ..models import OCRResult


class OpenVINOOCREngine:
    """RapidOCR pipeline forced to use Intel OpenVINO for all inference stages."""

    def __init__(self) -> None:
        self._engine = None

    def _load(self):
        if self._engine is None:
            from rapidocr import EngineType, RapidOCR

            self._engine = RapidOCR(
                params={
                    "Det.engine_type": EngineType.OPENVINO,
                    "Cls.engine_type": EngineType.OPENVINO,
                    "Rec.engine_type": EngineType.OPENVINO,
                }
            )
        return self._engine

    def ocr(self, image_path: Path) -> OCRResult:
        result = self._load()(str(image_path))
        texts = list(getattr(result, "txts", None) or [])
        scores = [float(x) for x in (getattr(result, "scores", None) or [])]
        boxes = list(getattr(result, "boxes", None) or [])
        if not texts and isinstance(result, (list, tuple)):
            rows = result[0] if len(result) == 2 and isinstance(result[0], list) else result
            for row in rows or []:
                if isinstance(row, (list, tuple)) and len(row) >= 3:
                    boxes.append(row[0])
                    texts.append(str(row[1]))
                    scores.append(float(row[2]))
        confidence = sum(scores) / len(scores) if scores else 0.0
        return OCRResult("\n".join(texts).strip(), confidence, boxes)

