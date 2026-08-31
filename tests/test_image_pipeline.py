import numpy as np

from second_brain.ingestion.image_pipeline import OpenVINOOCREngine


class FakeRapidResult:
    txts = ("公文写作", "主体分析")
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    boxes = np.asarray([[[0, 0], [1, 0], [1, 1], [0, 1]]])


def test_rapidocr_numpy_outputs_are_supported(tmp_path):
    engine = OpenVINOOCREngine()
    engine._engine = lambda _path: FakeRapidResult()
    result = engine.ocr(tmp_path / "note.png")
    assert result.text == "公文写作\n主体分析"
    assert 0.84 < result.confidence < 0.86
    assert len(result.boxes) == 1

