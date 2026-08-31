from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import TranscriptSegment


def _decode_mono_16k(path: Path) -> list[float]:
    import av

    container = av.open(str(path), mode="r")
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    parts: list[np.ndarray] = []
    for frame in container.decode(stream):
        converted = resampler.resample(frame)
        converted_frames = converted if isinstance(converted, list) else [converted]
        for item in converted_frames:
            if item is not None:
                parts.append(item.to_ndarray().reshape(-1).astype(np.float32))
    container.close()
    return np.concatenate(parts).tolist() if parts else []


class OpenVINOASREngine:
    def __init__(self, model_path: Path, device: str = "CPU") -> None:
        self.model_path = model_path
        self.device = device
        self._pipeline = None

    @property
    def available(self) -> bool:
        return (self.model_path / "openvino_encoder_model.xml").exists()

    def _load(self):
        if self._pipeline is None:
            import openvino_genai as ov_genai

            self._pipeline = ov_genai.WhisperPipeline(str(self.model_path), self.device)
        return self._pipeline

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        if not self.available:
            return []
        audio = _decode_mono_16k(audio_path)
        if not audio:
            return []
        result = self._load().generate(
            audio,
            language="<|zh|>",
            task="transcribe",
            return_timestamps=True,
            max_new_tokens=448,
        )
        chunks = getattr(result, "chunks", None) or []
        segments: list[TranscriptSegment] = []
        for chunk in chunks:
            start = float(getattr(chunk, "start_ts", 0.0))
            end = float(getattr(chunk, "end_ts", start))
            text = str(getattr(chunk, "text", "")).strip()
            if text:
                segments.append(TranscriptSegment(start, end, text))
        if not segments:
            text = str(getattr(result, "texts", [result])[0]).strip()
            if text:
                segments.append(TranscriptSegment(0.0, 0.0, text))
        return segments

