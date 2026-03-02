"""Piper TTS engine — fast local synthesis using ONNX voice models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from claw.audio_pipeline.tts.engine import TTSAudio, TTSEngine
from claw.config import PROJECT_ROOT

log = logging.getLogger(__name__)


class PiperTTSEngine(TTSEngine):
    """TTS engine backed by piper-tts (local ONNX models)."""

    def __init__(self, model_path: str, speed: float = 1.0, speaker_id: int | None = None) -> None:
        self._model_path = self._resolve_path(model_path)
        self._length_scale = 1.0 / max(speed, 0.1)  # inverse: speed=2 → length_scale=0.5
        self._speaker_id = speaker_id
        self._voice = None
        self._syn_config = None
        self._sample_rate: int = 22050  # updated on load from model config

    @staticmethod
    def _resolve_path(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def load(self) -> None:
        from piper.config import SynthesisConfig
        from piper.voice import PiperVoice

        if not self._model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {self._model_path}")

        self._voice = PiperVoice.load(str(self._model_path))
        self._sample_rate = self._voice.config.sample_rate
        self._syn_config = SynthesisConfig(
            speaker_id=self._speaker_id,
            length_scale=self._length_scale,
        )
        log.info(
            "Piper TTS loaded: %s (rate=%d, length_scale=%.2f)",
            self._model_path.name,
            self._sample_rate,
            self._length_scale,
        )

    def synthesize(self, text: str) -> TTSAudio:
        if self._voice is None:
            raise RuntimeError("Piper engine not loaded — call load() first")

        chunks = []
        for audio_chunk in self._voice.synthesize(text, syn_config=self._syn_config):
            chunks.append(audio_chunk.audio_int16_bytes)

        pcm = b"".join(chunks)
        return TTSAudio(pcm_data=pcm, sample_rate=self._sample_rate)

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        if self._voice is None:
            raise RuntimeError("Piper engine not loaded — call load() first")

        for audio_chunk in self._voice.synthesize(text, syn_config=self._syn_config):
            yield audio_chunk.audio_int16_bytes

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def shutdown(self) -> None:
        self._voice = None
        self._syn_config = None
        log.info("Piper TTS engine shut down")

    @property
    def engine_name(self) -> str:
        return "piper"
