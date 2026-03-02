"""Speech-to-text transcription using faster-whisper."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from faster_whisper import WhisperModel

from claw.config import get_settings, on_reload

log = logging.getLogger(__name__)


class Transcriber:
    """Transcribes audio using faster-whisper (CTranslate2 backend)."""

    def __init__(self) -> None:
        cfg = get_settings().whisper
        self._model_size = cfg.model_size
        self._compute_type = cfg.compute_type
        self._language = cfg.language
        self._beam_size = cfg.beam_size
        self._model: WhisperModel | None = None
        on_reload(self._on_config_reload)

    def load(self) -> None:
        """Load the Whisper model. Downloads from HuggingFace Hub on first run."""
        log.info("Loading Whisper model '%s' (compute_type=%s)...", self._model_size, self._compute_type)
        self._model = WhisperModel(
            self._model_size,
            device="cpu",
            compute_type=self._compute_type,
        )
        log.info("Whisper model loaded")

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription (called via to_thread)."""
        if self._model is None:
            raise RuntimeError("Whisper model not loaded")

        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        log.info("Transcribed (lang=%s, prob=%.2f): %s", info.language, info.language_probability, text)
        return text

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio asynchronously. Runs Whisper in a thread to avoid blocking."""
        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _on_config_reload(self, settings) -> None:
        cfg = settings.whisper
        if cfg.model_size != self._model_size or cfg.compute_type != self._compute_type:
            self._model_size = cfg.model_size
            self._compute_type = cfg.compute_type
            log.info("Whisper config changed, reloading model...")
            self.load()
        self._language = cfg.language
        self._beam_size = cfg.beam_size
