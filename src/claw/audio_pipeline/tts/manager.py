"""TTS Manager — orchestrates engine selection, synthesis, and playback."""

from __future__ import annotations

import asyncio
import io
import logging
import wave

import numpy as np
import sounddevice as sd

from claw.audio_pipeline.tts.engine import TTSAudio, TTSEngine
from claw.config import get_settings, on_reload

log = logging.getLogger(__name__)


class TTSManager:
    """High-level TTS interface used by the rest of The Claw.

    Handles engine initialization, audio synthesis, speaker playback,
    and WAV encoding for the admin API.
    """

    def __init__(self) -> None:
        self._engine: TTSEngine | None = None
        self._speaking = False
        on_reload(self._on_config_reload)

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def initialize(self) -> None:
        """Create and load the TTS engine from current config."""
        cfg = get_settings().tts
        if not cfg.enabled:
            log.info("TTS disabled in config")
            return

        self._engine = self._create_engine(cfg)
        try:
            self._engine.load()
            log.info("TTS engine '%s' initialized", self._engine.engine_name)
        except Exception:
            log.exception("TTS engine failed to load (non-fatal, TTS disabled)")
            self._engine = None

    @staticmethod
    def _create_engine(cfg) -> TTSEngine:
        """Factory: build the right engine from config."""
        if cfg.engine == "fish_speech":
            from claw.audio_pipeline.tts.fish_engine import FishSpeechEngine

            return FishSpeechEngine(
                server_url=cfg.fish_speech_url,
                reference_audio=cfg.fish_speech_reference_audio,
                reference_text=cfg.fish_speech_reference_text,
                speed=cfg.speed,
            )
        else:
            from claw.audio_pipeline.tts.piper_engine import PiperTTSEngine

            return PiperTTSEngine(
                model_path=cfg.piper_model,
                speed=cfg.speed,
                speaker_id=cfg.piper_speaker_id,
            )

    async def speak(self, text: str) -> None:
        """Synthesize text and play through the physical speaker.

        Runs synthesis + playback in a thread to avoid blocking the event loop.
        """
        if not self._engine:
            return
        self._speaking = True
        try:
            await asyncio.to_thread(self._speak_blocking, text)
        finally:
            self._speaking = False

    def _speak_blocking(self, text: str) -> None:
        """Blocking synthesis + speaker playback (runs in thread)."""
        try:
            audio = self._engine.synthesize(text)
            self._play_pcm(audio)
        except Exception:
            log.exception("TTS speak failed")

    @staticmethod
    def _play_pcm(audio: TTSAudio) -> None:
        """Play raw PCM audio through the default output device."""
        samples = np.frombuffer(audio.pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(samples, samplerate=audio.sample_rate)
        sd.wait()

    async def synthesize_wav(self, text: str) -> bytes:
        """Synthesize text and return WAV bytes (for the admin API endpoint)."""
        if not self._engine:
            return b""
        return await asyncio.to_thread(self._synthesize_wav_blocking, text)

    def _synthesize_wav_blocking(self, text: str) -> bytes:
        """Blocking WAV synthesis (runs in thread)."""
        audio = self._engine.synthesize(text)
        return self._pcm_to_wav(audio)

    @staticmethod
    def _pcm_to_wav(audio: TTSAudio) -> bytes:
        """Encode raw PCM into a WAV byte string."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(audio.channels)
            wf.setsampwidth(audio.sample_width)
            wf.setframerate(audio.sample_rate)
            wf.writeframes(audio.pcm_data)
        return buf.getvalue()

    def shutdown(self) -> None:
        """Release engine resources."""
        if self._engine:
            self._engine.shutdown()
            self._engine = None

    def _on_config_reload(self, settings) -> None:
        """Reinitialize engine if TTS config changed."""
        cfg = settings.tts
        if not cfg.enabled:
            if self._engine:
                self.shutdown()
                log.info("TTS disabled on config reload")
            return

        current_name = self._engine.engine_name if self._engine else None
        if current_name != cfg.engine:
            log.info("TTS engine changed to '%s', reinitializing...", cfg.engine)
            self.shutdown()
            self.initialize()
        else:
            # Recreate engine with updated params (speed, model, etc.)
            log.info("TTS config updated, reinitializing engine...")
            self.shutdown()
            self.initialize()
