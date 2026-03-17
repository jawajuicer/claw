"""Fish Speech TTS engine — HTTP client to external Fish Speech server."""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import Iterator

from claw.audio_pipeline.tts.engine import TTSAudio, TTSEngine
from claw.config import PROJECT_ROOT

log = logging.getLogger(__name__)


class FishSpeechEngine(TTSEngine):
    """TTS engine backed by a Fish Speech HTTP server.

    The user manages the Fish Speech server lifecycle independently.
    This engine is an HTTP client that POSTs text and receives audio.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8090",
        reference_audio: str = "",
        reference_text: str = "",
        speed: float = 1.0,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._reference_audio = reference_audio
        self._reference_text = reference_text
        self._speed = speed
        self._sample_rate: int = 44100  # Fish Speech default
        self._client = None

    def load(self) -> None:
        import httpx

        self._client = httpx.Client(timeout=60.0)

        # Verify server is reachable
        try:
            resp = self._client.get(f"{self._server_url}/v1/models")
            resp.raise_for_status()
            log.info("Fish Speech server connected at %s", self._server_url)
        except Exception:
            log.warning(
                "Fish Speech server at %s not reachable — will retry on first synthesis",
                self._server_url,
            )

    def synthesize(self, text: str) -> TTSAudio:
        if self._client is None:
            raise RuntimeError("Fish Speech engine not loaded — call load() first")

        payload = self._build_payload(text)
        resp = self._client.post(
            f"{self._server_url}/v1/tts",
            json=payload,
            headers={"Accept": "audio/wav"},
        )
        resp.raise_for_status()

        # Parse WAV to extract raw PCM
        wav_bytes = resp.content
        pcm_data, sample_rate = self._wav_to_pcm(wav_bytes)
        self._sample_rate = sample_rate

        return TTSAudio(pcm_data=pcm_data, sample_rate=sample_rate)

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        if self._client is None:
            raise RuntimeError("Fish Speech engine not loaded — call load() first")

        payload = self._build_payload(text)

        with self._client.stream(
            "POST",
            f"{self._server_url}/v1/tts",
            json=payload,
            headers={"Accept": "audio/wav"},
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(chunk_size=4096):
                yield chunk

    def _build_payload(self, text: str) -> dict:
        payload: dict = {"text": text, "speed": self._speed}

        if self._reference_audio:
            ref_path = Path(self._reference_audio)
            if not ref_path.is_absolute():
                ref_path = PROJECT_ROOT / ref_path
            if ref_path.exists():
                import base64

                payload["reference_audio"] = base64.b64encode(ref_path.read_bytes()).decode()
                if self._reference_text:
                    payload["reference_text"] = self._reference_text

        return payload

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
        """Extract raw PCM data and sample rate from WAV bytes."""
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                sample_rate = wf.getframerate()
                pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def update_config(self, cfg) -> None:
        self._speed = cfg.speed
        self._reference_audio = cfg.fish_speech_reference_audio
        self._reference_text = cfg.fish_speech_reference_text
        log.info("Fish Speech TTS config updated (speed=%.1f)", cfg.speed)

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        log.info("Fish Speech TTS engine shut down")

    @property
    def engine_name(self) -> str:
        return "fish_speech"
