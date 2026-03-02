"""Wake word detection using openWakeWord with ONNX backend."""

from __future__ import annotations

import logging

import numpy as np
import openwakeword
from openwakeword.model import Model as OWWModel

from claw.config import get_settings, on_reload

log = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects wake words in streaming audio chunks."""

    def __init__(self) -> None:
        cfg = get_settings().wake
        self._model: OWWModel | None = None
        self._model_paths = list(cfg.model_paths)
        self._thresholds = dict(cfg.thresholds)
        self._default_threshold = cfg.default_threshold
        self._framework = cfg.inference_framework
        self._paused = False
        on_reload(self._on_config_reload)

    def load(self) -> None:
        """Download/load the wake word model(s)."""
        openwakeword.utils.download_models()
        self._model = OWWModel(
            wakeword_models=self._model_paths,
            inference_framework=self._framework,
        )
        log.info(
            "Wake word models loaded: %s (framework=%s)",
            ", ".join(self._model_paths),
            self._framework,
        )

    def pause(self) -> None:
        """Suppress wake word detection (e.g., during TTS playback)."""
        self._paused = True
        self.reset()

    def resume(self) -> None:
        """Re-enable wake word detection after pause."""
        self._paused = False
        self.reset()

    def process_chunk(self, chunk: np.ndarray) -> str | None:
        """Process a single 80ms audio chunk.

        Args:
            chunk: float32 array of 1280 samples at 16kHz.

        Returns:
            Wake word name if detected above threshold, else None.
        """
        if self._model is None or self._paused:
            return None

        prediction = self._model.predict(chunk)

        for name, score in prediction.items():
            threshold = self._thresholds.get(name, self._default_threshold)
            if score >= threshold:
                log.info("Wake word '%s' detected (score=%.3f, threshold=%.3f)", name, score, threshold)
                self._model.reset()
                return name

        return None

    def reset(self) -> None:
        """Reset the model's internal state."""
        if self._model is not None:
            self._model.reset()

    def _on_config_reload(self, settings) -> None:
        cfg = settings.wake
        self._default_threshold = cfg.default_threshold
        self._thresholds = dict(cfg.thresholds)

        if list(cfg.model_paths) != self._model_paths:
            self._model_paths = list(cfg.model_paths)
            log.info("Wake word models changed to %s, reloading...", self._model_paths)
            self.load()
        else:
            log.info("Wake word thresholds updated (default=%.2f)", self._default_threshold)
