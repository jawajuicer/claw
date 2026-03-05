"""Acknowledgment chime — plays a short tone when the wake word is detected."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

from claw.config import get_settings

log = logging.getLogger(__name__)

_SAMPLE_RATE = 44100


def play_listening_chime() -> None:
    """Generate and play a short two-tone chime to acknowledge wake word detection.

    Runs synchronously and blocks until playback completes (~200ms).
    """
    cfg = get_settings().audio
    if not cfg.chime_enabled:
        return

    try:
        log.info("Playing listening chime (freq=%d, dur=%dms, vol=%.1f)",
                 cfg.chime_frequency, cfg.chime_duration_ms, cfg.chime_volume)

        half_dur = cfg.chime_duration_ms / 2000
        n_samples = int(_SAMPLE_RATE * half_dur)
        t = np.linspace(0, half_dur, n_samples, endpoint=False)

        # Two-tone ascending chime (e.g. 660 Hz → 880 Hz)
        tone1 = np.sin(2 * np.pi * cfg.chime_frequency * 0.75 * t)
        tone2 = np.sin(2 * np.pi * cfg.chime_frequency * t)
        samples = np.concatenate([tone1, tone2]).astype(np.float32)

        # Fade in/out envelope to avoid clicks (5ms each)
        fade_len = int(_SAMPLE_RATE * 0.005)
        samples[:fade_len] *= np.linspace(0, 1, fade_len).astype(np.float32)
        samples[-fade_len:] *= np.linspace(1, 0, fade_len).astype(np.float32)

        samples *= cfg.chime_volume

        sd.play(samples, samplerate=_SAMPLE_RATE)
        sd.wait()
        log.info("Chime playback complete")
    except Exception:
        log.exception("Failed to play listening chime")
