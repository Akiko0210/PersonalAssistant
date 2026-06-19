"""Earcon synthesis (§10).

A small, consistent, non-speech vocabulary so the agent's state is identifiable by ear
alone (§C5, FR-V1). Each earcon is short and unmistakably different from the others
(§10 rules). Tones are synthesized with numpy (no audio assets to ship) and played
through the output device.

Mapping (§10):
  unmute/listening   -> short rising two-tone
  mute               -> single descending tone   (no spoken "muted" — §V3)
  start note session -> distinct ascending chime
  stop note session  -> short confirming tone
  summary ready      -> soft chime
  working tick       -> subtle soft tick          (§FR-V4)
  error              -> distinct low buzz          (§FR-V5)
"""

from __future__ import annotations

from enum import Enum

import numpy as np

SAMPLE_RATE = 16000


class Earcon(str, Enum):
    LISTENING = "listening"
    MUTED = "muted"
    START_NOTES = "start_notes"
    STOP_NOTES = "stop_notes"
    SUMMARY_READY = "summary_ready"
    WORKING = "working"
    ERROR = "error"


def _tone(freq: float, dur: float, *, sr: int = SAMPLE_RATE, fade: float = 0.01) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
    # Short raised-cosine fade in/out so tones don't click.
    n = max(1, int(sr * fade))
    env = np.ones_like(wave)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    env[:n] *= ramp
    env[-n:] *= ramp[::-1]
    return wave * env


def _seq(*tones: np.ndarray, gap: float = 0.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    silence = np.zeros(int(sr * gap), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, tone in enumerate(tones):
        if i:
            parts.append(silence)
        parts.append(tone)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def render(earcon: Earcon, *, volume: float = 0.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render an earcon to a mono float32 buffer scaled by ``volume``."""
    if earcon is Earcon.LISTENING:           # rising two-tone
        buf = _seq(_tone(660, 0.09, sr=sr), _tone(990, 0.11, sr=sr), gap=0.02, sr=sr)
    elif earcon is Earcon.MUTED:             # single descending tone
        buf = _seq(_tone(520, 0.10, sr=sr), _tone(330, 0.16, sr=sr), sr=sr)
    elif earcon is Earcon.START_NOTES:       # distinct ascending chime
        buf = _seq(_tone(523, 0.08, sr=sr), _tone(659, 0.08, sr=sr), _tone(880, 0.12, sr=sr),
                   gap=0.015, sr=sr)
    elif earcon is Earcon.STOP_NOTES:        # short confirming tone
        buf = _seq(_tone(700, 0.10, sr=sr), _tone(560, 0.10, sr=sr), sr=sr)
    elif earcon is Earcon.SUMMARY_READY:     # soft chime
        buf = _seq(_tone(784, 0.10, sr=sr), _tone(1047, 0.14, sr=sr), gap=0.02, sr=sr)
    elif earcon is Earcon.WORKING:           # subtle soft tick
        buf = _tone(440, 0.04, sr=sr) * 0.5
    elif earcon is Earcon.ERROR:             # distinct low buzz
        buf = _seq(_tone(160, 0.12, sr=sr), _tone(160, 0.12, sr=sr), gap=0.04, sr=sr)
    else:  # pragma: no cover - exhaustive enum
        buf = np.zeros(0, dtype=np.float32)
    return (buf * float(volume)).astype(np.float32)
