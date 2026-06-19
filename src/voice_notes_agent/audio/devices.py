"""Audio device helpers and earcon/TTS playback (§7).

Thin wrappers over ``sounddevice`` (PortAudio). Imported lazily by callers so the
package imports without PortAudio present.
"""

from __future__ import annotations

import numpy as np


def list_devices() -> str:  # pragma: no cover - hardware dependent
    import sounddevice as sd

    return str(sd.query_devices())


def resolve_device(spec) -> object:
    """Resolve a device spec (int index, name substring, or None=default)."""
    return spec  # sounddevice accepts index, name, or None directly


def play(audio: np.ndarray, sample_rate: int, *, device=None, blocking: bool = False) -> None:
    """Play a mono float32 buffer on the output device (earcons + TTS)."""
    import sounddevice as sd  # pragma: no cover - hardware dependent

    sd.play(audio, samplerate=sample_rate, device=device, blocking=blocking)


def stop_playback() -> None:  # pragma: no cover - hardware dependent
    import sounddevice as sd

    sd.stop()
