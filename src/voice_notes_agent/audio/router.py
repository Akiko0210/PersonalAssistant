"""Shared microphone input stream with genuine hard mute (§5.7, §C7).

The router owns the single PortAudio input stream that both subsystems read from. Only
one subsystem consumes frames at a time (capture vs conversation are mutually exclusive,
§A4), so the router just exposes the raw frame callback and lets the app route it.

**Hard mute** (FR-M2) is the important property: muting *closes and releases* the input
stream so the OS shows the mic is no longer in use. It is not a software gain flag — when
MUTED, no PortAudio stream exists at all. Un-muting re-opens it.

Frames are delivered as 512-sample (32 ms) float32 mono blocks at 16 kHz to match Silero
and Whisper. ``blocksize`` is pinned to 512 so downstream code never has to re-chunk.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from .vad import FRAME_SAMPLES, SUPPORTED_RATE

FrameCallback = Callable[[np.ndarray], None]


class AudioRouter:
    """Manages the lifecycle of the shared mic input stream."""

    def __init__(self, *, sample_rate: int = SUPPORTED_RATE, device=None) -> None:
        if sample_rate != SUPPORTED_RATE:
            raise ValueError(f"router requires {SUPPORTED_RATE} Hz, got {sample_rate}")
        self._rate = sample_rate
        self._device = device
        self._lock = threading.RLock()
        self._stream = None
        self._sink: FrameCallback | None = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._stream is not None

    def set_sink(self, sink: FrameCallback | None) -> None:
        """Route incoming frames to ``sink`` (or ``None`` to drop them)."""
        with self._lock:
            self._sink = sink

    def open(self, sink: FrameCallback | None = None) -> None:
        """Open (or re-open) the input stream. Idempotent while open."""
        import sounddevice as sd  # pragma: no cover - hardware dependent

        with self._lock:
            if sink is not None:
                self._sink = sink
            if self._stream is not None:
                return

            def _callback(indata, frames, time_info, status):  # pragma: no cover
                if status:
                    # Overflows/underflows are logged by the app; keep the stream alive.
                    pass
                mono = indata[:, 0] if indata.ndim > 1 else indata
                cb = self._sink
                if cb is not None:
                    cb(np.ascontiguousarray(mono, dtype=np.float32))

            self._stream = sd.InputStream(  # pragma: no cover - hardware dependent
                samplerate=self._rate,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SAMPLES,
                device=self._device,
                callback=_callback,
            )
            self._stream.start()

    def close(self) -> None:
        """Close and release the stream — this is the *hard* part of hard mute (FR-M2)."""
        with self._lock:
            stream = self._stream
            self._stream = None
            self._sink = None
        if stream is not None:  # pragma: no cover - hardware dependent
            stream.stop()
            stream.close()
