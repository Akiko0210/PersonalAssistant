"""Background local transcription worker (§5.3, FR-T1/T2/T3/T5).

faster-whisper runs on a dedicated worker thread fed by a queue. Segments are
transcribed **during** the session as they close, so the full transcript is ready within
a couple of seconds of stop with no end-of-session wait (NFR-3).

Anti-hallucination measures (FR-T3, R-2):
  * ``condition_on_previous_text=False`` — each segment is independent
  * ``no_speech_threshold`` / ``log_prob`` filtering — drop low-confidence noise output
  * VAD has already gated out silence, so the model rarely sees near-silence

If throughput can't keep up, the worker falls back to a smaller model (FR-T5).

The model is wrapped behind a ``Transcribe`` protocol so the worker is testable with a
fake transcriber (no model download).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger(__name__)


class Transcribe(Protocol):
    def __call__(self, audio: np.ndarray, sample_rate: int) -> str: ...


class FasterWhisperTranscriber:
    """Loads a faster-whisper model and transcribes a float32 buffer to text."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._model = None
        self._model_name = cfg.model

    def _ensure_model(self, name: str) -> None:  # pragma: no cover - heavy dep
        from faster_whisper import WhisperModel

        if self._model is None or self._model_name != name:
            log.info("loading faster-whisper model: %s (%s)", name, self._cfg.compute_type)
            self._model = WhisperModel(name, device="cpu", compute_type=self._cfg.compute_type)
            self._model_name = name

    def __call__(self, audio: np.ndarray, sample_rate: int) -> str:  # pragma: no cover
        self._ensure_model(self._model_name)
        segments, _info = self._model.transcribe(
            audio,
            language=self._cfg.language,
            beam_size=self._cfg.beam_size,
            condition_on_previous_text=self._cfg.condition_on_previous_text,
            no_speech_threshold=self._cfg.no_speech_threshold,
            log_prob_threshold=self._cfg.log_prob_threshold,
            vad_filter=False,  # our segmenter already did VAD gating
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def use_fallback(self) -> None:  # pragma: no cover
        if self._model_name != self._cfg.fallback_model:
            log.warning("falling back to smaller Whisper model: %s", self._cfg.fallback_model)
            self._model = None
            self._model_name = self._cfg.fallback_model


# A unit of work: (segment_index, float32 audio, sample_rate).
_Job = tuple[int, np.ndarray, int]


class TranscriptionWorker:
    """Single background thread that drains a queue of segments through the model."""

    def __init__(
        self,
        transcribe: Transcribe,
        on_text: Callable[[int, str], None],
        *,
        fallback: Callable[[], None] | None = None,
        slow_factor: float = 1.0,
    ) -> None:
        self._transcribe = transcribe
        self._on_text = on_text
        self._fallback = fallback
        self._slow_factor = slow_factor  # if processing > slow_factor*audio_dur, fall back
        self._q: "queue.Queue[_Job | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._fell_back = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="whisper-worker", daemon=True)
        self._thread.start()

    def submit(self, index: int, audio: np.ndarray, sample_rate: int) -> None:
        self._q.put((index, audio, sample_rate))

    def drain_and_stop(self, timeout: float | None = None) -> None:
        """Block until all queued segments are transcribed, then stop the worker."""
        self._q.put(None)  # sentinel
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        import time

        while True:
            job = self._q.get()
            if job is None:
                return
            index, audio, sr = job
            try:
                t0 = time.monotonic()
                text = self._transcribe(audio, sr)
                elapsed = time.monotonic() - t0
                audio_dur = len(audio) / sr
                # Throughput watchdog (FR-T5): if we can't keep up, drop to a smaller model.
                if (
                    not self._fell_back
                    and self._fallback is not None
                    and audio_dur > 0
                    and elapsed > self._slow_factor * audio_dur
                ):
                    self._fallback()
                    self._fell_back = True
                self._on_text(index, text)
            except Exception:  # pragma: no cover - model/runtime errors
                log.exception("transcription failed for segment %d", index)
                self._on_text(index, "")
