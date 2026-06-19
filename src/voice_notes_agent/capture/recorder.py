"""VAD-gated recorder: glue between mic frames, segmenter, session, and the worker (§5.2).

The recorder is the capture-mode frame sink. For each incoming mic frame it:

  1. runs the :class:`VadSegmenter`; when a segment closes,
  2. appends the segment's speech-only audio to the :class:`NoteSession` (crash-safe), and
  3. submits the segment to the background :class:`TranscriptionWorker`.

On stop it flushes a trailing open segment, drains the worker (so the transcript is
ready, NFR-3), and finalizes the session files. The cloud agent is never involved while
the recorder runs — capture is entirely local (§A4, C4).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

import numpy as np

from ..audio.vad import SpeechProbability, VadSegmenter, make_segmenter
from .session import NoteSession
from .transcriber import TranscriptionWorker

log = logging.getLogger(__name__)


class Recorder:
    """Owns one capture session's segmenter + worker for its lifetime."""

    def __init__(
        self,
        session: NoteSession,
        vad_cfg,
        prob_fn: SpeechProbability,
        worker: TranscriptionWorker,
    ) -> None:
        self._session = session
        self._worker = worker
        self._segmenter: VadSegmenter = make_segmenter(vad_cfg, session.started, prob_fn)
        self._running = False

    @property
    def session(self) -> NoteSession:
        return self._session

    def start(self) -> None:
        self._worker.start()
        self._running = True
        log.info("capture started: session %s", self._session.id)

    def on_frame(self, frame: np.ndarray) -> None:
        """Frame sink registered with the AudioRouter while in CAPTURING."""
        if not self._running:
            return
        for seg in self._segmenter.process(frame):
            self._ingest(seg)

    def _ingest(self, seg) -> None:
        self._session.append_segment_audio(seg)
        self._worker.submit(seg.index, seg.audio, seg.sample_rate)

    def stop(self) -> NoteSession:
        """Flush, drain transcription, finalize files; return the completed session."""
        self._running = False
        tail = self._segmenter.flush()
        if tail is not None:
            self._ingest(tail)
        self._worker.drain_and_stop()
        self._session.finalize()
        log.info(
            "capture stopped: session %s, %.1fs speech",
            self._session.id,
            self._session.total_speech_sec,
        )
        return self._session
