"""Silero VAD and the speech segmenter (§5.2, §6/C6).

The segmenter is the mechanism that makes long, mostly-silent sessions tractable on a
CPU: only detected speech is ever emitted, so Whisper only sees ~minutes of audio out of
hours of wall-clock time (§NFR-2). It implements the four capture parameters from §5.2:

  * pre-roll  (FR-C3): a rolling buffer prepended to each segment so onsets aren't clipped
  * hangover  (FR-C4): trailing silence that closes a segment
  * tail pad  (FR-C4): extra audio kept after close so trailing words aren't clipped
  * min length(FR-C5): segments shorter than this are dropped as likely noise

Each emitted segment carries **wall-clock** start/end (FR-C6) computed from a monotonic
sample counter and the session start time, so "what did I note around 3pm" resolves
correctly despite silence removal.

``SpeechProbability`` is a small protocol so the segmenter can be unit-tested with a fake
probability function — no model, no microphone (see tests/test_segmenter.py).
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterator, Protocol

import numpy as np

# Silero operates on fixed 512-sample frames at 16 kHz (32 ms). The segmenter assumes
# frames of this size; the audio router is configured to deliver them.
FRAME_SAMPLES = 512
SUPPORTED_RATE = 16000


class SpeechProbability(Protocol):
    """Returns P(speech) in [0, 1] for a single 512-sample float32 frame."""

    def __call__(self, frame: np.ndarray) -> float: ...


@dataclass
class SpeechSegment:
    """One contiguous speech segment with wall-clock bounds and its float32 audio."""

    index: int
    start_wallclock: datetime
    end_wallclock: datetime
    audio: np.ndarray  # mono float32 @ SUPPORTED_RATE, pre-roll + speech + tail pad
    sample_rate: int = SUPPORTED_RATE

    @property
    def duration_sec(self) -> float:
        return len(self.audio) / self.sample_rate


@dataclass
class _SegmenterParams:
    threshold: float
    pre_roll_frames: int
    hangover_frames: int
    tail_pad_frames: int
    min_segment_frames: int
    energy_floor: float


class VadSegmenter:
    """Streaming speech segmenter driven frame-by-frame.

    Feed it 512-sample frames via :meth:`process`; it yields a :class:`SpeechSegment`
    whenever a segment closes. Call :meth:`flush` at session end to emit any segment
    still open.
    """

    def __init__(
        self,
        prob_fn: SpeechProbability,
        *,
        session_start: datetime,
        sample_rate: int = SUPPORTED_RATE,
        threshold: float = 0.5,
        pre_roll_sec: float = 0.5,
        hangover_sec: float = 1.0,
        tail_pad_sec: float = 0.2,
        min_segment_sec: float = 0.4,
        energy_floor: float = 0.0,
    ) -> None:
        if sample_rate != SUPPORTED_RATE:
            raise ValueError(f"VAD requires {SUPPORTED_RATE} Hz audio, got {sample_rate}")
        self._prob = prob_fn
        self._rate = sample_rate
        self._session_start = session_start

        frame_sec = FRAME_SAMPLES / sample_rate
        self._p = _SegmenterParams(
            threshold=threshold,
            pre_roll_frames=max(0, round(pre_roll_sec / frame_sec)),
            hangover_frames=max(1, round(hangover_sec / frame_sec)),
            tail_pad_frames=max(0, round(tail_pad_sec / frame_sec)),
            min_segment_frames=max(1, round(min_segment_sec / frame_sec)),
            energy_floor=energy_floor,
        )

        # Rolling pre-roll buffer of recent silent frames (FR-C3).
        self._preroll: collections.deque[np.ndarray] = collections.deque(
            maxlen=self._p.pre_roll_frames or 1
        )
        self._in_speech = False
        self._silence_run = 0          # consecutive non-speech frames inside a segment
        self._cur_frames: list[np.ndarray] = []
        self._cur_start_sample = 0     # sample index at segment audio start (incl pre-roll)
        self._tail_left = 0            # tail-pad frames still being collected after close
        self._sample_cursor = 0        # total samples seen — drives wall-clock (FR-C6)
        self._seg_index = 0

    # -- timing helpers -------------------------------------------------------
    def _wallclock(self, sample_index: int) -> datetime:
        return self._session_start + timedelta(seconds=sample_index / self._rate)

    def _is_speech(self, frame: np.ndarray) -> bool:
        if self._p.energy_floor > 0.0:
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
            if rms < self._p.energy_floor:
                return False
        return self._prob(frame) >= self._p.threshold

    # -- main entry point -----------------------------------------------------
    def process(self, frame: np.ndarray) -> Iterator[SpeechSegment]:
        """Feed one 512-sample float32 frame; yield 0 or 1 closed segments."""
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"expected {FRAME_SAMPLES}-sample frame, got {frame.shape[0]}")
        frame = frame.astype(np.float32, copy=False)
        frame_start_sample = self._sample_cursor
        self._sample_cursor += FRAME_SAMPLES

        speech = self._is_speech(frame)

        if not self._in_speech:
            if speech:
                # Open a segment, prepending the pre-roll buffer.
                pre = list(self._preroll)
                self._cur_frames = pre + [frame]
                self._cur_start_sample = frame_start_sample - len(pre) * FRAME_SAMPLES
                self._in_speech = True
                self._silence_run = 0
            else:
                self._preroll.append(frame)
            return iter(())

        # Inside a segment.
        self._cur_frames.append(frame)
        if speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._p.hangover_frames:
                # Close: keep tail-pad frames already accumulated within the hangover.
                seg = self._close_segment()
                if seg is not None:
                    yield seg

    def _close_segment(self) -> SpeechSegment | None:
        frames = self._cur_frames
        self._in_speech = False
        self._silence_run = 0
        self._cur_frames = []
        # Reset pre-roll from the tail of this segment so it primes the next onset.
        self._preroll.clear()

        # Trim hangover but retain tail pad (FR-C4).
        keep = max(0, len(frames) - self._p.hangover_frames + self._p.tail_pad_frames)
        frames = frames[:keep] if keep < len(frames) else frames

        # Drop sub-minimum segments as noise (FR-C5).
        if len(frames) < self._p.min_segment_frames:
            return None

        audio = np.concatenate(frames).astype(np.float32, copy=False)
        start = max(0, self._cur_start_sample)
        seg = SpeechSegment(
            index=self._seg_index,
            start_wallclock=self._wallclock(start),
            end_wallclock=self._wallclock(start + len(audio)),
            audio=audio,
            sample_rate=self._rate,
        )
        self._seg_index += 1
        return seg

    def flush(self) -> SpeechSegment | None:
        """Close any in-progress segment at session end."""
        if self._in_speech and self._cur_frames:
            # Treat remaining frames as the tail; no hangover to trim.
            frames = self._cur_frames
            self._in_speech = False
            self._cur_frames = []
            if len(frames) < self._p.min_segment_frames:
                return None
            audio = np.concatenate(frames).astype(np.float32, copy=False)
            start = max(0, self._cur_start_sample)
            seg = SpeechSegment(
                index=self._seg_index,
                start_wallclock=self._wallclock(start),
                end_wallclock=self._wallclock(start + len(audio)),
                audio=audio,
                sample_rate=self._rate,
            )
            self._seg_index += 1
            return seg
        return None


# -- Real Silero-backed probability function -------------------------------------
def load_silero_probability() -> SpeechProbability:
    """Load Silero VAD and return a per-frame probability callable.

    Imported lazily so the package imports (and unit tests run) without torch/silero
    installed. Raises a clear error if the dependency is missing.
    """
    try:
        import torch  # noqa: F401
        from silero_vad import load_silero_vad
    except Exception as exc:  # pragma: no cover - depends on optional heavy deps
        raise RuntimeError(
            "Silero VAD unavailable. Install with `pip install silero-vad torch`."
        ) from exc

    model = load_silero_vad()  # pragma: no cover - requires model download

    def prob(frame: np.ndarray) -> float:  # pragma: no cover
        import torch as _t

        with _t.no_grad():
            tensor = _t.from_numpy(frame).float()
            return float(model(tensor, SUPPORTED_RATE).item())

    return prob


def make_segmenter(cfg, session_start: datetime, prob_fn: SpeechProbability) -> VadSegmenter:
    """Construct a :class:`VadSegmenter` from a :class:`~voice_notes_agent.config.VadConfig`."""
    return VadSegmenter(
        prob_fn,
        session_start=session_start,
        threshold=cfg.threshold,
        pre_roll_sec=cfg.pre_roll_sec,
        hangover_sec=cfg.hangover_sec,
        tail_pad_sec=cfg.tail_pad_sec,
        min_segment_sec=cfg.min_segment_sec,
        energy_floor=cfg.energy_floor,
    )
