"""Tests for the VAD speech segmenter (§5.2) using a fake probability function.

No microphone and no Silero model — speech is driven by a scripted probability sequence,
so these exercise the pre-roll / hangover / tail-pad / min-length logic and wall-clock
timestamping deterministically.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from voice_notes_agent.audio.vad import FRAME_SAMPLES, SUPPORTED_RATE, VadSegmenter

START = datetime(2026, 6, 19, 14, 0, 0)
FRAME_SEC = FRAME_SAMPLES / SUPPORTED_RATE  # 0.032 s


def _frame(value: float = 0.1) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.float32)


class ScriptedProb:
    """Yields a preset P(speech) per call, cycling the last value."""

    def __init__(self, seq: list[float]) -> None:
        self._seq = seq
        self._i = 0

    def __call__(self, frame: np.ndarray) -> float:
        v = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return v


def _run(probs: list[float], **kwargs) -> list:
    seg_out = []
    prob = ScriptedProb(probs)
    seg = VadSegmenter(prob, session_start=START, threshold=0.5, **kwargs)
    for _ in probs:
        seg_out.extend(seg.process(_frame()))
    tail = seg.flush()
    if tail is not None:
        seg_out.append(tail)
    return seg_out


def test_pure_silence_yields_no_segments():
    segs = _run([0.0] * 50, hangover_sec=FRAME_SEC * 2, min_segment_sec=0.0)
    assert segs == []


def test_single_speech_burst_closes_after_hangover():
    # 10 speech frames, then enough silence to exceed a 3-frame hangover.
    probs = [0.9] * 10 + [0.0] * 5
    segs = _run(
        probs,
        pre_roll_sec=FRAME_SEC * 2,
        hangover_sec=FRAME_SEC * 3,
        tail_pad_sec=0.0,
        min_segment_sec=0.0,
    )
    assert len(segs) == 1
    seg = segs[0]
    assert seg.index == 0
    assert seg.duration_sec > 0
    # Wall-clock start must be at/after the session start (FR-C6).
    assert seg.start_wallclock >= START
    assert seg.end_wallclock > seg.start_wallclock


def test_pre_roll_is_prepended():
    # 3 silent frames build the pre-roll, then speech. Pre-roll of 2 frames means the
    # segment should include 2 prepended silent frames before the speech onset.
    probs = [0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0]
    segs = _run(
        probs,
        pre_roll_sec=FRAME_SEC * 2,
        hangover_sec=FRAME_SEC * 3,
        tail_pad_sec=0.0,
        min_segment_sec=0.0,
    )
    assert len(segs) == 1
    # 2 pre-roll + 2 speech + hangover frames retained (hangover trimmed on close).
    # At minimum it must exceed the bare 2 speech frames thanks to pre-roll.
    assert segs[0].audio.shape[0] >= 4 * FRAME_SAMPLES


def test_min_segment_length_drops_noise():
    # One isolated speech frame, below a generous min-segment length -> dropped (FR-C5).
    probs = [0.0, 0.9, 0.0, 0.0, 0.0]
    segs = _run(
        probs,
        pre_roll_sec=0.0,
        hangover_sec=FRAME_SEC * 2,
        tail_pad_sec=0.0,
        min_segment_sec=FRAME_SEC * 5,
    )
    assert segs == []


def test_two_separate_bursts_make_two_segments():
    probs = [0.9, 0.9] + [0.0] * 4 + [0.9, 0.9] + [0.0] * 4
    segs = _run(
        probs,
        pre_roll_sec=0.0,
        hangover_sec=FRAME_SEC * 3,
        tail_pad_sec=0.0,
        min_segment_sec=0.0,
    )
    assert len(segs) == 2
    assert [s.index for s in segs] == [0, 1]
    # Second segment starts later in wall-clock than the first ends.
    assert segs[1].start_wallclock > segs[0].start_wallclock


def test_flush_emits_open_segment_at_end():
    # Speech that never closes (no trailing silence) must be emitted by flush().
    probs = [0.9] * 6
    segs = _run(probs, pre_roll_sec=0.0, hangover_sec=FRAME_SEC * 3, min_segment_sec=0.0)
    assert len(segs) == 1


def test_energy_floor_suppresses_quiet_speech():
    # High P(speech) but the frame RMS is below the energy floor -> treated as non-speech.
    prob = ScriptedProb([0.99] * 10)
    seg = VadSegmenter(
        prob,
        session_start=START,
        threshold=0.5,
        energy_floor=0.5,  # frames are 0.1 amplitude, well below
        min_segment_sec=0.0,
    )
    out = []
    for _ in range(10):
        out.extend(seg.process(_frame(0.1)))
    assert seg.flush() is None
    assert out == []


def test_wrong_frame_size_raises():
    seg = VadSegmenter(lambda f: 0.0, session_start=START)
    with pytest.raises(ValueError):
        list(seg.process(np.zeros(256, dtype=np.float32)))
