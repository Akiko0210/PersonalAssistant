"""Tests for earcon synthesis (§10) — shape, finiteness, and distinctness."""

from __future__ import annotations

import numpy as np

from voice_notes_agent.feedback.earcons import SAMPLE_RATE, Earcon, render


def test_every_earcon_renders_finite_audio():
    for ec in Earcon:
        buf = render(ec, volume=0.5)
        assert buf.dtype == np.float32
        assert buf.ndim == 1
        assert buf.size > 0
        assert np.all(np.isfinite(buf))
        assert float(np.max(np.abs(buf))) <= 0.5 + 1e-6  # respects volume ceiling


def test_volume_scales_amplitude():
    quiet = render(Earcon.LISTENING, volume=0.1)
    loud = render(Earcon.LISTENING, volume=0.8)
    assert np.max(np.abs(loud)) > np.max(np.abs(quiet))


def test_earcons_are_distinct():
    # Different events should not render to identical buffers (§10: identifiable by ear).
    rendered = {ec: render(ec, volume=0.5) for ec in Earcon}
    listening = rendered[Earcon.MUTED]
    start = rendered[Earcon.START_NOTES]
    assert listening.shape != start.shape or not np.array_equal(listening, start)


def test_sample_rate_constant_matches_default_render():
    buf = render(Earcon.WORKING, sr=SAMPLE_RATE)
    # WORKING is a short single tick — well under half a second.
    assert buf.size < SAMPLE_RATE // 2
