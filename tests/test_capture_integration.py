"""End-to-end capture path: recorder -> session -> background worker -> disk (§5.2-5.3, §9).

Drives the real :class:`Recorder`, :class:`NoteSession`, segmenter, and
:class:`TranscriptionWorker` with a scripted VAD probability and a fake transcriber, so
no microphone, Silero, or Whisper is needed. Asserts the crash-safe artifacts land on
disk with transcribed text and a finalized manifest.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from voice_notes_agent.audio.vad import FRAME_SAMPLES
from voice_notes_agent.capture.recorder import Recorder
from voice_notes_agent.capture.session import NoteSession
from voice_notes_agent.capture.transcriber import TranscriptionWorker
from voice_notes_agent.config import VadConfig
from voice_notes_agent.paths import Paths

FRAME_SEC = FRAME_SAMPLES / 16000


def _frame(v: float = 0.2) -> np.ndarray:
    return np.full(FRAME_SAMPLES, v, dtype=np.float32)


class ScriptedProb:
    def __init__(self, seq):
        self._seq = seq
        self._i = 0

    def __call__(self, frame):
        v = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return v


def test_full_capture_produces_transcript(tmp_path):
    paths = Paths.resolve(tmp_path)
    session = NoteSession.create(paths, sample_rate=16000)

    # Speech for 8 frames, then silence long enough to close a 3-frame hangover.
    probs = [0.9] * 8 + [0.0] * 6
    vad_cfg = VadConfig(
        threshold=0.5,
        pre_roll_sec=0.0,
        hangover_sec=FRAME_SEC * 3,
        tail_pad_sec=0.0,
        min_segment_sec=0.0,
    )

    transcribed = {}

    def fake_transcribe(audio, sr):
        return "hello from the meeting"

    worker = TranscriptionWorker(fake_transcribe, on_text=session.set_segment_text)
    recorder = Recorder(session, vad_cfg, ScriptedProb(probs), worker)
    recorder.start()
    for _ in probs:
        recorder.on_frame(_frame())
    finished = recorder.stop()

    # Transcript JSON exists with the transcribed text (FR-T4).
    data = json.loads((finished.dir / "transcript.json").read_text(encoding="utf-8"))
    assert data["segments"], "expected at least one segment"
    assert any("hello from the meeting" in s["text"] for s in data["segments"])

    # Human-readable transcript + speech audio written (§9).
    assert (finished.dir / "transcript.txt").read_text(encoding="utf-8").strip()
    assert (finished.dir / "speech.flac").exists()

    # Manifest is finalized (crash recovery would skip it, R-8).
    manifest = json.loads((finished.dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "finalized"
    assert manifest["ended"] is not None

    # Wall-clock timestamps present and ordered (FR-C6).
    seg0 = data["segments"][0]
    assert datetime.fromisoformat(seg0["end_wallclock"]) >= datetime.fromisoformat(
        seg0["start_wallclock"]
    )
