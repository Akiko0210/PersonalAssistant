"""Note-capture session: on-disk model + crash-safe incremental writes (§9, FR-C7, R-8).

A session owns one directory under ``sessions/`` (§9) and writes incrementally as speech
arrives, so a crash, power loss, or stray mute never loses captured audio or text
(NFR-5). Specifically:

  * ``speech.flac``     — appended segment-by-segment via a streaming soundfile writer
  * ``transcript.json`` — rewritten atomically after each segment is transcribed
  * ``manifest.json``   — small status file so a partial session can be recovered on
                          restart (§FR resilience, R-8)

Wall-clock timestamps come straight from the :class:`SpeechSegment` (FR-C6).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..audio.vad import SpeechSegment


@dataclass
class TranscriptSegment:
    index: int
    start_wallclock: str  # ISO 8601
    end_wallclock: str
    text: str = ""        # filled in once the background transcriber finishes


def _atomic_write_json(path: Path, payload) -> None:
    """Write JSON atomically (temp file + os.replace) so readers never see half a file."""
    tmp = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    )
    try:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    finally:
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


class NoteSession:
    """One note-taking session. Thread-safe; the recorder and transcriber both touch it."""

    def __init__(self, directory: Path, *, session_id: str, started: datetime, sample_rate: int):
        self.id = session_id
        self.dir = directory
        self.started = started
        self.sample_rate = sample_rate
        self.ended: datetime | None = None

        self._lock = threading.RLock()
        self._segments: dict[int, TranscriptSegment] = {}
        self._total_speech_samples = 0
        self._audio_writer = None  # lazy soundfile.SoundFile

        self._flac_path = directory / "speech.flac"
        self._json_path = directory / "transcript.json"
        self._txt_path = directory / "transcript.txt"
        self._manifest_path = directory / "manifest.json"
        self._write_manifest(status="recording")

    # -- factory --------------------------------------------------------------
    @classmethod
    def create(cls, paths, *, sample_rate: int) -> "NoteSession":
        session_id = uuid.uuid4().hex[:12]
        started = datetime.now().astimezone()
        directory = paths.session_dir(session_id, started)
        return cls(directory, session_id=session_id, started=started, sample_rate=sample_rate)

    # -- audio append (crash-safe, FR-C7) -------------------------------------
    def append_segment_audio(self, seg: SpeechSegment) -> None:
        """Append a segment's speech audio to ``speech.flac`` and register the segment."""
        with self._lock:
            if self._audio_writer is None:
                import soundfile as sf  # lazy

                self._audio_writer = sf.SoundFile(
                    str(self._flac_path),
                    mode="w",
                    samplerate=self.sample_rate,
                    channels=1,
                    format="FLAC",
                )
            self._audio_writer.write(seg.audio)
            self._audio_writer.flush()  # flush so partial sessions survive a crash
            self._total_speech_samples += len(seg.audio)
            self._segments[seg.index] = TranscriptSegment(
                index=seg.index,
                start_wallclock=seg.start_wallclock.isoformat(),
                end_wallclock=seg.end_wallclock.isoformat(),
            )
            self._flush_transcript_locked()

    # -- transcript text (filled by the background worker) --------------------
    def set_segment_text(self, index: int, text: str) -> None:
        with self._lock:
            seg = self._segments.get(index)
            if seg is None:
                return
            seg.text = text.strip()
            self._flush_transcript_locked()

    @property
    def total_speech_sec(self) -> float:
        with self._lock:
            return self._total_speech_samples / self.sample_rate

    def pending_indices(self) -> list[int]:
        """Segment indices that have audio but no transcript text yet."""
        with self._lock:
            return [i for i, s in self._segments.items() if not s.text]

    def transcript_text(self) -> str:
        with self._lock:
            return "\n".join(
                s.text for s in sorted(self._segments.values(), key=lambda x: x.index) if s.text
            )

    def segments_sorted(self) -> list[TranscriptSegment]:
        with self._lock:
            return [self._segments[i] for i in sorted(self._segments)]

    # -- finalize -------------------------------------------------------------
    def finalize(self) -> None:
        """Close the audio writer and flush final transcript + manifest (FR-T4)."""
        with self._lock:
            self.ended = datetime.now().astimezone()
            if self._audio_writer is not None:
                self._audio_writer.close()
                self._audio_writer = None
            self._flush_transcript_locked()
            self._write_manifest(status="finalized")

    # -- internals ------------------------------------------------------------
    def _flush_transcript_locked(self) -> None:
        segs = [asdict(self._segments[i]) for i in sorted(self._segments)]
        _atomic_write_json(self._json_path, {"session_id": self.id, "segments": segs})
        # Human-readable companion (transcript.txt, §9).
        lines = []
        for s in segs:
            if s["text"]:
                ts = s["start_wallclock"][11:19]  # HH:MM:SS
                lines.append(f"[{ts}] {s['text']}")
        tmp = self._txt_path.with_suffix(".txt.tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, self._txt_path)

    def _write_manifest(self, *, status: str) -> None:
        _atomic_write_json(
            self._manifest_path,
            {
                "session_id": self.id,
                "started": self.started.isoformat(),
                "ended": self.ended.isoformat() if self.ended else None,
                "sample_rate": self.sample_rate,
                "status": status,
            },
        )


def find_unfinalized_sessions(paths) -> list[Path]:
    """Return session dirs whose manifest is still 'recording' (crash recovery, R-8)."""
    out: list[Path] = []
    for d in sorted(paths.sessions.glob("*")):
        manifest = d / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") == "recording":
            out.append(d)
    return out
