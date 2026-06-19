"""Per-session file storage and listing (§5.5, §9).

Sessions already write ``transcript.json``/``transcript.txt``/``speech.flac`` themselves
(see capture/session.py). This module handles the summary file and read-side queries
that back the ``list_sessions`` / ``get_session_summary`` tools (§8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass
class SessionInfo:
    session_id: str
    date: str           # YYYY-MM-DD
    started: str        # ISO 8601
    title: str
    dir: Path

    @property
    def summary_path(self) -> Path:
        return self.dir / "summary.md"

    @property
    def transcript_path(self) -> Path:
        return self.dir / "transcript.txt"


class Store:
    """Read/write access to the ``sessions/`` tree."""

    def __init__(self, paths) -> None:
        self._paths = paths

    def save_summary(self, session_dir: Path, markdown: str, *, title: str) -> Path:
        """Write ``summary.md`` and stamp the title into the manifest."""
        path = session_dir / "summary.md"
        path.write_text(markdown, encoding="utf-8")
        manifest = session_dir / "manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            data["title"] = title
            manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def _read_manifest(self, session_dir: Path) -> dict:
        manifest = session_dir / "manifest.json"
        if manifest.exists():
            try:
                return json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def info_for(self, session_dir: Path) -> SessionInfo | None:
        data = self._read_manifest(session_dir)
        if not data:
            return None
        started = data.get("started", "")
        return SessionInfo(
            session_id=data.get("session_id", session_dir.name),
            date=started[:10] if started else "",
            started=started,
            title=data.get("title", "Untitled note"),
            dir=session_dir,
        )

    def list_sessions(
        self, date_from: date | None = None, date_to: date | None = None
    ) -> list[SessionInfo]:
        """List sessions, optionally filtered by date range (FR-R5)."""
        out: list[SessionInfo] = []
        for d in sorted(self._paths.sessions.glob("*"), reverse=True):
            if not d.is_dir():
                continue
            info = self.info_for(d)
            if info is None or not info.date:
                continue
            try:
                day = datetime.fromisoformat(info.started).date()
            except ValueError:
                continue
            if date_from and day < date_from:
                continue
            if date_to and day > date_to:
                continue
            out.append(info)
        return out

    def find(self, session_id: str) -> SessionInfo | None:
        for d in self._paths.sessions.glob(f"*_{session_id}"):
            return self.info_for(d)
        # Fall back to scanning manifests (handles renamed dirs).
        for d in self._paths.sessions.glob("*"):
            info = self.info_for(d)
            if info and info.session_id == session_id:
                return info
        return None

    def get_summary(self, session_id: str) -> tuple[str, Path] | None:
        info = self.find(session_id)
        if info is None or not info.summary_path.exists():
            return None
        return info.summary_path.read_text(encoding="utf-8"), info.summary_path
