"""Storage layout for the local data store (§9).

    .voice-notes-agent/
    ├── sessions/
    │   └── <date>_<time>_<session_id>/
    │       ├── speech.flac        # concatenated speech-only audio (small)
    │       ├── transcript.json    # segments: {start_wallclock, end_wallclock, text}
    │       ├── transcript.txt     # human-readable
    │       └── summary.md         # full summary
    ├── chroma/                    # local vector index
    ├── config.yaml                # tunables (§14)
    └── logs/

Set VOICE_NOTES_HOME to override the project-local default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

APP_DIR_NAME = ".voice-notes-agent"


def app_data_root() -> Path:
    """Return the local app-data root, creating nothing."""
    override = os.environ.get("VOICE_NOTES_HOME")
    if override:
        return Path(override).expanduser()
    return Path.cwd() / APP_DIR_NAME


@dataclass(frozen=True)
class Paths:
    """Resolved, ensured-to-exist directories for the data store."""

    root: Path
    sessions: Path
    chroma: Path
    logs: Path
    config_file: Path

    @classmethod
    def resolve(cls, root: Path | None = None, *, ensure: bool = True) -> "Paths":
        root = root or app_data_root()
        paths = cls(
            root=root,
            sessions=root / "sessions",
            chroma=root / "chroma",
            logs=root / "logs",
            config_file=root / "config.yaml",
        )
        if ensure:
            for d in (paths.root, paths.sessions, paths.chroma, paths.logs):
                d.mkdir(parents=True, exist_ok=True)
        return paths

    def session_dir(self, session_id: str, started: datetime) -> Path:
        """Directory for one session, named ``<date>_<time>_<session_id>`` (§9)."""
        stamp = started.strftime("%Y-%m-%d_%H-%M-%S")
        d = self.sessions / f"{stamp}_{session_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d
