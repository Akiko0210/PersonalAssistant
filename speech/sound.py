"""Looping 'thinking' audio cue.

A single sound (e.g. a Minecraft theme) is looped while the agent is busy waiting
on the model — answering, summarising, or filing a note — so the dead air has an
audible "still working" signal. Playback uses the built-in winsound, which loops
and stops cleanly and needs no extra dependencies (WAV only).
"""

import logging
import threading
from pathlib import Path

import config as cfg

log = logging.getLogger("sound")


class IdleSound:
    """Loop a WAV while the agent thinks. ``start``/``stop`` are idempotent and
    thread-safe, so callers can bracket every model call without tracking state —
    overlapping or back-to-back thinking spans won't cut the loop. Never raises:
    a missing file or unavailable winsound just means silence, so audio trouble
    can't break the agent."""

    def __init__(self, path=None):
        self.path = cfg.IDLE_SOUND if path is None else path
        self._lock = threading.Lock()
        self._playing = False

    def start(self):
        with self._lock:
            if self._playing:
                return
            if not (self.path and Path(self.path).is_file()):
                return
            try:
                import winsound
                winsound.PlaySound(
                    str(self.path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
                )
                self._playing = True
            except Exception:  # noqa: BLE001 - audio must never break the agent
                log.exception("could not start idle sound")

    def stop(self):
        with self._lock:
            if not self._playing:
                return
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:  # noqa: BLE001
                log.exception("could not stop idle sound")
            finally:
                self._playing = False
