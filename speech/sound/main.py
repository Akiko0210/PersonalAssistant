"""Looping 'thinking' audio cue.

A single sound (e.g. a Minecraft theme) is looped while the agent is busy waiting
on the model — answering, summarising, or filing a note — so the dead air has an
audible "still working" signal. Playback itself lives in the variant files
beside this one: `windows.py` (winsound) and `posix.py` (sounddevice, shared by
macOS and Linux).
"""

import logging
import os
import threading
from pathlib import Path

import config as cfg

if os.name == "nt":
    from speech.sound import windows as _impl
else:
    from speech.sound import posix as _impl

log = logging.getLogger("sound")


class IdleSound:
    """Loop a WAV while the agent thinks. ``start``/``stop`` are idempotent and
    thread-safe, so callers can bracket every model call without tracking state —
    overlapping or back-to-back thinking spans won't cut the loop. Never raises:
    a missing file or unavailable audio backend just means silence, so audio
    trouble can't break the agent."""

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
                _impl.play_loop(self.path)
                self._playing = True
            except Exception:  # noqa: BLE001 - audio must never break the agent
                log.exception("could not start idle sound")

    def stop(self):
        with self._lock:
            if not self._playing:
                return
            try:
                _impl.stop()
            except Exception:  # noqa: BLE001
                log.exception("could not stop idle sound")
            finally:
                self._playing = False
