"""Single-instance lock (Windows).

Stops a second copy of the agent from running against the same data/ directory.
Two live instances would both capture the mic, both call Claude (double cost),
talk over each other, and — the real damage — race to write history.json and
the Chroma index, corrupting them.

Uses an `msvcrt` byte-range lock on a lock file. Windows releases the lock
automatically when the holding process exits, so a crash never leaves a stale
lock behind; the leftover lock *file* is harmless (the next start just re-locks
it). Nothing is written to the file — writing/resizing it after taking the lock
silently drops the lock on Windows, which would defeat the whole guard.

This project targets Windows (SAPI text-to-speech, SMTC media buttons). On any
other OS the lock is a no-op — the guard simply isn't enforced there.
"""

import os

_WINDOWS = os.name == "nt"
if _WINDOWS:
    import msvcrt


class AlreadyRunning(RuntimeError):
    """Raised by acquire() when another process already holds the lock."""


class SingleInstance:
    """Hold an exclusive lock on `path` for the life of this process.

    Use as a context manager (`with SingleInstance(path): ...`) or call
    acquire()/release() directly. acquire() raises AlreadyRunning if another
    live process holds the lock."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self._fh = None

    def acquire(self):
        if not _WINDOWS:
            return self  # guard not enforced off-Windows (app is Windows-only)
        fh = open(self.path, "a+")
        try:
            fh.seek(0)
            # Lock one byte; raises OSError (PermissionError) if another process
            # holds it. Locking beyond EOF is fine, so the file can stay empty —
            # and it MUST stay empty: any write/truncate would drop this lock.
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            fh.close()
            raise AlreadyRunning(
                f"another instance already holds {self.path}"
            ) from e
        self._fh = fh
        return self

    def release(self):
        if self._fh is None:
            return
        fh, self._fh = self._fh, None
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        fh.close()
        # Best-effort tidy-up; the lock is already gone, so a leftover file is
        # harmless either way.
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False
