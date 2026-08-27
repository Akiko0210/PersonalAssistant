"""Single-instance lock.

Stops a second copy of the agent from running against the same data/ directory.
Two live instances would both capture the mic, both call Claude (double cost),
talk over each other, and — the real damage — race to write history.json and
the Chroma index, corrupting them.

The OS-specific locking lives in the variant files beside this one —
`windows.py` (msvcrt byte-range lock) and `posix.py` (fcntl.flock) — both
released by the kernel when the holding process exits, so a crash never leaves
a stale lock behind; the leftover lock *file* is harmless (the next start just
re-locks it).
"""

import os

if os.name == "nt":
    from lib.single_instance import windows as _impl
else:
    from lib.single_instance import posix as _impl


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
        fh = open(self.path, "a+")
        try:
            _impl.lock(fh)
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
            _impl.unlock(fh)
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
