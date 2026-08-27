"""POSIX lock variant (macOS + Linux share it): `fcntl.flock`.

The kernel drops the lock when the holding process exits, so a crash never
leaves a stale lock behind — the same property the Windows variant gets from
msvcrt.
"""

import fcntl


def lock(fh):
    # flock is tied to the open file description, so a second open() conflicts
    # even within the same process — which is what makes the guard testable
    # without subprocesses.
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock(fh):
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
