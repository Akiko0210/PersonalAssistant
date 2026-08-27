"""Windows lock variant: an `msvcrt` byte-range lock.

Windows releases the lock automatically when the holding process exits, so a
crash never leaves a stale lock behind. Nothing may ever be written to the
locked file — writing/resizing it after taking the lock silently drops the
lock on Windows, which would defeat the whole guard.
"""

import msvcrt


def lock(fh):
    fh.seek(0)
    # Lock one byte; raises OSError (PermissionError) if another process
    # holds it. Locking beyond EOF is fine, so the file can stay empty —
    # and it MUST stay empty: any write/truncate would drop this lock.
    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)


def unlock(fh):
    fh.seek(0)
    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
