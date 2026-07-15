"""Crash / power-loss safe file writes.

Overwriting a file in place (open-truncate-write) leaves a window where a power
cut can leave it half-written, truncated, or empty — and for our JSON state
files that means the next boot fails to parse it and falls back to empty,
losing the data. Instead: write to a temp file in the same directory, flush it
to physical disk (fsync), then os.replace() it over the target. os.replace is an
atomic rename on the same volume, so at any instant of power loss you have
either the complete old file or the complete new file — never a torn one.
"""

import json
import os
import tempfile


def write_json_atomic(path, obj, *, indent=2, ensure_ascii=False):
    """Atomically write `obj` as JSON to `path`. Thin convenience over
    write_text_atomic for the common case (indent=2, ensure_ascii=False) — the
    shape almost every state file in this project uses."""
    write_text_atomic(path, json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii))


def write_text_atomic(path, text, encoding="utf-8"):
    """Atomically replace `path`'s contents with `text`. Raises on I/O failure
    (leaving the existing file untouched); callers that must not fail should
    catch, as before."""
    path = os.fspath(path)
    directory = os.path.dirname(path) or "."
    # Temp file in the SAME directory, so os.replace() is a same-volume atomic
    # rename rather than a cross-volume copy (which wouldn't be atomic).
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".swap")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())   # force to disk before the rename is exposed
        os.replace(tmp, path)      # atomic on the same filesystem
    except BaseException:
        # Any failure: don't leave the temp file behind, and leave the original
        # in place (we never touched it).
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
