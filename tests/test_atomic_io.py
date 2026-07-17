"""Unit tests for atomic_io.write_text_atomic.

The safety property under test: a failure while replacing the file leaves the
existing contents intact (never truncated/empty) and leaves no temp file
behind. That's exactly the power-cut guarantee — os.replace() is atomic, so a
crash yields either the whole old file or the whole new one."""

import os
import tempfile
import unittest
from unittest import mock

from atomic_io import write_text_atomic, write_json_atomic


class TestAtomicWrite(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "state.json")

    def tearDown(self):
        for name in os.listdir(self.dir):
            try:
                os.unlink(os.path.join(self.dir, name))
            except OSError:
                pass
        try:
            os.rmdir(self.dir)
        except OSError:
            pass

    def _read(self):
        with open(self.path, encoding="utf-8") as f:
            return f.read()

    def _temp_leftovers(self):
        return [n for n in os.listdir(self.dir) if n.startswith(".tmp-")]

    def test_writes_content(self):
        write_text_atomic(self.path, "hello")
        self.assertEqual(self._read(), "hello")

    def test_overwrites_existing(self):
        write_text_atomic(self.path, "first")
        write_text_atomic(self.path, "second")
        self.assertEqual(self._read(), "second")

    def test_no_temp_file_left_behind(self):
        write_text_atomic(self.path, "clean")
        self.assertEqual(self._temp_leftovers(), [])

    def test_failure_preserves_original_and_cleans_up(self):
        write_text_atomic(self.path, "ORIGINAL")
        # Simulate a crash/power-loss at the rename step.
        with mock.patch("os.replace", side_effect=OSError("simulated power loss")):
            with self.assertRaises(OSError):
                write_text_atomic(self.path, "NEW-BUT-DOOMED")
        self.assertEqual(self._read(), "ORIGINAL")  # old file untouched
        self.assertEqual(self._temp_leftovers(), [])  # temp cleaned up

    def test_transient_sharing_violation_is_retried(self):
        # Windows: os.replace fails with PermissionError while another process
        # (Dropbox sync, AV scanner) briefly holds the destination open. Those
        # holds clear in milliseconds — the write must retry, not crash.
        real_replace = os.replace
        calls = {"n": 0}

        def flaky_replace(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError(13, "sharing violation", dst)
            return real_replace(src, dst)

        with mock.patch("atomic_io.os.replace", side_effect=flaky_replace):
            write_text_atomic(self.path, "survived the sync client")
        self.assertEqual(self._read(), "survived the sync client")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(self._temp_leftovers(), [])

    def test_persistent_permission_error_still_raises(self):
        write_text_atomic(self.path, "ORIGINAL")
        with mock.patch("atomic_io.os.replace",
                        side_effect=PermissionError(13, "held forever")), \
             mock.patch("atomic_io.time.sleep"):  # don't actually back off
            with self.assertRaises(PermissionError):
                write_text_atomic(self.path, "NEVER-LANDS")
        self.assertEqual(self._read(), "ORIGINAL")
        self.assertEqual(self._temp_leftovers(), [])

    def test_unicode_roundtrip(self):
        write_text_atomic(self.path, "café — spread — 日本語")
        self.assertEqual(self._read(), "café — spread — 日本語")

    def test_write_json_atomic_roundtrips(self):
        import json
        obj = {"a": 1, "notes": ["x", "y"], "unicode": "café"}
        write_json_atomic(self.path, obj)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), obj)
        self.assertEqual(self._temp_leftovers(), [])


if __name__ == "__main__":
    unittest.main()
