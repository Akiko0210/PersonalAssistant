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
