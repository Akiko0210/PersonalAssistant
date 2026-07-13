"""Unit tests for the cross-platform single-instance lock.

A second handle on the same lock file conflicts with the first on both backends
(fcntl.flock and msvcrt.locking), so the "second instance is refused" guarantee
is testable within one process — no subprocess or real agent needed."""

import os
import tempfile
import unittest

from single_instance import SingleInstance, AlreadyRunning


@unittest.skipUnless(os.name == "nt", "single-instance lock is Windows-only for now")
class TestSingleInstance(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "agent.lock")

    def tearDown(self):
        for p in (self.path, self.dir):
            try:
                os.unlink(p)
            except OSError:
                try:
                    os.rmdir(p)
                except OSError:
                    pass

    def test_second_acquire_is_refused(self):
        first = SingleInstance(self.path).acquire()
        try:
            with self.assertRaises(AlreadyRunning):
                SingleInstance(self.path).acquire()
        finally:
            first.release()

    def test_release_frees_the_lock(self):
        SingleInstance(self.path).acquire().release()
        # a fresh instance can now take it — no stale-lock lockout
        second = SingleInstance(self.path).acquire()
        second.release()

    def test_context_manager_holds_then_frees(self):
        with SingleInstance(self.path):
            with self.assertRaises(AlreadyRunning):
                SingleInstance(self.path).acquire()
        # released on exit -> acquirable again
        SingleInstance(self.path).acquire().release()

    def test_lock_file_is_created(self):
        inst = SingleInstance(self.path).acquire()
        try:
            self.assertTrue(os.path.exists(self.path))
        finally:
            inst.release()

    def test_release_is_idempotent(self):
        inst = SingleInstance(self.path)
        inst.acquire()
        inst.release()
        inst.release()  # must not raise


if __name__ == "__main__":
    unittest.main()
