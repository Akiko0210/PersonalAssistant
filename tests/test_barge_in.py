"""Unit tests for barge_in.BargeInDetector using synthetic frames.

Frame timing here: frame_ms=30, calib 90ms (3 frames), fire at 90ms
(3 qualifying frames), pad 90ms (3-frame pre-roll)."""

import unittest

from barge_in import BargeInDetector


def make_detector(**over):
    kw = dict(energy_floor=100, energy_ratio=3.0, calib_ms=90,
              fire_ms=90, decay=0.4, frame_ms=30, pad_ms=90)
    kw.update(over)
    return BargeInDetector(**kw)


class TestBargeIn(unittest.TestCase):
    def calibrate(self, d, rms=50, n=3):
        for i in range(n):
            self.assertFalse(d.feed(False, rms, f"c{i}".encode()))

    def test_no_fire_during_calibration(self):
        d = make_detector()
        # even loud voiced frames can't fire while calibrating
        self.assertFalse(d.feed(True, 9999, b"x"))
        self.assertFalse(d.feed(True, 9999, b"y"))
        self.assertFalse(d.feed(True, 9999, b"z"))
        self.assertFalse(d.calibrating)

    def test_threshold_scales_with_echo(self):
        d = make_detector()
        self.calibrate(d, rms=200)  # baseline 200 * ratio 3 = 600 > floor
        self.assertEqual(d.threshold, 600.0)

    def test_threshold_floor_wins_when_quiet(self):
        d = make_detector()
        self.calibrate(d, rms=10)  # 10*3=30 < floor 100
        self.assertEqual(d.threshold, 100.0)

    def test_fires_after_sustained_loud_speech(self):
        d = make_detector()
        self.calibrate(d)
        self.assertFalse(d.feed(True, 500, b"1"))
        self.assertFalse(d.feed(True, 500, b"2"))
        self.assertTrue(d.feed(True, 500, b"3"))
        self.assertIsNotNone(d.run)  # retained frames to push back

    def test_echo_level_speech_does_not_fire(self):
        d = make_detector()
        self.calibrate(d, rms=200)  # threshold 600
        for _ in range(20):
            self.assertFalse(d.feed(True, 300, b"e"))  # voiced but below threshold

    def test_leaky_counter_survives_brief_dropout(self):
        d = make_detector()
        self.calibrate(d)
        self.assertFalse(d.feed(True, 500, b"1"))    # 30ms
        self.assertFalse(d.feed(True, 500, b"2"))    # 60ms
        self.assertFalse(d.feed(False, 10, b"gap"))  # decays 0.4*30=12 -> 48ms (not reset)
        self.assertFalse(d.feed(True, 500, b"3"))    # 78ms
        self.assertTrue(d.feed(True, 500, b"4"))     # 108ms >= 90 -> fires
        # a hard-reset counter would have needed 3 fresh frames after the gap

    def test_false_start_fizzles_and_run_clears(self):
        d = make_detector()
        self.calibrate(d)
        self.assertFalse(d.feed(True, 500, b"1"))  # 30ms
        for _ in range(5):
            self.assertFalse(d.feed(False, 10, b"q"))  # decays to 0
        self.assertIsNone(d.run)


if __name__ == "__main__":
    unittest.main()
