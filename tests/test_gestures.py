"""Unit tests for gestures.ClickGestureDecoder (real timers, short windows)."""

import time
import unittest

from gestures import ClickGestureDecoder


class TestGestures(unittest.TestCase):
    def make(self, dedupe_s=0.03, window_s=0.12):
        self.gestures = []
        self.presses = 0

        def on_press():
            self.presses += 1

        return ClickGestureDecoder(self.gestures.append, on_press,
                                   dedupe_s=dedupe_s, window_s=window_s)

    def settle(self, window_s=0.12):
        time.sleep(window_s + 0.1)

    def test_single_click(self):
        d = self.make()
        d.click()
        self.settle()
        self.assertEqual(self.gestures, [1])
        self.assertEqual(self.presses, 1)

    def test_double_and_triple(self):
        d = self.make()
        d.click(); time.sleep(0.06); d.click()
        self.settle()
        d.click(); time.sleep(0.06); d.click(); time.sleep(0.06); d.click()
        self.settle()
        self.assertEqual(self.gestures, [2, 3])

    def test_duplicate_events_deduped(self):
        # one physical press surfacing on both channels within the dedupe window
        d = self.make()
        d.click(); d.click()
        self.settle()
        self.assertEqual(self.gestures, [1])
        self.assertEqual(self.presses, 1)  # on_press fires once per accepted click

    def test_stop_cancels_pending_gesture(self):
        d = self.make()
        d.click()
        d.stop()
        self.settle()
        self.assertEqual(self.gestures, [])


if __name__ == "__main__":
    unittest.main()
