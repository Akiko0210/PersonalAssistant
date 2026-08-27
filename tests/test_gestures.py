"""Unit tests for gestures.ClickGestureDecoder (real timers, short windows)."""

import time
import unittest

from media_control.gestures import ClickGestureDecoder


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

    def test_stale_timer_callback_is_ignored(self):
        # A Timer whose callback fired but hasn't run yet can't be cancel()ed;
        # if a new click has re-armed the window since, the stale callback must
        # do nothing — not consume the new click's count or orphan its timer.
        d = self.make()
        d.click()                 # arms generation 1
        stale_gen = d._gen
        time.sleep(0.2)           # let gesture 1 resolve normally
        d.click()                 # arms generation 2
        d._resolve(stale_gen)     # simulate gen-1's late callback arriving now
        self.assertEqual(self.gestures, [1])  # gen 1 resolved once, not twice
        self.settle()
        self.assertEqual(self.gestures, [1, 1])  # gen 2 still resolves on time


class TestDarwinMediaKeySuppression(unittest.TestCase):
    """The pure decide-to-swallow half of the macOS event tap (the Quartz
    half is untestable off-device). data1 packs the keycode in the high 16
    bits; 0x0A00/0x0B00 in the low word are the down/up halves."""

    def test_play_pause_is_suppressed_down_and_up(self):
        from media_control.macos import is_agent_media_key, NX_KEYTYPE_PLAY
        self.assertTrue(is_agent_media_key(8, (NX_KEYTYPE_PLAY << 16) | 0x0A00))
        self.assertTrue(is_agent_media_key(8, (NX_KEYTYPE_PLAY << 16) | 0x0B00))

    def test_next_and_previous_are_suppressed(self):
        from media_control.macos import (is_agent_media_key,
                                         NX_KEYTYPE_NEXT, NX_KEYTYPE_PREVIOUS)
        self.assertTrue(is_agent_media_key(8, (NX_KEYTYPE_NEXT << 16) | 0x0A00))
        self.assertTrue(is_agent_media_key(8, (NX_KEYTYPE_PREVIOUS << 16) | 0x0A00))

    def test_unhandled_media_keys_pass_through(self):
        from media_control.macos import is_agent_media_key
        # volume up (keycode 0) — swallowing it would mute the user's Mac
        self.assertFalse(is_agent_media_key(8, (0 << 16) | 0x0A00))

    def test_other_subtypes_pass_through(self):
        from media_control.macos import is_agent_media_key, NX_KEYTYPE_PLAY
        self.assertFalse(is_agent_media_key(7, (NX_KEYTYPE_PLAY << 16) | 0x0A00))


if __name__ == "__main__":
    unittest.main()
