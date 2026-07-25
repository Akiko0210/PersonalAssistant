"""Mute stops the microphone, not the reply.

A headset click pauses playback the instant it lands (the dongle only
transmits the gesture's next click if the host really stops — see
MEDIA_CONTROL.md), and the reply then waits for the multi-click window to say
what the gesture was: mute resumes it, note-taking and quit end it. These
tests drive that verdict logic with fakes — no audio device, no SAPI, no
timers.
"""

import logging
import queue
import threading
import unittest
from unittest import mock

import config as cfg
from voice_agent import Agent


class FakeTTS:
    """Records the playback verbs in order, so a test can assert the reply was
    paused and resumed rather than discarded."""

    supports_async = True

    def __init__(self, busy_polls=3, can_pause=True, on_pause=None,
                 on_poll=None, click_at=1):
        self.events = []          # begin / pause / resume / stop, in order
        self.spoken = []
        self._left = 0
        self._busy_polls = busy_polls
        self._polls = 0
        self.can_pause = can_pause
        self.on_pause = on_pause  # fires while paused: the gesture resolving
        self.on_poll = on_poll    # fires mid-reply: a click landing
        self.click_at = click_at
        self.stopped = False

    def begin(self, text):
        self.spoken.append(text)
        self.events.append("begin")
        self._left = self._busy_polls

    def is_busy(self):
        if self.stopped:
            return False
        self._polls += 1
        if self.on_poll and self._polls == self.click_at:
            self.on_poll()
        if self._left > 0:
            self._left -= 1
            return True
        return False

    def pause(self):
        self.events.append("pause")
        if not self.can_pause:
            return False
        if self.on_pause:
            self.on_pause()
        return True

    def resume(self):
        self.events.append("resume")

    def stop(self):
        self.events.append("stop")
        self.stopped = True


class FakeIdle:
    def stop(self):
        pass


class FakeAudio:
    def __init__(self):
        self.muted = threading.Event()
        self.pushed = []

    def poll_speech(self, timeout=0.1, return_frame=False):
        return None  # a quiet room: nothing ever barges in

    def pushback(self, frames):
        self.pushed.append(list(frames))


def make_agent(**tts_kwargs):
    agent = Agent.__new__(Agent)
    agent.log = logging.getLogger("test")
    agent.audio = FakeAudio()
    agent.idle = FakeIdle()
    agent.cmds = queue.Queue()
    agent.interrupt = threading.Event()
    agent.hush = threading.Event()
    agent.silence = threading.Event()
    agent.resume_speech = threading.Event()
    agent.status = "conversation_mode"
    agent.tts = FakeTTS(**tts_kwargs)
    return agent


class TestClickWhileSpeaking(unittest.TestCase):
    """say() must pause on the click, then act on the resolved gesture."""

    def setUp(self):
        # Pin the tunables the path reads, so a local config override can't
        # change what these tests exercise.
        p1 = mock.patch.object(cfg, "BARGE_IN", True)
        p2 = mock.patch.object(cfg, "GESTURE_VERDICT_TIMEOUT_S", 0.2)
        p1.start(), p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def test_mute_click_lets_the_reply_finish(self):
        agent = make_agent()
        # A click lands mid-reply; the gesture behind it resolves to mute
        # while playback is paused.
        agent.tts.on_poll = agent.hush.set
        agent.tts.on_pause = agent.resume_speech.set

        interrupted = agent.say("the whole answer")

        self.assertFalse(interrupted)
        self.assertEqual(agent.tts.events, ["begin", "pause", "resume"])
        self.assertNotIn("stop", agent.tts.events)  # nothing was discarded

    def test_note_gesture_ends_the_reply(self):
        agent = make_agent()
        agent.tts.on_poll = agent.hush.set
        agent.tts.on_pause = agent.silence.set

        interrupted = agent.say("the whole answer")

        self.assertFalse(interrupted)
        self.assertEqual(agent.tts.events, ["begin", "pause", "stop"])

    def test_reply_resumes_when_no_gesture_follows_the_click(self):
        # A stray/swallowed press with no gesture behind it must never leave
        # the reply stuck mid-word: the verdict times out and speech resumes.
        agent = make_agent()
        agent.tts.on_poll = agent.hush.set

        interrupted = agent.say("the whole answer")

        self.assertFalse(interrupted)
        self.assertEqual(agent.tts.events, ["begin", "pause", "resume"])

    def test_backend_that_cannot_pause_falls_back_to_stopping(self):
        # pyttsx3 (or a SAPI voice that refuses to pause) keeps the old
        # all-or-nothing behaviour rather than talking over a gesture.
        agent = make_agent(can_pause=False)
        agent.tts.on_poll = agent.hush.set
        agent.tts.on_pause = agent.resume_speech.set

        interrupted = agent.say("the whole answer")

        self.assertFalse(interrupted)
        self.assertEqual(agent.tts.events, ["begin", "pause", "stop"])

    def test_firmware_decoded_gesture_stops_the_reply_without_a_click(self):
        # AirPods decode multi-press themselves: Next/Previous arrive already
        # resolved, with no raw click (so no hush) behind them.
        agent = make_agent()
        agent.tts.on_poll = agent.silence.set

        interrupted = agent.say("the whole answer")

        self.assertFalse(interrupted)
        self.assertEqual(agent.tts.events, ["begin", "stop"])

    def test_uninterrupted_reply_is_never_paused(self):
        agent = make_agent()
        interrupted = agent.say("the whole answer")
        self.assertFalse(interrupted)
        self.assertEqual(agent.tts.events, ["begin"])

    def test_commands_off_ignores_the_click(self):
        # The folder-destination question opts out of action commands.
        agent = make_agent()
        agent.tts.on_poll = agent.hush.set
        agent.say("which folder?", commands=False)
        self.assertEqual(agent.tts.events, ["begin"])

    def test_a_stale_verdict_cannot_release_the_next_reply(self):
        # resume_speech left set by a gesture that resolved while nothing was
        # being said must not pre-authorise the *next* reply's first pause.
        agent = make_agent()
        agent.resume_speech.set()
        agent.tts.on_poll = agent.hush.set
        agent.tts.on_pause = agent.silence.set

        agent.say("the whole answer")

        self.assertEqual(agent.tts.events, ["begin", "pause", "stop"])


class TestMuteTakesEffectImmediately(unittest.TestCase):
    """The microphone must go deaf when the gesture resolves — while the reply
    is still playing — not when the main loop next drains."""

    def test_toggle_mute_applies_at_once_and_queues_the_announcement(self):
        agent = make_agent()
        agent._toggle_mute()

        self.assertTrue(agent.audio.muted.is_set())
        self.assertTrue(agent.interrupt.is_set())
        self.assertEqual(agent.cmds.get_nowait(), "announce_mute")

    def test_drain_announces_without_toggling_a_second_time(self):
        agent = make_agent()
        spoken = []
        agent.say = lambda text, **kw: spoken.append(text)

        agent._toggle_mute()
        signals = agent._drain()

        self.assertTrue(agent.audio.muted.is_set())  # still muted, not flipped
        self.assertEqual(spoken, ["Muted."])
        self.assertEqual(signals, set())
        self.assertFalse(agent.interrupt.is_set())

    def test_unmute_round_trip(self):
        agent = make_agent()
        spoken = []
        agent.say = lambda text, **kw: spoken.append(text)

        agent._toggle_mute()
        agent._drain()
        agent._toggle_mute()
        agent._drain()

        self.assertFalse(agent.audio.muted.is_set())
        self.assertEqual(spoken, ["Muted.", "Listening."])

    def test_rapid_toggles_announce_the_final_state_once(self):
        agent = make_agent()
        spoken = []
        agent.say = lambda text, **kw: spoken.append(text)

        agent._toggle_mute()   # muted
        agent._toggle_mute()   # and straight back, before the loop drained
        agent._drain()

        self.assertFalse(agent.audio.muted.is_set())
        self.assertEqual(spoken, ["Listening."])

    def test_mute_gesture_releases_a_paused_reply(self):
        agent = make_agent()
        agent._on_media_gesture(1)
        self.assertTrue(agent.resume_speech.is_set())
        self.assertFalse(agent.silence.is_set())


class TestSilencingGestures(unittest.TestCase):
    def test_note_gesture_marks_silence(self):
        agent = make_agent()
        agent._toggle_note()
        self.assertTrue(agent.silence.is_set())
        self.assertEqual(agent.cmds.get_nowait(), "start_note")

    def test_quit_marks_silence(self):
        agent = make_agent()
        agent._quit()
        self.assertTrue(agent.silence.is_set())
        self.assertEqual(agent.cmds.get_nowait(), "quit")

    def test_drain_clears_silence_for_the_next_reply(self):
        agent = make_agent()
        agent.say = lambda text, **kw: None
        agent._quit()
        signals = agent._drain()
        self.assertEqual(signals, {"quit"})
        self.assertFalse(agent.silence.is_set())

    def test_notetaking_unmutes(self):
        agent = make_agent()
        agent.audio.muted.set()
        agent._toggle_note()
        self.assertFalse(agent.audio.muted.is_set())


if __name__ == "__main__":
    unittest.main()
