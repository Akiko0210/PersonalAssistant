"""Tests for the settle-before-answering turn logic in voice_agent.

The guarantee under test: one spoken turn triggers exactly ONE llm.converse()
call, with any mid-thought continuations merged into it — no speculative call
per pause. Driven with fakes so no mic/model/hardware is needed. The Agent is
built via __new__ so its heavy __init__ (audio devices, models) is skipped; the
methods here only touch the few attributes set below.
"""

import logging
import threading
import unittest

import numpy as np

from voice_agent import Agent


def _audio(n=4):
    return np.ones(n, dtype=np.int16)


class FakeAudio:
    def __init__(self, poll_frames=None, utterances=None):
        self._poll = list(poll_frames or [])
        self._utterances = list(utterances or [])
        self.pushed = []

    def poll_speech(self, timeout=0.05, return_frame=False):
        return self._poll.pop(0) if self._poll else None

    def pushback(self, frames):
        self.pushed.append(list(frames))

    def collect_utterance(self, interrupt=None, endpoint_ms=None):
        return self._utterances.pop(0) if self._utterances else None


class FakeSTT:
    def __init__(self, texts):
        self._texts = list(texts)

    def transcribe(self, audio):
        return self._texts.pop(0) if self._texts else ""


class FakeLLM:
    def __init__(self):
        self.calls = []

    def converse(self, text):
        self.calls.append(text)
        return f"reply::{text}"


def make_agent(*, await_seq=None, stt=None, utterances=None, poll_frames=None):
    agent = Agent.__new__(Agent)
    agent.log = logging.getLogger("test")
    agent.interrupt = threading.Event()
    agent.audio = FakeAudio(poll_frames=poll_frames, utterances=utterances)
    agent.stt = FakeSTT(stt or [])
    agent.llm = FakeLLM()
    if await_seq is not None:
        answers = iter(await_seq)
        agent._await_continuation = lambda settle: next(answers)
    return agent


class TestOneConversePerTurn(unittest.TestCase):
    def test_no_continuation_single_call(self):
        agent = make_agent(await_seq=[False])
        reply = agent._converse_with_followups("hello")
        self.assertEqual(agent.llm.calls, ["hello"])
        self.assertEqual(reply, "reply::hello")

    def test_one_continuation_merges_into_single_call(self):
        agent = make_agent(await_seq=[True, False], stt=["and another thing"],
                           utterances=[_audio()])
        reply = agent._converse_with_followups("okay a question")
        self.assertEqual(agent.llm.calls, ["okay a question and another thing"])
        self.assertEqual(reply, "reply::okay a question and another thing")

    def test_multiple_continuations_all_merge_into_one_call(self):
        agent = make_agent(await_seq=[True, True, False],
                           stt=["part two", "part three"],
                           utterances=[_audio(), _audio()])
        agent._converse_with_followups("part one")
        self.assertEqual(agent.llm.calls, ["part one part two part three"])

    def test_cough_during_settle_costs_nothing(self):
        # settle triggered (True) but the chunk transcribes to nothing: no merge,
        # and — unlike the old speculative design — no wasted model call.
        agent = make_agent(await_seq=[True, False], stt=[""], utterances=[_audio()])
        agent._converse_with_followups("hello")
        self.assertEqual(agent.llm.calls, ["hello"])

    def test_hotkey_during_settle_aborts_without_calling_model(self):
        agent = make_agent(await_seq=[False])
        agent.interrupt.set()  # a mute/note/quit command landed
        reply = agent._converse_with_followups("hello")
        self.assertEqual(reply, "")
        self.assertEqual(agent.llm.calls, [])  # nothing billed


class TestAwaitContinuation(unittest.TestCase):
    def make(self, poll_frames=None):
        agent = Agent.__new__(Agent)
        agent.log = logging.getLogger("test")
        agent.interrupt = threading.Event()
        agent.audio = FakeAudio(poll_frames=poll_frames)
        return agent

    def test_returns_true_on_qualifying_speech(self):
        # pad ring holds 10 frames; >6 must qualify (is_speech + loud) to trigger
        frames = [(True, 300, b"x")] * 8
        agent = self.make(poll_frames=frames)
        self.assertTrue(agent._await_continuation(5000))
        self.assertTrue(agent.audio.pushed)  # frames handed back for capture

    def test_quiet_frames_time_out_to_false(self):
        # sub-threshold noise never triggers; the window elapses -> False
        frames = [(False, 10, b"q")] * 3
        agent = self.make(poll_frames=frames)
        self.assertFalse(agent._await_continuation(40))

    def test_interrupt_returns_false_immediately(self):
        agent = self.make(poll_frames=[(True, 300, b"x")] * 8)
        agent.interrupt.set()
        self.assertFalse(agent._await_continuation(5000))


if __name__ == "__main__":
    unittest.main()
