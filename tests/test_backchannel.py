"""Tests for backchannel tolerance: filler words must not derail a reply.

The guarantee under test: when a barge-in turns out to be nothing but listener
filler ("yeah", "uh-huh", "oh okay"), the agent resumes the interrupted reply
where it left off — no model call, no lost answer. Real speech (a question, a
"stop", anything substantive) still takes the floor as before.
"""

import logging
import queue
import threading
import unittest

import numpy as np

from voice_agent import Agent, is_backchannel


class TestIsBackchannel(unittest.TestCase):
    def test_common_fillers(self):
        for t in ("Yeah.", "yeah", "Uh-huh.", "Mm-hmm.", "Okay", "Oh, okay.",
                  "Aha", "Right", "Got it.", "yeah yeah", "Hmm.", "I see."):
            self.assertTrue(is_backchannel(t), t)

    def test_real_speech_is_not_filler(self):
        for t in ("Stop.", "Wait", "No.", "What about Tuesday?",
                  "Yeah but what about the second one",
                  "Okay now delete that note"):
            self.assertFalse(is_backchannel(t), t)

    def test_empty_is_not_filler(self):
        self.assertFalse(is_backchannel(""))
        self.assertFalse(is_backchannel("   "))
        self.assertFalse(is_backchannel("..."))


class FakeAudio:
    def __init__(self, utterances=None):
        self._utterances = list(utterances or [])
        self.flushed = 0

    def collect_utterance(self, interrupt=None, endpoint_ms=None):
        return self._utterances.pop(0) if self._utterances else None

    def flush(self):
        self.flushed += 1


class FakeSTT:
    def __init__(self, texts):
        self._texts = list(texts)

    def transcribe(self, audio):
        return self._texts.pop(0) if self._texts else ""


class FakeLLM:
    active = "alice"  # persona routing needs to know who is active

    def __init__(self):
        self.calls = []

    def converse(self, text):
        self.calls.append(text)
        return f"reply::{text}"

    def record_unanswered(self, text):
        pass

    def take_pending_note(self):
        return None

    def take_pending_switch(self):
        return None


def make_agent(*, heard, interrupted_remaining=None):
    agent = Agent.__new__(Agent)
    agent.log = logging.getLogger("test")
    agent.interrupt = threading.Event()
    agent.cmds = queue.Queue()
    agent.audio = FakeAudio(utterances=[np.ones(4, dtype=np.int16)])
    agent.stt = FakeSTT([heard])
    agent.llm = FakeLLM()
    agent._interrupted_reply = "the full reply"
    agent._interrupted_remaining = interrupted_remaining
    agent.spoken = []

    def fake_say(text, **kwargs):
        agent.spoken.append(text)
        return False  # finishes uninterrupted

    agent.say = fake_say
    return agent


class TestFillerResumesReply(unittest.TestCase):
    def test_filler_resumes_without_model_call(self):
        agent = make_agent(heard="Yeah.",
                           interrupted_remaining="rest of the reply")
        agent.run_conversation_turn()
        self.assertEqual(agent.spoken, ["rest of the reply"])  # picked back up
        self.assertEqual(agent.llm.calls, [])                  # nothing billed
        self.assertIsNone(agent._interrupted_remaining)        # state cleared

    def test_continue_still_resumes(self):
        agent = make_agent(heard="continue",
                           interrupted_remaining="rest of the reply")
        agent.run_conversation_turn()
        self.assertEqual(agent.spoken, ["rest of the reply"])
        self.assertEqual(agent.llm.calls, [])

    def test_real_speech_takes_the_floor(self):
        agent = make_agent(heard="What about the second trade?",
                           interrupted_remaining="rest of the reply")
        agent._converse_with_followups = lambda text: agent.llm.converse(text)
        agent.run_conversation_turn()
        # The reply is abandoned and the question is answered instead.
        self.assertEqual(agent.llm.calls, ["What about the second trade?"])
        self.assertEqual(agent.spoken, ["reply::What about the second trade?"])
        self.assertIsNone(agent._interrupted_remaining)

    def test_filler_with_nothing_to_resume_reaches_the_model(self):
        # "Yeah" as an ordinary turn (no interrupted reply pending) is a normal
        # utterance — it must still be answered, not swallowed.
        agent = make_agent(heard="Yeah.", interrupted_remaining=None)
        agent._converse_with_followups = lambda text: agent.llm.converse(text)
        agent.run_conversation_turn()
        self.assertEqual(agent.llm.calls, ["Yeah."])


if __name__ == "__main__":
    unittest.main()
