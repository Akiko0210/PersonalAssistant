"""Tests for backchannel tolerance: filler words must not derail a reply.

The guarantee under test: when a barge-in turns out to be nothing but listener
filler ("yeah", "uh-huh", "oh okay"), the agent resumes the interrupted reply
where it left off — no model call, no lost answer. Real speech (a question, a
"stop", anything substantive) still takes the floor as before.
"""

import unittest

from voice_agent import is_backchannel
from tests.agent_fixtures import audio_chunk, make_agent as _make_agent


def make_agent(*, heard, interrupted_remaining=None):
    return _make_agent(heard=[heard], utterances=[audio_chunk()],
                       interrupted_remaining=interrupted_remaining)


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
