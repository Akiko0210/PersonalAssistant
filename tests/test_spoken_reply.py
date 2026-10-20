"""Memory holds what the user heard, not what the model wrote (issue #26).

A reply cut off by a barge-in, or held unspoken because the user was already
talking, used to stay in history — and go to the exchange index and the
fact store — in full; the agent then argued from words never said aloud
("I just said you're welcome", 2026-10-09). Under test: say() records the
spoken part on every exit, settle_reply keeps only that part (marked) and
only then remembers the turn, and a "continue" resume extends it.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import config as cfg
from speech.tts import main as tts
from tests.agent_fixtures import FakeAudio, audio_chunk, make_agent
from tests.llm_fixtures import make_claude, text_reply, tool_reply
from tests.test_mute_speech import make_agent as make_speaking_agent

TEXT = "one two three four five six"


class TestHowFarSpeechGot(unittest.TestCase):
    def test_the_estimate_counts_whole_words(self):
        # 120 wpm = 2 words a second: 1.4 s is 2.8 words — "three" is cut.
        self.assertEqual(TEXT[:tts.estimate_chars(TEXT, 1.4, 120)], "one two ")
        self.assertEqual(tts.estimate_chars(TEXT, 60, 120), len(TEXT))

    def speaker(self, backend):
        with mock.patch.object(tts, "_select_backend", lambda: (lambda: backend,)):
            return tts.Speaker()

    def test_the_engines_own_position_wins(self):
        sp = self.speaker(SimpleNamespace(begin=lambda t: None, spoken_chars=lambda: 8))
        sp.begin(TEXT)
        self.assertEqual(sp.spoken_chars(), 8)

    def test_the_estimate_leaves_out_paused_time(self):
        # A mute click can hold a reply paused for seconds; none of that
        # time was speech.
        sp = self.speaker(SimpleNamespace(begin=lambda t: None, pause=lambda: None,
                                          resume=lambda: None))
        sp.set_voice(None, 120)
        clock = iter([0.0, 1.0, 6.0, 7.0])  # begin, pause, resume, read
        with mock.patch.object(tts.time, "monotonic", lambda: next(clock)):
            sp.begin(TEXT)
            sp.pause()
            sp.resume()
            at = sp.spoken_chars()
        self.assertEqual(TEXT[:at], "one two three four ")  # 2 s at 2 words/s


class TestSayRecordsWhatWasHeard(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(cfg, "BARGE_IN", True)
        patch.start()
        self.addCleanup(patch.stop)

    def test_a_finished_reply_was_heard_whole(self):
        agent = make_speaking_agent()
        agent.say(TEXT, save_resume=True)
        self.assertEqual(agent.last_heard, TEXT)

    def test_a_held_reply_was_heard_not_at_all(self):
        agent = make_speaking_agent()
        agent.audio = FakeAudio(poll_frames=[(True, 300, b"w")] * 8)  # already talking
        self.assertTrue(agent.say(TEXT, save_resume=True))
        self.assertEqual(agent.last_heard, "")
        self.assertEqual(agent._interrupted_remaining, TEXT)  # "okay" says it all

    def test_a_cut_reply_was_heard_up_to_the_word_in_progress(self):
        agent = make_speaking_agent()
        agent.tts.at = TEXT.index("five")       # the voice is on "five"
        agent.typed.put(("a typed message", None))  # the user takes the floor
        agent.say(TEXT, save_resume=True)
        self.assertEqual(agent.last_heard, "one two three four")
        # "continue" picks up two words back, for context.
        self.assertEqual(agent._interrupted_remaining, "three four five six")


class TestSettleReply(unittest.TestCase):
    def claude(self, *responses):
        c = make_claude(list(responses))
        self.indexed = []
        c.memory = SimpleNamespace(
            query_rows=c.memory.query_rows,
            index_exchanges=lambda msgs, owner, skip_existing=True:
                self.indexed.append(msgs[-1]["content"][-1]["text"]) or 0,
            exchanges_of=lambda *a, **k: [])
        return c

    def last_text(self, c):
        return c.history[-1]["content"][-1]["text"]

    def test_history_and_memory_keep_only_the_heard_part(self):
        c = self.claude(text_reply("The short expired. The long is open."))
        reply = c.converse("how's the diagonal?")
        self.assertEqual(self.indexed, [])  # nothing before the reply is spoken
        c.settle_reply(reply, "The short expired.")
        heard = f"The short expired. {cfg.REPLY_CUT_MARK}"
        self.assertEqual(self.last_text(c), heard)
        self.assertEqual(self.indexed, [heard])
        self.assertEqual(c.saved[-1][-1]["content"][-1]["text"], heard)  # persisted

    def test_a_reply_never_spoken_is_marked_unheard(self):
        c = self.claude(text_reply("Here is the plan."))
        c.settle_reply(c.converse("what's the plan?"), "")
        self.assertEqual(self.last_text(c), cfg.REPLY_UNHEARD_MARK)

    def test_a_reply_heard_whole_is_unchanged(self):
        c = self.claude(text_reply("Done."))
        c.settle_reply(c.converse("log it"), "Done.")
        self.assertEqual(self.last_text(c), "Done.")
        self.assertEqual(self.indexed, ["Done."])

    def test_a_failover_notice_heard_alone_leaves_the_answer_unheard(self):
        # The notice is spoken ahead of the answer but never in history.
        c = self.claude(text_reply("SPX is at 7,710."))
        c._notices.put("DeepSeek is failing, so memory saving switched to Anthropic.")
        reply = c.converse("where's SPX?")
        c.settle_reply(reply, reply[:reply.index("SPX is")].rstrip())
        self.assertEqual(self.last_text(c), cfg.REPLY_UNHEARD_MARK)

    def test_tool_calls_survive_the_rewrite(self):
        c = self.claude(tool_reply("get_current_time", {}), text_reply("It's noon. Lunch?"))
        c.settle_reply(c.converse("time?"), "It's noon.")
        kinds = [b.get("type") for m in c.history if isinstance(m["content"], list)
                 for b in m["content"]]
        self.assertIn("tool_use", kinds)
        self.assertIn("tool_result", kinds)

    def test_a_resume_that_finishes_restores_the_whole_reply(self):
        c = self.claude(text_reply(TEXT))
        reply = c.converse("count")
        c.settle_reply(reply, "one two")
        c.settle_reply(reply, TEXT)  # "continue" said the rest
        self.assertEqual(self.last_text(c), TEXT)
        self.assertEqual(len(self.indexed), 2)  # the same record, rewritten

    def test_an_unreported_turn_is_remembered_whole_at_the_next(self):
        c = self.claude(text_reply("first"), text_reply("second"))
        c.converse("a")
        c.converse("b")  # a script never settles: the first turn still counts
        self.assertEqual(self.indexed, ["first"])


class TestResumeExtendsWhatWasHeard(unittest.TestCase):
    def test_continue_reports_the_whole_reply_heard(self):
        agent = make_agent(heard=["continue"], utterances=[audio_chunk()],
                           interrupted_remaining="three four five six")
        agent._interrupted_reply = TEXT
        agent._resume_from = TEXT.index("three")
        agent._interrupted_heard = len("one two three four")
        agent.run_conversation_turn()
        self.assertEqual(agent.spoken, ["three four five six"])
        self.assertEqual(agent.llm.settled, [(TEXT, TEXT)])


if __name__ == "__main__":
    unittest.main()
