"""Failover with automatic switch-back (brain/llm/failover.py): the query
rewrite, fact extraction and the conversation each move to a standby when
their provider fails, announce it once, and move back on their own once the
provider answers again."""

import unittest

import anthropic
import httpx

import config as cfg
from brain.llm.failover import Failover
from tests.llm_fixtures import ScriptedMessages, make_claude, text_reply
from tests.test_balance_failover import broke
from tools import dispatch

HAIKU = cfg.CONVO_MODELS["haiku"]
FLASH = "deepseek-v4-flash"


def timeout():
    return anthropic.APITimeoutError(
        request=httpx.Request("POST", "https://api.example.com/v1/messages"))


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeClient:
    """A client plain.ask can bind a timeout to."""

    def __init__(self, responses):
        self.messages = ScriptedMessages(responses)

    def with_options(self, **kwargs):
        return self


class TestStateMachine(unittest.TestCase):
    def setUp(self):
        self.clock, self.said = Clock(), []
        self.fo = Failover("memory lookups", lambda m: HAIKU, self.said.append,
                           clock=self.clock)

    def test_switch_is_announced_once_and_return_once(self):
        self.assertEqual(self.fo.failed(FLASH, timeout()), HAIKU)
        self.assertEqual(self.said, ["DeepSeek isn't responding, so memory "
                                     "lookups switched to Anthropic."])
        self.assertEqual(self.fo.pick(FLASH), HAIKU)           # cooling down
        self.clock.now += cfg.FAILOVER_COOLDOWN_S
        self.assertEqual(self.fo.pick(FLASH), FLASH)           # probe the default
        self.fo.failed(FLASH, timeout())                       # still down:
        self.assertEqual(len(self.said), 1)                    # silent restart
        self.assertEqual(self.fo.pick(FLASH), HAIKU)
        self.fo.ok(HAIKU)                                      # the standby working
        self.assertEqual(len(self.said), 1)                    # is not news
        self.fo.ok(FLASH)
        self.assertEqual(self.said[-1], "DeepSeek is answering again, so memory "
                                        "lookups moved back to it.")

    def test_reasons_and_no_same_provider_standby(self):
        self.fo.failed(FLASH, broke())
        self.assertIn("has insufficient balance", self.said[0])
        same = Failover("x", lambda m: "deepseek-v4-pro", self.said.append)
        self.assertIsNone(same.failed(FLASH, timeout()))

    def test_clear_forgets_the_outage(self):
        self.fo.failed(FLASH, timeout())
        self.fo.clear()
        self.assertEqual(self.fo.pick(FLASH), FLASH)

    def test_wrap_retries_on_the_standby_then_skips_the_default(self):
        calls = []

        def make_ask(model):
            def ask(system, prompt):
                calls.append(model)
                if model == FLASH:
                    raise timeout()
                return "answer"
            return ask
        call = self.fo.wrap(make_ask, FLASH)
        self.assertEqual(call("s", "p"), "answer")
        self.assertEqual(call("s", "p"), "answer")
        self.assertEqual(calls, [FLASH, HAIKU, HAIKU])         # no 8 s wait twice


class TestThroughClaude(unittest.TestCase):
    def test_a_rewrite_timeout_uses_haiku_and_says_so_before_the_reply(self):
        c = make_claude()
        c.client = FakeClient([text_reply('{"query": "hi"}'), text_reply("answer")])
        c._deepseek = FakeClient([timeout()])
        del c._rewriter  # the fixture's stub; exercise the real one
        reply = c.converse("hi")
        self.assertEqual(reply, "DeepSeek isn't responding, so memory lookups "
                                "switched to Anthropic. answer")
        self.assertEqual(c.client.messages.calls[0]["model"], cfg.QUERY_FALLBACK_MODEL)

    def test_an_extraction_failure_is_spoken_on_the_next_turn(self):
        c = make_claude([text_reply("next answer")])
        c.client = FakeClient([text_reply('{"ops": []}'), text_reply("next answer")])
        c._deepseek = FakeClient([RuntimeError("'NoneType' object is not iterable")])
        del c._extractor
        self.assertEqual(c._extractor()("system", "prompt"), '{"ops": []}')
        self.assertEqual(c.client.messages.calls[0]["model"], cfg.FACTS_FALLBACK_MODEL)
        self.assertEqual(c.converse("hi"), "DeepSeek is failing, so memory saving "
                                           "switched to Anthropic. next answer")

    def _claude(self, primary, secondary):
        c = make_claude(primary, convo_model=HAIKU)
        c._deepseek = FakeClient(secondary)
        self.clock = Clock()
        c._setup_failover(clock=self.clock)
        c._ctx.failover = c._convo_failover
        return c

    def test_the_conversation_moves_back_after_the_cooldown(self):
        c = self._claude([broke(), text_reply("on Claude again")],
                         [text_reply("on DeepSeek")])
        self.assertIn("switched to DeepSeek", c.converse("hi"))
        self.assertEqual(c.active_model, cfg.CONVO_MODELS["deepseek"])
        self.clock.now += cfg.FAILOVER_COOLDOWN_S
        self.assertEqual(c.converse("hi again"),
                         "Anthropic is answering again, so the conversation "
                         "moved back to it. on Claude again")
        self.assertEqual(c.active_model, HAIKU)

    def test_choosing_a_model_by_hand_clears_the_outage(self):
        c = self._claude([broke()], [text_reply("ok")])
        c.converse("hi")
        dispatch(c._ctx, "set_conversation_model", {"model": "haiku"})
        self.assertEqual(c.active_model, HAIKU)

    def test_a_failover_inside_a_delegation_is_announced_by_the_next_reply(self):
        c = self._claude([broke()], [text_reply("done"), text_reply("reply")])
        c.run_delegated_task("bob", "look something up")
        self.assertEqual(c.converse("hi"), "Anthropic has insufficient balance, "
                                           "so the conversation switched to "
                                           "DeepSeek. reply")


if __name__ == "__main__":
    unittest.main()
