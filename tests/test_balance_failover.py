"""An empty balance on one provider moves the conversation to the other and
says so — instead of the generic "I hit an error" (2026-09-29 15:50 log)."""

import unittest
from types import SimpleNamespace

import anthropic
import httpx

import config as cfg
from tests.llm_fixtures import ScriptedMessages, make_claude, text_reply

DEEPSEEK = cfg.CONVO_MODELS["deepseek"]
HAIKU = cfg.CONVO_MODELS["haiku"]


def status_error(status, message, cls=anthropic.APIStatusError):
    request = httpx.Request("POST", "https://api.example.com/v1/messages")
    return cls(message, response=httpx.Response(status, request=request),
               body={"error": {"message": message}})


def broke():
    return status_error(402, "Insufficient Balance")


def claude_with(primary, secondary, model):
    c = make_claude(primary, convo_model=model)
    c._deepseek = SimpleNamespace(messages=ScriptedMessages(secondary))
    return c


class TestBalanceFailover(unittest.TestCase):
    def test_anthropic_empty_moves_to_deepseek_and_announces(self):
        c = claude_with([broke()], [text_reply("It's three o'clock.")], HAIKU)
        reply = c.converse("what time is it")
        self.assertIn("Anthropic has insufficient balance", reply)
        self.assertIn("switched to DeepSeek", reply)
        self.assertTrue(reply.endswith("It's three o'clock."))
        self.assertEqual(c._ctx.convo_model, DEEPSEEK)   # sticky, not per-call
        self.assertIsNone(c._ctx.failover_notice)         # said once

    def test_deepseek_empty_moves_to_anthropic(self):
        c = claude_with([text_reply("Back on Claude.")], [broke()], DEEPSEEK)
        reply = c.converse("hello")
        self.assertIn("DeepSeek has insufficient balance", reply)
        self.assertIn("switched to Anthropic", reply)
        self.assertEqual(c._ctx.convo_model, HAIKU)

    def test_anthropic_credit_wording_also_counts(self):
        # Anthropic's own shape: a 400, not a 402.
        e = status_error(400, "Your credit balance is too low to access the "
                              "Anthropic API.", anthropic.BadRequestError)
        c = claude_with([e], [text_reply("ok")], HAIKU)
        self.assertIn("switched to DeepSeek", c.converse("hi"))

    def test_both_empty_raises_instead_of_ping_ponging(self):
        c = claude_with([broke()], [broke()], HAIKU)
        with self.assertRaises(anthropic.APIStatusError):
            c.converse("hi")
        self.assertEqual(len(c.client.messages.calls), 1)
        self.assertEqual(len(c._deepseek.messages.calls), 1)

    def test_other_errors_do_not_fail_over(self):
        c = claude_with([status_error(500, "boom")], [text_reply()], HAIKU)
        with self.assertRaises(anthropic.APIStatusError):
            c.converse("hi")
        self.assertEqual(c._ctx.convo_model, HAIKU)
        self.assertEqual(c._deepseek.messages.calls, [])


if __name__ == "__main__":
    unittest.main()
