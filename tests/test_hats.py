"""Persona ("hat") smoke tests against a fake client: switching agents must
change the model, the tool subset, and the system prompt — while the one
shared history keeps flowing through untouched."""

import unittest

from brain import agents
import config as cfg
from tests.llm_fixtures import make_claude


class TestHats(unittest.TestCase):
    def test_alice_defaults(self):
        c = make_claude()
        c.converse("hello")
        call = c.client.messages.calls[0]
        self.assertEqual(call["model"], cfg.CONVO_MODELS["haiku"])
        names = {t["name"] for t in call["tools"]}
        self.assertEqual(names, set(agents.AGENTS["alice"]["tools"]))
        self.assertIn("You are Alice", call["system"])

    def test_switch_to_tom_changes_model_tools_and_prompt(self):
        c = make_claude()
        c.switch_to("tom")
        c.converse("how did SPX trades go?")
        call = c.client.messages.calls[0]
        self.assertEqual(call["model"], cfg.CONVO_MODELS["sonnet"])
        names = {t["name"] for t in call["tools"]}
        self.assertIn("search_knowledge", names)
        self.assertNotIn("search_notes", names)
        # The trading tools must actually reach the API call — they existed
        # on feat/trading but were missing from this allowlist, so voice
        # trading was silently unreachable.
        self.assertIn("build_strategy", names)
        self.assertIn("submit_order", names)
        self.assertIn("You are Tom", call["system"])

    def test_shared_history_survives_switching(self):
        c = make_claude()
        c.converse("tell alice something")
        c.switch_to("bob")
        c.converse("bob, do you remember?")
        # One list, both turns present in the second call's messages.
        texts = str(c.client.messages.calls[1]["messages"])
        self.assertIn("tell alice something", texts)

    def test_model_override_sticks_to_its_hat(self):
        c = make_claude()
        c.switch_to("tom")
        c._ctx.convo_model = cfg.CONVO_MODELS["opus"]  # "make Tom smarter"
        c.switch_to("alice")
        self.assertEqual(c._ctx.convo_model, cfg.CONVO_MODELS["haiku"])
        c.switch_to("tom")
        self.assertEqual(c._ctx.convo_model, cfg.CONVO_MODELS["opus"])

    def test_active_model_reports_registry_default_then_override(self):
        c = make_claude()
        c.switch_to("tom")
        self.assertEqual(c.active_model, cfg.CONVO_MODELS["sonnet"])
        self.assertEqual(c.active_model_label, "Sonnet 5")
        c._ctx.convo_model = cfg.CONVO_MODELS["deepseek pro"]
        self.assertEqual(c.active_model_label, "DeepSeek V4 Pro")

    def test_each_hat_reports_its_own_remembered_model(self):
        # The switch announcement's whole point: leave Tom on DeepSeek, visit
        # Bob (Haiku), come back to Tom and still be told DeepSeek.
        c = make_claude()
        c.switch_to("tom")
        c._ctx.convo_model = cfg.CONVO_MODELS["deepseek pro"]
        c.switch_to("bob")
        self.assertEqual(c.active_model_label, "Haiku 4.5")
        c.switch_to("tom")
        self.assertEqual(c.active_model_label, "DeepSeek V4 Pro")

    def test_switch_leaves_a_who_took_over_marker_in_history(self):
        # The messages themselves carry no speaker identity, so without this
        # marker nothing in the history says Bob's turns weren't Alice's.
        c = make_claude()
        c.converse("tell alice something")
        c.switch_to("bob")
        c.converse("bob, do you remember?")
        texts = str(c.client.messages.calls[1]["messages"])
        self.assertIn("Bob took over the conversation here", texts)

    def test_switch_marker_keeps_roles_alternating(self):
        # It is appended as an assistant turn straight after an assistant
        # reply; sanitize must fold them together or the API rejects the turn.
        c = make_claude()
        c.converse("hello")
        c.switch_to("tom")
        roles = [m["role"] for m in c.history]
        self.assertEqual(roles, ["user", "assistant"])

    def test_no_marker_when_switching_to_the_active_hat(self):
        c = make_claude()
        c.switch_to(agents.DEFAULT_AGENT)
        self.assertEqual(c.history, [])

    def test_pending_switch_roundtrip(self):
        c = make_claude()
        c._ctx.pending_switch = ("bob", "what's my last note?")
        self.assertEqual(c.take_pending_switch(), ("bob", "what's my last note?"))
        self.assertIsNone(c.take_pending_switch())


if __name__ == "__main__":
    unittest.main()
