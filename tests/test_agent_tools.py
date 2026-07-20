"""Tests for the switch_agent tool and per-agent tool filtering."""

import unittest

import agents
from tools import ToolContext, api_tools, dispatch, _REGISTRY


class TestSwitchAgent(unittest.TestCase):
    def setUp(self):
        self.ctx = ToolContext(active_agent="alice")

    def test_registered(self):
        self.assertIn("switch_agent", _REGISTRY)

    def test_switch_sets_pending(self):
        out = dispatch(self.ctx, "switch_agent", {"agent": "bob"})
        self.assertEqual(self.ctx.pending_switch, ("bob", ""))
        self.assertIn("Bob", out)

    def test_forward_travels_with_the_switch(self):
        dispatch(self.ctx, "switch_agent",
                 {"agent": "cobe", "forward": "what came in on Discord today?"})
        self.assertEqual(self.ctx.pending_switch,
                         ("cobe", "what came in on Discord today?"))

    def test_alias_resolution(self):
        dispatch(self.ctx, "switch_agent", {"agent": "kobe"})
        self.assertEqual(self.ctx.pending_switch[0], "cobe")

    def test_self_switch_is_a_noop(self):
        out = dispatch(self.ctx, "switch_agent", {"agent": "alice"})
        self.assertIsNone(self.ctx.pending_switch)
        self.assertIn("already", out.lower())

    def test_unknown_agent_is_rejected(self):
        out = dispatch(self.ctx, "switch_agent", {"agent": "dave"})
        self.assertIsNone(self.ctx.pending_switch)
        self.assertIn("don't know", out)


class TestPerAgentToolFiltering(unittest.TestCase):
    def test_include_filters_to_allowlist(self):
        names = [t["name"] for t in api_tools(include={"get_current_time",
                                                       "switch_agent"})]
        self.assertEqual(sorted(names), ["get_current_time", "switch_agent"])

    def test_alice_cannot_touch_notes_and_cobe_cannot_save(self):
        alice = {t["name"] for t in api_tools(include=agents.AGENTS["alice"]["tools"])}
        cobe = {t["name"] for t in api_tools(include=agents.AGENTS["cobe"]["tools"])}
        bob = {t["name"] for t in api_tools(include=agents.AGENTS["bob"]["tools"])}
        self.assertNotIn("search_notes", alice)
        self.assertNotIn("save_conversation_note", cobe)
        self.assertIn("search_notes", bob)
        self.assertIn("search_knowledge", cobe)

    def test_exclude_still_works_alongside_include(self):
        names = [t["name"] for t in
                 api_tools(exclude={"switch_agent"},
                           include={"switch_agent", "get_current_time"})]
        self.assertEqual(names, ["get_current_time"])


if __name__ == "__main__":
    unittest.main()
