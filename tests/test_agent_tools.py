"""Tests for the switch_agent / ask_agent tools and per-agent tool filtering."""

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
                 {"agent": "tom", "forward": "what came in on Discord today?"})
        self.assertEqual(self.ctx.pending_switch,
                         ("tom", "what came in on Discord today?"))

    def test_alias_resolution(self):
        dispatch(self.ctx, "switch_agent", {"agent": "thom"})
        self.assertEqual(self.ctx.pending_switch[0], "tom")

    def test_self_switch_is_a_noop(self):
        out = dispatch(self.ctx, "switch_agent", {"agent": "alice"})
        self.assertIsNone(self.ctx.pending_switch)
        self.assertIn("already", out.lower())

    def test_unknown_agent_is_rejected(self):
        out = dispatch(self.ctx, "switch_agent", {"agent": "dave"})
        self.assertIsNone(self.ctx.pending_switch)
        self.assertIn("don't know", out)


class TestAskAgent(unittest.TestCase):
    def setUp(self):
        self.ctx = ToolContext(active_agent="tom")

    def test_registered(self):
        self.assertIn("ask_agent", _REGISTRY)

    def test_queues_a_delegation_without_moving_the_user(self):
        out = dispatch(self.ctx, "ask_agent",
                       {"agent": "bob", "task": "Save this note: milk."})
        self.assertEqual(self.ctx.pending_delegations,
                         [("bob", "Save this note: milk.")])
        self.assertIsNone(self.ctx.pending_switch)  # a task moves, not the user
        self.assertIn("Bob", out)

    def test_two_delegations_queue_in_order(self):
        dispatch(self.ctx, "ask_agent", {"agent": "bob", "task": "one"})
        dispatch(self.ctx, "ask_agent", {"agent": "alice", "task": "two"})
        self.assertEqual([k for k, _ in self.ctx.pending_delegations],
                         ["bob", "alice"])

    def test_alias_resolution(self):
        dispatch(self.ctx, "ask_agent", {"agent": "bobby", "task": "t"})
        self.assertEqual(self.ctx.pending_delegations[0][0], "bob")

    def test_self_delegation_is_refused(self):
        out = dispatch(self.ctx, "ask_agent", {"agent": "tom", "task": "t"})
        self.assertEqual(self.ctx.pending_delegations, [])
        self.assertIn("yourself", out)

    def test_unknown_agent_is_rejected(self):
        out = dispatch(self.ctx, "ask_agent", {"agent": "dave", "task": "t"})
        self.assertEqual(self.ctx.pending_delegations, [])
        self.assertIn("don't know", out)

    def test_empty_task_is_rejected(self):
        out = dispatch(self.ctx, "ask_agent", {"agent": "bob"})
        self.assertEqual(self.ctx.pending_delegations, [])
        self.assertIn("task", out.lower())

    def test_every_hat_can_delegate(self):
        for key, agent in agents.AGENTS.items():
            self.assertIn("ask_agent", agent["tools"],
                          f"{key} cannot delegate")


class TestPerAgentToolFiltering(unittest.TestCase):
    def test_include_filters_to_allowlist(self):
        names = [t["name"] for t in api_tools(include={"get_current_time",
                                                       "switch_agent"})]
        self.assertEqual(sorted(names), ["get_current_time", "switch_agent"])

    def test_alice_cannot_touch_notes_and_tom_cannot_save(self):
        alice = {t["name"] for t in api_tools(include=agents.AGENTS["alice"]["tools"])}
        tom = {t["name"] for t in api_tools(include=agents.AGENTS["tom"]["tools"])}
        bob = {t["name"] for t in api_tools(include=agents.AGENTS["bob"]["tools"])}
        self.assertNotIn("search_notes", alice)
        self.assertNotIn("save_conversation_note", tom)
        self.assertIn("search_notes", bob)
        self.assertIn("search_knowledge", tom)

    def test_exclude_still_works_alongside_include(self):
        names = [t["name"] for t in
                 api_tools(exclude={"switch_agent"},
                           include={"switch_agent", "get_current_time"})]
        self.assertEqual(names, ["get_current_time"])


if __name__ == "__main__":
    unittest.main()
