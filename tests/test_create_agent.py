"""Tests for create_agent: a fresh-context copy of a persona under a new name (#72)."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config as cfg
from brain import agents
from tools import ToolContext, dispatch


class TestCreateAgent(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for name, path in (("AGENTS_PATH", "agents.json"),
                           ("CREATED_AGENTS_PATH", "created_agents.json")):
            p = mock.patch.object(cfg, name, self.dir / path)
            p.start()
            self.addCleanup(p.stop)
        saved = copy.deepcopy(agents.AGENTS)
        self.addCleanup(lambda: (agents.AGENTS.clear(), agents.AGENTS.update(saved)))
        agents.load_agents()

    def create(self, name, said=None, by="tom"):
        ctx = ToolContext(active_agent=by,
                          user_text=said or f"Create an agent called {name}.")
        return dispatch(ctx, "create_agent", {"name": name})

    def test_copy_has_the_creators_settings_under_its_own_name(self):
        out = self.create("Noah")
        self.assertIn("Created Noah", out)
        noah, tom = agents.AGENTS["noah"], agents.AGENTS["tom"]
        for field in ("role", "persona", "tools", "brokers", "record_rules",
                      "model", "tts_voice", "tts_rate"):
            self.assertEqual(noah[field], tom[field], field)
        self.assertEqual((noah["name"], noah["aliases"]), ("Noah", ("noah",)))
        self.assertEqual(agents.match_address("Noah, what's SPX at?"),
                         ("noah", "what's SPX at?"))

    def test_fresh_context(self):
        # Every private store, the record and the thread are keyed by the
        # persona key: a new key reads nothing of Tom's.
        self.create("Noah")
        self.assertEqual(agents.readable_owners("noah"), ("noah",))
        self.assertNotEqual(cfg.history_path("noah"), cfg.history_path("tom"))

    def test_inherits_the_creators_dashboard_edits_but_not_its_aliases(self):
        (self.dir / "agents.json").write_text(json.dumps(
            {"tom": {"model": "haiku", "tts_rate": 190, "aliases": ["tom"]}}))
        agents.load_agents()
        self.create("Noah")
        noah = agents.AGENTS["noah"]
        self.assertEqual((noah["model"], noah["tts_rate"]), ("haiku", 190))
        self.assertEqual(noah["aliases"], ("noah",))

    def test_survives_a_restart_and_a_dashboard_reset(self):
        self.create("Noah")
        # A fresh process, with the dashboard's reset having dropped every
        # agents.json entry: the assistant is still there.
        (self.dir / "agents.json").unlink(missing_ok=True)
        agents.AGENTS.clear()
        agents.AGENTS.update(copy.deepcopy(agents._DEFAULTS))
        agents.load_agents()
        self.assertEqual(agents.AGENTS["noah"]["name"], "Noah")
        self.assertIn("noah", agents.defaults())  # the dashboard's reset view

    def test_a_copy_of_a_copy_copies_the_original(self):
        self.create("Noah")
        self.create("Liam", by="noah")
        self.assertEqual(agents.AGENTS["liam"]["parent"], "tom")
        self.assertEqual(agents.AGENTS["liam"]["tools"], agents.AGENTS["tom"]["tools"])

    def test_reachable_at_once_by_switch_and_ask(self):
        self.create("Noah")
        ctx = ToolContext(active_agent="alice", user_text="Ask Noah to check.")
        dispatch(ctx, "switch_agent", {"agent": "Noah"})
        self.assertEqual(ctx.pending_switch, ("noah", ""))
        dispatch(ctx, "ask_agent", {"agent": "noah", "task": "check SPX"})
        self.assertEqual(ctx.pending_delegations, [("noah", "check SPX")])

    def test_refusals_create_nothing(self):
        cases = (("Noah", "Make me another trading assistant."),   # not said
                 ("Bob", None),                                    # a name
                 ("Tommy", None),                                  # an alias
                 ("Noah Smith", None),                             # two words
                 ("N0ah", None))
        for name, said in cases:
            with self.subTest(name=name):
                out = self.create(name, said)
                self.assertTrue(out.startswith("Not created"), out)
        self.assertFalse((self.dir / "created_agents.json").exists())
        self.assertEqual(set(agents.AGENTS), set(agents._DEFAULTS))

    def test_only_tom_holds_the_tool(self):
        holders = {k for k, a in agents.AGENTS.items() if "create_agent" in a["tools"]}
        self.assertEqual(holders, {"tom"})


if __name__ == "__main__":
    unittest.main()
