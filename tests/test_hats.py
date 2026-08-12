"""Persona ("hat") smoke tests against a fake client: switching agents must
change the model, the tool subset, the system prompt — and since the memory
split, the history THREAD: each persona's conversation is its own, and
nothing of another's leaks across a switch."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from brain import agents
from brain.llm import Claude
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

    def test_threads_are_isolated_across_switching(self):
        # The strict-isolation core: Bob must not see a word of Alice's
        # conversation — context crosses over only as ask_agent summaries.
        c = make_claude()
        c.converse("tell alice something")
        c.switch_to("bob")
        c.converse("bob, do you remember?")
        texts = str(c.client.messages.calls[1]["messages"])
        self.assertNotIn("tell alice something", texts)

    def test_a_thread_survives_switching_away_and_back(self):
        # Isolation must not mean amnesia: Tom's own thread resumes intact
        # after a detour through Alice.
        c = make_claude(active="tom")
        c.converse("remember my butterfly")
        tom_thread = list(c.history)
        c.switch_to("alice")
        c.converse("hello alice")
        c.switch_to("tom")
        self.assertEqual(c.history, tom_thread)

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

    def test_departing_thread_is_saved_before_the_swap(self):
        # _save_history writes to the ACTIVE persona's file; saving after the
        # flip would clobber the target's thread with the departing one.
        c = make_claude()
        c.converse("alice's words")
        c.switch_to("bob")
        self.assertIn("alice's words", str(c.threads["alice"]))
        self.assertNotIn("alice's words", str(c.threads.get("bob", [])))

    def test_switching_to_the_active_hat_is_a_no_op(self):
        c = make_claude(history=[{"role": "user", "content": "hi"}])
        c.switch_to(agents.DEFAULT_AGENT)
        self.assertEqual(c.history, [{"role": "user", "content": "hi"}])
        self.assertEqual(c.saved, [])  # not even a save

    def test_pending_switch_roundtrip(self):
        c = make_claude()
        c._ctx.pending_switch = ("bob", "what's my last note?")
        self.assertEqual(c.take_pending_switch(), ("bob", "what's my last note?"))
        self.assertIsNone(c.take_pending_switch())


class TestLegacyHistoryMigration(unittest.TestCase):
    """The pre-isolation shared history.json is parked as .bak, once — its
    turns reach per-agent memory via scripts/seed_agent_memory.py (the logs
    carry the same turns WITH attribution), not via staging."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.legacy = Path(self.tmp.name) / "history.json"
        p = mock.patch.object(cfg, "HISTORY_PATH", self.legacy)
        p.start()
        self.addCleanup(p.stop)

    def test_legacy_file_is_parked_as_bak(self):
        self.legacy.write_text('[{"role": "user", "content": "old"}]',
                               encoding="utf-8")
        Claude._migrate_legacy_history()
        self.assertFalse(self.legacy.exists())
        bak = self.legacy.with_suffix(".json.bak")
        self.assertIn("old", bak.read_text(encoding="utf-8"))

    def test_no_legacy_file_is_a_quiet_no_op(self):
        Claude._migrate_legacy_history()  # must not raise
        self.assertFalse(self.legacy.with_suffix(".json.bak").exists())

    def test_an_existing_backup_is_never_overwritten(self):
        # Path.replace clobbers its destination silently, and this machine
        # had a hand-made history.json.bak from a month before the migration
        # was written — parking over it would have destroyed the only copy.
        bak = self.legacy.with_suffix(".json.bak")
        bak.write_text("older hand-made backup", encoding="utf-8")
        self.legacy.write_text("[]", encoding="utf-8")
        Claude._migrate_legacy_history()
        self.assertEqual(bak.read_text(encoding="utf-8"),
                         "older hand-made backup")
        self.assertEqual(self.legacy.with_suffix(".json.bak2")
                         .read_text(encoding="utf-8"), "[]")


if __name__ == "__main__":
    unittest.main()
