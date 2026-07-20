"""Tests for the agents registry and the spoken-address router."""

import unittest

import agents
from tools import _REGISTRY


class TestRegistryIntegrity(unittest.TestCase):
    def test_every_allowlisted_tool_exists(self):
        # A typo in an allowlist would silently strip the tool from the API
        # call; catch it at test time instead.
        for key, agent in agents.AGENTS.items():
            for name in agent["tools"]:
                self.assertIn(name, _REGISTRY, f"{key} allowlists unknown tool {name}")

    def test_aliases_globally_unique(self):
        agents.alias_map()  # raises on a duplicate

    def test_personas_and_roles_non_empty(self):
        for key, agent in agents.AGENTS.items():
            self.assertTrue(agent["persona"].strip(), key)
            self.assertTrue(agent["role"].strip(), key)
            self.assertTrue(agent["aliases"], key)

    def test_default_agent_exists(self):
        self.assertIn(agents.DEFAULT_AGENT, agents.AGENTS)

    def test_every_agent_can_switch_and_change_model(self):
        # Without switch_agent an agent is a roach motel; without
        # set_conversation_model "make this smarter" breaks in that hat.
        for key, agent in agents.AGENTS.items():
            self.assertIn("switch_agent", agent["tools"], key)
            self.assertIn("set_conversation_model", agent["tools"], key)

    def test_roster_mentions_the_others_and_shared_memory(self):
        block = agents.roster_block("alice")
        self.assertIn("Bob", block)
        self.assertIn("Cobe", block)
        self.assertIn("share", block.lower())

    def test_resolve_tolerates_aliases_and_case(self):
        self.assertEqual(agents.resolve("Kobe"), "cobe")
        self.assertEqual(agents.resolve("BOB"), "bob")
        self.assertEqual(agents.resolve("alice"), "alice")
        self.assertIsNone(agents.resolve("nobody"))
        self.assertIsNone(agents.resolve(""))


class TestMatchAddress(unittest.TestCase):
    def test_name_prefix_switches_and_keeps_the_question(self):
        key, rest = agents.match_address("Bob, what was my last note?")
        self.assertEqual(key, "bob")
        self.assertEqual(rest, "what was my last note?")

    def test_misheard_alias_with_filler(self):
        key, rest = agents.match_address("hey kobe check trades")
        self.assertEqual(key, "cobe")
        self.assertEqual(rest, "check trades")

    def test_explicit_switch_phrases(self):
        self.assertEqual(agents.match_address("switch to bob"), ("bob", ""))
        self.assertEqual(agents.match_address("I wanna talk to Cobe")[0], "cobe")
        self.assertEqual(agents.match_address("let me talk to alice")[0], "alice")

    def test_bare_name(self):
        self.assertEqual(agents.match_address("Alice."), ("alice", ""))

    def test_mid_sentence_name_never_triggers(self):
        key, rest = agents.match_address("tell Bob thanks")
        self.assertIsNone(key)
        self.assertEqual(rest, "tell Bob thanks")

    def test_plain_speech_passes_through(self):
        key, rest = agents.match_address("what time is it")
        self.assertIsNone(key)
        self.assertEqual(rest, "what time is it")

    def test_remainder_keeps_original_casing(self):
        _, rest = agents.match_address("Bob, Read my Trading note")
        self.assertEqual(rest, "Read my Trading note")

    def test_empty_input(self):
        self.assertEqual(agents.match_address(""), (None, ""))


if __name__ == "__main__":
    unittest.main()
