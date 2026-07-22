"""Tests for same-session memory recall (memory.search_staged).

The guarantee under test: text that ages out of the live window MID-SESSION is
still findable. Consolidation only runs at boot, so staged lines used to be a
blind spot — neither in the window nor in the archive — which is how Cobe
couldn't recall a trade structure discussed three hours earlier
(session_2026-07-20.log 21:07, "Review your memory" / "we just talked about
it, forgot it already")."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config as cfg
from memory import ConversationMemory


class TestSearchStaged(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.pending = Path(self.dir) / "memory_pending.json"
        patcher = mock.patch.object(cfg, "MEMORY_PENDING_PATH", self.pending)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mem = ConversationMemory.__new__(ConversationMemory)  # skip ensure_dirs
        self.mem._col = None

    def _stage(self, *lines):
        self.mem.record_dropped([{"role": "user", "content": ln} for ln in lines])

    def test_finds_aged_out_lines_by_keyword(self):
        # The exact Cobe scenario: the structure aged out, then is asked about.
        self._stage(
            "put two-thirds into the core thirty-five delta puts, the guts",
            "the kicker is the fifteen delta tail sleeve sixty-five days out",
        )
        hits = self.mem.search_staged("guts kicker delta structure")
        self.assertEqual(len(hits), 2)
        self.assertIn("guts", hits[0])
        self.assertIn("kicker", hits[1])

    def test_no_match_returns_empty(self):
        self._stage("we talked about the weather")
        self.assertEqual(self.mem.search_staged("bitcoin futures"), [])

    def test_short_and_empty_queries_return_empty(self):
        self._stage("something about options")
        self.assertEqual(self.mem.search_staged(""), [])
        self.assertEqual(self.mem.search_staged("a an of"), [])  # all too short

    def test_recent_matches_win_the_budget(self):
        for i in range(30):
            self._stage(f"trading detail number {i}")
        hits = self.mem.search_staged("trading detail", max_lines=5)
        self.assertEqual(len(hits), 5)
        self.assertIn("number 29", hits[-1])  # newest kept

    def test_long_lines_are_truncated(self):
        self._stage("delta " * 200)
        (hit,) = self.mem.search_staged("delta")
        self.assertLess(len(hit), 340)

    def test_nothing_staged_is_fine(self):
        self.assertEqual(self.mem.search_staged("anything"), [])


class TestEveryHatCanSearchMemory(unittest.TestCase):
    def test_shared_memory_is_searchable_by_every_persona(self):
        # The conversation memory is shared; a hat without the search tool
        # cannot obey "review your memory" — the exact 2026-07-20 failure.
        import agents
        for key, agent in agents.AGENTS.items():
            self.assertIn("search_past_conversations", agent["tools"], key)


if __name__ == "__main__":
    unittest.main()
