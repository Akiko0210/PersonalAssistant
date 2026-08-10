"""Tests for same-session memory recall (memory.search_staged).

The guarantee under test: text that ages out of the live window MID-SESSION is
still findable. Consolidation only runs at boot, so staged lines used to be a
blind spot — neither in the window nor in the archive — which is how the
trading hat couldn't recall a trade structure discussed three hours earlier
(session_2026-07-20.log 21:07, "Review your memory" / "we just talked about
it, forgot it already")."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config as cfg
from brain.memory import ConversationMemory


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
        # The exact trading-hat scenario: aged out, then asked about.
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


class FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeClient:
    """Captures the recall prompt; returns a scripted answer or raises."""

    def __init__(self, answer=None, error=None):
        self.prompts = []
        self.models = []
        self._answer, self._error = answer, error
        self.messages = self

    def create(self, **kwargs):
        if self._error:
            raise self._error
        self.prompts.append(kwargs["messages"][0]["content"])
        self.models.append(kwargs.get("model"))
        return type("R", (), {"content": [FakeBlock(self._answer)]})()


class TestRecallStaged(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        patcher = mock.patch.object(cfg, "MEMORY_PENDING_PATH",
                                    Path(self.dir) / "memory_pending.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mem = ConversationMemory.__new__(ConversationMemory)
        # A pre-set empty collection keeps search() from loading the real
        # embedding model + Chroma store inside a unit test.
        self.mem._col = type("Col", (), {"count": lambda self: 0})()

    def _stage(self, *lines):
        self.mem.record_dropped([{"role": "user", "content": ln} for ln in lines])

    def test_llm_read_answers_complex_queries(self):
        # The staged text never contains the word "sizing" — a keyword scan
        # whiffs, but the model reads everything and can still answer.
        self._stage("put two-thirds into the thirty-five delta guts",
                    "the rest goes to the fifteen delta tail")
        client = FakeClient(answer="You decided two-thirds in the guts, "
                                   "the rest in the tail.")
        out = self.mem.recall_staged(client, "what did we decide about sizing?")
        self.assertIn("two-thirds", out)
        # The model saw the verbatim staged lines and the query.
        self.assertIn("thirty-five delta guts", client.prompts[0])
        self.assertIn("sizing", client.prompts[0])

    def test_nothing_relevant_is_an_answer_not_a_failure(self):
        self._stage("we talked about the weather")
        client = FakeClient(answer="NOTHING_RELEVANT")
        self.assertEqual(self.mem.recall_staged(client, "bitcoin"), "")

    def test_call_failure_returns_none_for_keyword_fallback(self):
        self._stage("delta sizing details")
        client = FakeClient(error=ConnectionError("offline"))
        self.assertIsNone(self.mem.recall_staged(client, "sizing"))
        # ...and search() then still finds it via the keyword scan.
        out = self.mem.search("delta sizing", client=client)
        self.assertIn("delta sizing details", out)

    def test_nothing_staged_never_calls_the_model(self):
        client = FakeClient(answer="should never be used")
        self.assertIsNone(self.mem.recall_staged(client, "anything"))
        self.assertEqual(client.prompts, [])

    def test_search_without_client_uses_keyword_scan(self):
        self._stage("the kicker is sixty-five days out")
        out = self.mem.search("kicker")
        self.assertIn("kicker", out)

    def test_recall_uses_its_own_model_not_the_conversation_one(self):
        # Recall pins a stronger model than the latency-picked conversation
        # default: on Haiku it answered NOTHING_RELEVANT for topics plainly in
        # the staged text (2026-07-27). Sending the wrong model here is what
        # made the whole feature unreliable, so it is asserted.
        from brain import memory as memory_mod
        self._stage("the kicker is sixty-five days out")
        client = FakeClient(answer="something")
        self.mem.recall_staged(client, "kicker")
        self.assertEqual(client.models, [memory_mod.RECALL_MODEL])


class TestFalseNegativeDoesNotHideStagedLines(unittest.TestCase):
    """A NOTHING_RELEVANT verdict must no longer suppress the keyword scan."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        patcher = mock.patch.object(cfg, "MEMORY_PENDING_PATH",
                                    Path(self.dir) / "memory_pending.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mem = ConversationMemory.__new__(ConversationMemory)
        self.mem._col = type("Col", (), {"count": lambda self: 0})()

    def _stage(self, *lines):
        self.mem.record_dropped([{"role": "user", "content": ln} for ln in lines])

    def test_keyword_hits_survive_a_nothing_relevant_verdict(self):
        # The 2026-07-27 failure in miniature: the read says nothing is
        # relevant while the words sit in the staged lines.
        self._stage("I'm going to be long the butterfly against the diagonal")
        out = self.mem.search("butterfly", client=FakeClient(answer="NOTHING_RELEVANT"))
        self.assertIn("long the butterfly", out)

    def test_those_hits_are_labelled_as_unverified(self):
        self._stage("I'm going to be long the butterfly")
        out = self.mem.search("butterfly", client=FakeClient(answer="NOTHING_RELEVANT"))
        self.assertIn("keyword matches", out)

    def test_a_real_read_is_not_labelled_as_unverified(self):
        self._stage("I'm going to be long the butterfly")
        out = self.mem.search("butterfly", client=FakeClient(answer="You went long."))
        self.assertIn("You went long.", out)
        self.assertNotIn("keyword matches", out)

    def test_genuinely_absent_topic_still_reports_nothing(self):
        # The fallback must not invent relevance: no keyword hit, no section.
        self._stage("we talked about the weather")
        out = self.mem.search("bitcoin", client=FakeClient(answer="NOTHING_RELEVANT"))
        self.assertIn("No archived conversations yet", out)


class TestArchiveFailureIsNotFatal(unittest.TestCase):
    """A broken Chroma archive must not discard the staged results."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        patcher = mock.patch.object(cfg, "MEMORY_PENDING_PATH",
                                    Path(self.dir) / "memory_pending.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mem = ConversationMemory.__new__(ConversationMemory)
        self.mem._col = None
        self.ensure_calls = 0

    def _stage(self, *lines):
        self.mem.record_dropped([{"role": "user", "content": ln} for ln in lines])

    def _break_chroma(self, error="Error creating hnsw segment reader: "
                                  "Nothing found on disk.", heal_on=None):
        """Install an _ensure_chroma that yields a failing collection, and
        optionally a working one from attempt `heal_on` onward."""
        outer = self

        class Col:
            def __init__(self, ok):
                self.ok = ok

            def count(self):
                return 3

            def query(self, **kwargs):
                if not self.ok:
                    raise RuntimeError(error)
                return {"documents": [["archived text"]],
                        "metadatas": [[{"date": "2026-07-25"}]]}

        def fake_ensure():
            outer.ensure_calls += 1
            ok = heal_on is not None and outer.ensure_calls >= heal_on
            outer.mem._col = Col(ok)

        self.mem._ensure_chroma = fake_ensure

    def test_staged_results_survive_an_archive_failure(self):
        self._stage("the kicker is sixty-five days out")
        self._break_chroma()
        out = self.mem.search("kicker")
        self.assertIn("kicker", out)                       # not lost
        self.assertIn("could not be read", out)            # and reported

    def test_failure_alone_is_reported_as_unsearchable_not_as_absent(self):
        # "Nothing matches" would be a lie the model could repeat back.
        self._break_chroma()
        out = self.mem.search("anything")
        self.assertIn("could not be read", out)
        self.assertNotIn("Nothing in past conversations matches", out)

    def test_it_retries_once_with_a_rebuilt_collection(self):
        self._break_chroma(heal_on=2)  # second _ensure_chroma yields a good one
        out = self.mem.search("anything")
        self.assertEqual(self.ensure_calls, 2)
        self.assertIn("archived text", out)
        self.assertNotIn("could not be read", out)

    def test_it_does_not_retry_forever(self):
        self._break_chroma()
        self.mem.search("anything")
        self.assertEqual(self.ensure_calls, 2)


class TestEveryHatCanSearchMemory(unittest.TestCase):
    def test_shared_memory_is_searchable_by_every_persona(self):
        # The conversation memory is shared; a hat without the search tool
        # cannot obey "review your memory" — the exact 2026-07-20 failure.
        from brain import agents
        for key, agent in agents.AGENTS.items():
            self.assertIn("search_past_conversations", agent["tools"], key)


if __name__ == "__main__":
    unittest.main()
