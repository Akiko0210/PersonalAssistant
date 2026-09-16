"""Tests for the exchange index (brain/memory.py): what gets written per
turn, how it is read back (dense + lexical, per persona), and the honest
failure paths — a broken collection is reported as unsearchable, never as
"nothing there" (2026-07-27 08:12, the hnsw segment reader fault).

Nothing here loads Chroma, the embedding model, or a model client.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config as cfg
from brain import agents
from brain.memory import ConversationMemory, exchange_id
from lib.atomic_io import write_json_atomic
from tests.store_fixtures import FakeCol


def user(text, ts="2026-08-26T22:44:41-07:00"):
    return {"role": "user", "content": text, "ts": ts}


def assistant(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def make_memory(cols):
    mem = ConversationMemory.__new__(ConversationMemory)
    mem._cols, mem._lexical = {}, {}
    mem._col_for = lambda name: cols.setdefault(name, FakeCol())
    return mem


class TestIndexing(unittest.TestCase):
    def setUp(self):
        self.cols = {}
        self.mem = make_memory(self.cols)
        self.col_name = cfg.agent_memory_collection("alice")

    def test_exchange_ids_are_deterministic_and_stamp_sensitive(self):
        a, b = user("hi"), user("hi")
        self.assertEqual(exchange_id("alice", a), exchange_id("alice", b))
        self.assertNotEqual(exchange_id("alice", a), exchange_id("bob", a))
        self.assertNotEqual(exchange_id("alice", a),
                            exchange_id("alice", user("hi", ts="2026-01-01T00:00:00")))

    def test_one_record_per_exchange_with_tool_traffic_folded_to_metadata(self):
        history = [
            user("what time is it"),
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "get_current_time", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "9:00"}]},
            assistant("It's nine."),
            assistant("(Note to self — I saved the note.)"),
        ]
        self.assertEqual(self.mem.index_exchanges(history, "alice"), 1)
        ((ids, docs, metas),) = self.cols[self.col_name].upserts
        self.assertEqual(docs[0], "user: what time is it\nassistant: It's nine.\n"
                                  "assistant: (Note to self — I saved the note.)")
        self.assertEqual(metas[0]["tools"], "get_current_time")
        self.assertEqual(metas[0]["kind"], "exchange")
        self.assertIsInstance(metas[0]["epoch"], float)
        self.assertNotIn("tool_result", docs[0])

    def test_an_unanswered_turn_is_not_a_record(self):
        self.assertEqual(self.mem.index_exchanges([user("hello?")], "alice"), 0)
        self.assertNotIn(self.col_name, self.cols)

    def test_skip_existing_makes_the_backfill_idempotent(self):
        history = [user("a"), assistant("b")]
        self.assertEqual(self.mem.index_exchanges(history, "alice"), 1)
        self.assertEqual(self.mem.index_exchanges(history, "alice"), 0)
        # The live turn overwrites so a later self-note lands in the record.
        self.assertEqual(self.mem.index_exchanges(history, "alice",
                                                  skip_existing=False), 1)


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.cols = {}
        self.mem = make_memory(self.cols)

    def test_lexical_hits_come_from_the_indexed_records(self):
        # The dense fake returns nothing; BM25 over the same records finds
        # the exact ticker.
        self.mem.index_exchanges([user("sell the SPX 5800 put"),
                                  assistant("done")], "alice")
        self.mem.index_exchanges([user("how is the weather"),
                                  assistant("sunny")], "alice")
        rows = self.mem.query_rows("SPX", caller="alice")
        self.assertEqual(rows.dense, [])
        self.assertEqual([h.doc.split("\n")[0] for h in rows.lexical],
                         ["user: sell the SPX 5800 put"])
        self.assertEqual(rows.count, 2)

    def test_the_lexical_index_sees_a_record_written_after_it_was_built(self):
        self.mem.index_exchanges([user("first"), assistant("ok")], "alice")
        self.mem.query_rows("first", caller="alice")  # builds the index
        self.mem.index_exchanges([user("NVDA earnings"), assistant("ok")], "alice")
        rows = self.mem.query_rows("NVDA", caller="alice")
        self.assertEqual(len(rows.lexical), 1)

    def test_a_persona_reads_only_its_own_and_the_legacy_archive(self):
        self.mem.query_rows("anything", caller="tom")
        self.assertEqual(set(self.cols),
                         {cfg.agent_memory_collection("tom"), cfg.MEMORY_COLLECTION})

    def test_legacy_rows_are_labelled_in_the_tool_result(self):
        self.cols[cfg.MEMORY_COLLECTION] = FakeCol(
            [("archived summary text", {"date": "2026-07-25"}, 0.3)])
        out = self.mem.search("summary", caller="alice")
        self.assertIn("[2026-07-25 (from the shared archive", out)
        self.assertIn("archived summary text", out)


class TestArchiveFailureIsHonest(unittest.TestCase):
    def setUp(self):
        self.mem = ConversationMemory.__new__(ConversationMemory)
        self.mem._cols, self.mem._lexical = {}, {}
        self.ensure_calls = 0

    def _break_chroma(self, heal_on=None):
        outer = self

        class Col:
            def __init__(self, ok):
                self.ok = ok

            def count(self):
                return 3

            def query(self, **kwargs):
                if not self.ok:
                    raise RuntimeError("Error creating hnsw segment reader")
                return {"ids": [["c1"]], "documents": [["archived text"]],
                        "metadatas": [[{"date": "2026-07-25"}]],
                        "distances": [[0.3]]}

            def get(self, **kwargs):
                return {"ids": ["c1"], "documents": ["archived text"],
                        "metadatas": [{"date": "2026-07-25"}]}

        def fake_col_for(name):
            outer.ensure_calls += 1
            return Col(heal_on is not None and outer.ensure_calls >= heal_on)

        self.mem._col_for = fake_col_for

    def test_failure_is_reported_as_unsearchable_not_as_absent(self):
        self._break_chroma()
        out = self.mem.search("anything", caller="alice")
        self.assertIn("could not be read", out)
        self.assertNotIn("Nothing in past conversations matches", out)

    def test_it_retries_once_with_a_rebuilt_collection(self):
        self._break_chroma(heal_on=2)
        out = self.mem.search("anything", caller="alice")
        self.assertIn("archived text", out)
        self.assertNotIn("could not be read", out)

    def test_it_does_not_retry_forever(self):
        self._break_chroma()
        self.mem.search("anything", caller="alice")
        # Two collections (own + legacy), two attempts each.
        self.assertEqual(self.ensure_calls, 4)


class TestMigratePending(unittest.TestCase):
    def test_batches_are_indexed_and_the_file_is_parked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory_pending.json"
            write_json_atomic(path, [
                {"ts": "2026-08-26T22:44:41", "agent": "bob",
                 "lines": ["user: switch to sonnet", "assistant: Done."]},
                {"ts": "2026-07-01T10:00:00", "agent": None,
                 "lines": ["user: legacy", "assistant: untagged"]},
            ])
            cols = {}
            mem = make_memory(cols)
            with mock.patch.object(cfg, "MEMORY_PENDING_PATH", path):
                self.assertEqual(mem.migrate_pending(), 1)
                self.assertEqual(mem.migrate_pending(), 0)  # file is gone
            self.assertFalse(path.exists())
            self.assertTrue((Path(tmp) / "memory_pending.json.bak").exists())
            ((ids, docs, metas),) = cols[cfg.agent_memory_collection("bob")].upserts
            self.assertEqual(docs[0], "user: switch to sonnet\nassistant: Done.")
            self.assertEqual(metas[0]["ts"], "2026-08-26T22:44:41")


class TestEveryHatCanSearchMemory(unittest.TestCase):
    def test_own_memory_is_searchable_by_every_persona(self):
        # A hat without the search tool cannot obey "review your memory" —
        # the exact 2026-07-20 failure. The Background covers the common
        # case now, but the rule stands: every persona must reliably reach
        # its OWN past when asked for more.
        for key, agent in agents.AGENTS.items():
            self.assertIn("search_past_conversations", agent["tools"], key)


if __name__ == "__main__":
    unittest.main()
