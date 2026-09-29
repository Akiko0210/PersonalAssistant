"""Tests for the exchange index (brain/memory.py): what gets written per
turn, how it is read back (dense + lexical, per persona), the marks the fact
extractor leaves, the recall tool's seams, and the honest failure paths — a
broken collection is reported as unsearchable, never as "nothing there"
(2026-07-27 08:12, the hnsw segment reader fault).

Nothing here loads Chroma, the embedding model, or a model client.
"""

import unittest
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import config as cfg
from brain import agents, context
from brain.memory import ConversationMemory, exchange_id
from lib.dates import period_range, window_epochs
from tests.store_fixtures import FakeCol, memory_over
from tools import ToolContext, dispatch


def user(text, ts="2026-08-26T22:44:41-07:00"):
    return {"role": "user", "content": text, "ts": ts}


def assistant(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


class TestIndexing(unittest.TestCase):
    def setUp(self):
        self.cols = {}
        self.mem = memory_over(self.cols)
        self.col_name = cfg.agent_memory_collection("alice")

    def test_exchange_ids_are_deterministic_and_stamp_sensitive(self):
        a, b = user("hi"), user("hi")
        self.assertEqual(exchange_id("alice", a), exchange_id("alice", b))
        self.assertNotEqual(exchange_id("alice", a), exchange_id("bob", a))
        self.assertNotEqual(exchange_id("alice", a),
                            exchange_id("alice", user("hi", ts="2026-01-01T00:00:00")))

    def test_an_exchange_is_its_parent_plus_an_assistant_companion(self):
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
        xid = exchange_id("alice", history[0])
        self.assertEqual(ids, [xid, xid + ":a"])
        self.assertEqual(docs[0], "user: what time is it\nassistant: It's nine.\n"
                                  "assistant: (Note to self — I saved the note.)")
        # The companion embeds the reply's lead line but carries the whole
        # exchange, so a hit on it reads back as its parent.
        self.assertEqual(docs[1], "assistant: It's nine.")
        self.assertEqual(metas[1]["exchange"], xid)
        self.assertEqual(metas[1]["text"], docs[0])
        for meta in metas:
            self.assertEqual(meta["tools"], "get_current_time")
            self.assertEqual(meta["kind"], "exchange")
            self.assertIsInstance(meta["epoch"], float)
            self.assertIs(meta["approx"], False)  # an offset-bearing stamp is exact
        self.assertNotIn("tool_result", docs[0])
        # exchanges_of is the one derivation the index and the extractor share.
        ((x, doc, meta),) = self.mem.exchanges_of(history, "alice")
        self.assertEqual((x, doc, meta["tools"]), (xid, docs[0], "get_current_time"))

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


class TestExtractorMarks(unittest.TestCase):
    def test_unread_exchanges_come_oldest_first_and_marks_stick(self):
        cols = {}
        mem = memory_over(cols)
        later = user("second", ts="2026-09-18T09:25:00-07:00")
        first = user("first", ts="2026-09-18T09:24:00-07:00")
        mem.index_exchanges([later, assistant("b"), first, assistant("a")], "alice")
        pending = mem.unextracted("alice")
        self.assertEqual([x for x, _, _ in pending],
                         [exchange_id("alice", first), exchange_id("alice", later)])
        self.assertEqual(pending[0][1], "user: first\nassistant: a")
        mem.mark_extracted("alice", exchange_id("alice", first))
        self.assertEqual([x for x, _, _ in mem.unextracted("alice")],
                         [exchange_id("alice", later)])
        # A metadata-only merge: the record keeps everything else it had.
        col = cols[cfg.agent_memory_collection("alice")]
        meta = col.rows[exchange_id("alice", first)][1]
        self.assertEqual((meta["extracted"], meta["kind"]), (True, "exchange"))


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.cols = {}
        self.mem = memory_over(self.cols)

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
        self.assertEqual(rows.count, 4)  # records: two exchanges, two keys each

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

    def test_fetch_reads_exchanges_by_id_oldest_first(self):
        a = user("first thing", ts="2026-09-18T09:24:00-07:00")
        b = user("second thing", ts="2026-09-18T09:25:00-07:00")
        self.mem.index_exchanges([b, assistant("B"), a, assistant("A")], "alice")
        out = self.mem.fetch([exchange_id("alice", b), exchange_id("alice", a), "xc_nope"],
                             caller="alice")
        self.assertLess(out.index("first thing"), out.index("second thing"))
        self.assertIn("[9:24am 9/18/2026]", out)
        self.assertIn("(1 id(s) not found.)", out)
        self.assertIn("None of those", self.mem.fetch(["xc_nope"], caller="alice"))


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


class TestCompanionKeys(unittest.TestCase):
    """Two keys, one value (2026-09-18): a hit on the reply-only companion
    reads back as its exchange, counts once, and is excluded with it."""

    def setUp(self):
        self.cols = {}
        self.mem = memory_over(self.cols)
        self.name = cfg.agent_memory_collection("alice")
        self.full = "user: my first item\nassistant: Got it. Your list: one, first item."
        stamp = {"ts": "2026-09-17T09:42:33-07:00", "approx": False}
        self.cols[self.name] = FakeCol([
            (self.full, stamp, 0.8, "xc_1"),
            ("assistant: Got it. Your list: one, first item.",
             {**stamp, "exchange": "xc_1", "part": "a", "text": self.full}, 0.2, "xc_1:a"),
        ])

    def test_key_hits_collapse_to_one_exchange_at_the_best_similarity(self):
        rows = self.mem.query_rows("what is my list", caller="alice")
        self.assertEqual([h.id for h in rows.dense], ["xc_1", "xc_1"])
        self.assertEqual({h.doc for h in rows.dense}, {self.full})
        self.assertNotIn("text", rows.dense[0].meta)
        (c,) = context.fuse(rows.dense, rows.lexical)
        self.assertAlmostEqual(c.sim, context.similarity(0.2))

    def test_an_excluded_exchange_takes_its_companion_with_it(self):
        out = self.mem.search("what is my list", caller="alice", exclude_ids={"xc_1"})
        self.assertIn("Nothing in past conversations", out)

    def test_the_lexical_corpus_has_one_entry_per_exchange(self):
        cols = {}
        mem = memory_over(cols)
        mem.index_exchanges([user("sell the SPX put"), assistant("done")], "alice")
        _, ids, docs, _ = mem._lexical_index(cfg.agent_memory_collection("alice"))
        self.assertEqual(ids, [exchange_id("alice", user("sell the SPX put"))])
        self.assertEqual(docs, ["user: sell the SPX put\nassistant: done"])


class TestBackfillKeys(unittest.TestCase):
    def test_legacy_records_gain_a_companion_and_an_approx_stamp_once(self):
        cols = {}
        mem = memory_over(cols)
        col = cols.setdefault(cfg.agent_memory_collection("alice"), FakeCol())
        # Two records from before the companion keys: one flush-stamped by
        # the old staging file (naive ts), one exact (offset-bearing).
        col.upsert(["xc_old", "xc_new"],
                   ["user: I\nassistant: It's morning now — what's up?",
                    "user: hi\nassistant: hello"],
                   [{"agent": "alice", "kind": "exchange",
                     "ts": "2026-09-17T09:43:11", "epoch": 1.0},
                    {"agent": "alice", "kind": "exchange",
                     "ts": "2026-09-17T09:43:11-07:00", "epoch": 2.0}])
        self.assertEqual(mem.backfill_keys("alice"), (2, 2))
        self.assertIs(col.rows["xc_old"][1]["approx"], True)
        self.assertIs(col.rows["xc_new"][1]["approx"], False)
        doc, meta = col.rows["xc_old:a"]
        self.assertEqual(doc, "assistant: It's morning now — what's up?")
        self.assertEqual((meta["exchange"], meta["approx"]), ("xc_old", True))
        self.assertEqual(len(col.updates), 1)  # metadata only: nothing re-embedded
        self.assertEqual(mem.backfill_keys("alice"), (0, 0))


class TestTimeWindow(unittest.TestCase):
    """A time window is a filter on exact stamps, not a similarity: the
    exchanges from "this morning at 9:41" sit at cosine 0.02-0.09 against
    that question (2026-09-17 12:10)."""

    def setUp(self):
        self.cols = {}
        self.mem = memory_over(self.cols)
        self.name = cfg.agent_memory_collection("alice")

    def test_a_window_becomes_a_where_on_epoch_and_filters_the_lexical_list(self):
        for text, ts in (("sell the SPX put", "2026-09-17T09:42:33-07:00"),
                         ("SPX again, later", "2026-09-17T15:00:00-07:00"),
                         ("SPX with a flush stamp", "2026-09-17T09:43:11")):
            self.mem.index_exchanges([user(text, ts=ts), assistant("ok")], "alice")
        window = window_epochs(since="2026-09-17T09:00:00-07:00",
                               until="2026-09-17T10:00:00-07:00")
        rows = self.mem.query_rows("SPX", caller="alice", window=window)
        lo, hi = window
        self.assertEqual(self.cols[self.name].wheres[-1],
                         {"$and": [{"approx": False},
                                   {"epoch": {"$gte": lo}}, {"epoch": {"$lte": hi}}]})
        self.assertEqual([h.doc.split("\n")[0] for h in rows.lexical],
                         ["user: sell the SPX put"])

    def test_a_windowed_search_skips_the_gate_and_reads_oldest_first(self):
        t1 = datetime.fromisoformat("2026-09-17T09:41:28-07:00").timestamp()
        t2 = datetime.fromisoformat("2026-09-17T09:45:23-07:00").timestamp()
        self.cols[self.name] = FakeCol([
            ("user: item three\nassistant: Your list now: one, two, three.",
             {"ts": "2026-09-17T09:45:23-07:00", "approx": False, "epoch": t2}, 0.95, "xc_2"),
            ("user: start my list\nassistant: Ready. First item?",
             {"ts": "2026-09-17T09:41:28-07:00", "approx": False, "epoch": t1}, 0.95, "xc_1"),
        ])
        out = self.mem.search("recall this morning", caller="alice",
                              window=(t1 - 60, t2 + 60))
        self.assertIn("(oldest first)", out)
        self.assertLess(out.index("start my list"), out.index("item three"))
        # The same hits, unwindowed, fail the gate: cosine 0.05 is noise.
        self.assertIn("Nothing in past conversations",
                      self.mem.search("recall this morning", caller="alice"))

    def test_a_window_without_a_topic_is_a_plain_read_of_the_stretch(self):
        for text, ts in (("third", "2026-09-17T09:45:00-07:00"),
                         ("first", "2026-09-17T09:41:00-07:00"),
                         ("outside", "2026-09-17T15:00:00-07:00")):
            self.mem.index_exchanges([user(text, ts=ts), assistant("ok")], "alice")
        window = window_epochs(since="2026-09-17T09:00:00-07:00",
                               until="2026-09-17T10:00:00-07:00")
        out = self.mem.search("", caller="alice", window=window)
        self.assertEqual(self.cols[self.name].queries, 0)  # nothing to embed
        self.assertLess(out.index("user: first"), out.index("user: third"))
        self.assertNotIn("outside", out)

    def test_an_empty_window_says_so(self):
        col = self.cols.setdefault(self.name, FakeCol())
        t = datetime.fromisoformat("2026-09-17T15:00:00-07:00").timestamp()
        col.upsert(["xc_1"], ["user: hi\nassistant: hello"],
                   [{"ts": "2026-09-17T15:00:00-07:00", "approx": False, "epoch": t}])
        out = self.mem.search("zzz", caller="alice", window=(t - 7200, t - 3600))
        self.assertIn("among exactly-timed exchanges", out)
        # A window with no `since` starts at epoch 0, which Windows cannot
        # turn into a local time — the label must not try.
        self.assertIn("between the beginning and",
                      self.mem.search("zzz", caller="alice", window=(0.0, t - 3600)))


class TestRecallTool(unittest.TestCase):
    def setUp(self):
        self.seen = {}
        seen = self.seen

        def search(query, caller=None, window=None, exclude_ids=()):
            seen.update(query=query, caller=caller, window=window,
                        exclude_ids=exclude_ids)
            return "From past conversations:\n[9:00am] user: x"
        self.kb_text = ""
        self.ctx = ToolContext(
            memory=SimpleNamespace(search=search,
                                   fetch=lambda ids, caller=None:
                                   seen.update(ids=ids, fetch_caller=caller) or "fetched"),
            kb=SimpleNamespace(search=lambda q, caller=None, focus=None: self.kb_text),
            active_agent="alice", tail_ids=frozenset({"xc_tail"}))

    def test_a_topic_and_window_reach_the_search_with_the_verbatim_tail(self):
        since, until = "2026-09-17T09:00:00-07:00", "2026-09-17T10:00:00-07:00"
        out = dispatch(self.ctx, "recall", {"query": "the list", "since": since,
                                            "until": until})
        self.assertIn("From past conversations", out)
        self.assertEqual(self.seen["window"], window_epochs(since=since, until=until))
        self.assertEqual(self.seen["exclude_ids"], frozenset({"xc_tail"}))
        self.assertEqual(self.seen["caller"], "alice")
        dispatch(self.ctx, "recall", {"query": "the list"})
        self.assertIsNone(self.seen["window"])  # no time words, no window

    def test_a_window_alone_is_enough_and_nothing_at_all_is_not(self):
        dispatch(self.ctx, "recall", {"period": "yesterday"})
        self.assertEqual(self.seen["query"], "")
        self.assertIsNotNone(self.seen["window"])
        self.assertIn("Give recall a topic", dispatch(self.ctx, "recall", {}))

    def test_ids_read_the_exchanges_behind_a_fact(self):
        self.assertEqual(dispatch(self.ctx, "recall", {"ids": ["xc_1", "xc_2"]}), "fetched")
        self.assertEqual((self.seen["ids"], self.seen["fetch_caller"]),
                         (["xc_1", "xc_2"], "alice"))

    def test_reference_passages_are_appended_only_when_there_are_any(self):
        self.assertNotIn("reference material", dispatch(self.ctx, "recall", {"query": "x"}))
        self.kb_text = "[Book, p.3] a passage"
        out = dispatch(self.ctx, "recall", {"query": "x"})
        self.assertIn("From the reference material:\n[Book, p.3]", out)


class TestDateWords(unittest.TestCase):
    def test_periods_and_window_edges(self):
        friday = date(2026, 9, 18)
        self.assertEqual(period_range("last_week", today=friday),
                         ("2026-09-07", "2026-09-13"))
        self.assertEqual(period_range(None, "2026-09-01", None, today=friday),
                         ("2026-09-01", "2026-09-18"))
        self.assertIsNone(window_epochs())
        lo, hi = window_epochs(since="2026-09-17", until="2026-09-17", today=friday)
        self.assertEqual(datetime.fromtimestamp(hi) - datetime.fromtimestamp(lo),
                         timedelta(hours=23, minutes=59, seconds=59))


class TestEveryHatCanRecall(unittest.TestCase):
    def test_own_memory_is_reachable_by_every_persona(self):
        # A hat without the memory tool cannot obey "review your memory" —
        # the exact 2026-07-20 failure. The Background covers the common
        # case now, but the rule stands: every persona must reliably reach
        # its OWN past when asked for more.
        for key, agent in agents.AGENTS.items():
            self.assertIn("recall", agent["tools"], key)


if __name__ == "__main__":
    unittest.main()
