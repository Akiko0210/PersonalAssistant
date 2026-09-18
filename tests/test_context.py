"""Tests for the per-turn Background (brain/context.py) and its seams into
converse(): what the model is shown about the past is chosen by relevance
and recency, never by a fixed window — and a retrieval failure costs a turn
nothing but its Background.

The fused scorer is pinned here because its thresholds are tuned from the
context log; a silent change to the formula would make those tunings lie.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import config as cfg
from brain import context
from brain.memory import Rows, exchange_id
from stores import chroma_store
from stores.chroma_store import Hit
from tests.llm_fixtures import (make_claude, system_text, text_reply,
                                tool_reply)

H = 3600.0
NOW = 1_800_000_000.0  # any fixed epoch; ages are relative to it


def dense(id_, doc, sim, age_h=0.0, vec=None, **meta):
    """A dense hit at cosine `sim` (the store is cosine space: distance is
    1 - cos) aged `age_h`, optionally carrying its stored vector."""
    return Hit(1 - sim, doc, {"epoch": NOW - age_h * H, **meta}, id_, vec)


def lexical(id_, doc, bm25, age_h=0.0, **meta):
    return Hit(bm25, doc, {"epoch": NOW - age_h * H, **meta}, id_)


def fake_memory(dense_hits=(), lexical_hits=(), error=None):
    calls = []

    def query_rows(query, n, caller, **kwargs):
        calls.append((query, n, caller))
        return Rows(list(dense_hits), list(lexical_hits), 1, error)
    return SimpleNamespace(query_rows=query_rows, calls=calls,
                           index_exchanges=lambda *a, **k: 0,
                           backfill_keys=lambda *a, **k: (0, 0))


def fake_kb(hits=()):
    return SimpleNamespace(query_rows=lambda *a, **k: list(hits))


class TestScoring(unittest.TestCase):
    def test_similarity_follows_the_collections_space(self):
        # chromadb gives a sentence-transformer collection cosine space, so
        # d = 1 - cos. The old squared-L2 formula turned every distance into
        # 0.5 + cos/2 and the relevance gate never fired (2026-09-18).
        self.assertEqual(chroma_store.SPACE, "cosine")
        self.assertAlmostEqual(context.similarity(0), 1.0)
        self.assertAlmostEqual(context.similarity(0.5), 0.5)
        self.assertAlmostEqual(context.similarity(1), 0.0)
        self.assertEqual(context.similarity(1.1), 0.0)  # HNSW drift clamps
        self.assertAlmostEqual(context.similarity(1, space="l2"), 0.5)

    def test_recency_halves_at_the_half_life_and_is_zero_when_unknown(self):
        with mock.patch.object(cfg, "CONTEXT_RECENCY_HALF_LIFE_H", 168):
            self.assertAlmostEqual(context.recency(0), 1.0)
            self.assertAlmostEqual(context.recency(168), 0.5)
            self.assertEqual(context.recency(None), 0.0)

    def test_legacy_summaries_age_by_their_date(self):
        # A summary covers its day, so it is dated at that day's end; an
        # unparseable date is unknown, never "now".
        age = context.age_hours({"date": "2026-07-01 to 2026-07-03"},
                                now=NOW)
        self.assertIsNotNone(age)
        self.assertIsNone(context.age_hours({"date": "sometime"}, now=NOW))


class TestFuse(unittest.TestCase):
    def ranked(self, dense_hits=(), lexical_hits=(), exclude=()):
        return context.fuse(list(dense_hits), list(lexical_hits), now=NOW,
                            exclude_ids=exclude)

    def test_old_but_relevant_beats_recent_but_mediocre(self):
        out = self.ranked([dense("old", "old relevant", 0.7, age_h=30 * 24),
                           dense("new", "new so-so", 0.4, age_h=1)])
        self.assertEqual([c.id for c in out], ["old", "new"])
        self.assertTrue(all(c.gate for c in out))

    def test_recency_reorders_equally_relevant_hits(self):
        out = self.ranked([dense("old", "x", 0.6, age_h=30 * 24),
                           dense("new", "x", 0.6, age_h=1)])
        self.assertEqual([c.id for c in out], ["new", "old"])

    def test_irrelevant_but_recent_fails_the_gate(self):
        (c,) = self.ranked([dense("junk", "unrelated", 0.1, age_h=0.1)])
        self.assertFalse(c.gate)

    def test_an_exact_lexical_hit_passes_and_outranks_a_weak_dense_one(self):
        # The point of the lexical list: "SPX 5800" matches even when the
        # embeddings shrug.
        out = self.ranked([dense("weak", "some chat", 0.3)],
                          [lexical("spx", "user: the SPX 5800 put", bm25=4.2)])
        self.assertEqual(out[0].id, "spx")
        self.assertTrue(out[0].gate)
        self.assertAlmostEqual(out[0].lex, 1.0)

    def test_excluded_ids_never_appear(self):
        out = self.ranked([dense("tail", "already verbatim", 0.9)],
                          [lexical("tail", "already verbatim", 3.0)],
                          exclude={"tail"})
        self.assertEqual(out, [])


class TestBudget(unittest.TestCase):
    def pull(self, memory, kb=None, **overrides):
        with mock.patch.multiple(cfg, **overrides):
            return context.build_context("q", owner="alice", memory=memory,
                                         kb=kb or fake_kb(), now=NOW)

    def test_an_oversize_document_is_skipped_not_truncated_to_fit(self):
        pull = self.pull(fake_memory([dense("big", "x" * 150, 0.9),
                                      dense("small", "y" * 50, 0.5)]),
                         CONTEXT_CONVO_CHARS=100, CONTEXT_HIT_CHARS=800)
        kept = {c.id: c.kept for c in pull.candidates}
        self.assertEqual(kept, {"big": False, "small": True})
        self.assertIn("y" * 50, pull.block)

    def test_knowledge_takes_the_conversation_budgets_leftover(self):
        chunk = Hit(0.2, "z" * 60, {"title": "Book", "page": 3}, "kb1")
        pull = self.pull(fake_memory([dense("c", "w" * 40, 0.9)]), fake_kb([chunk]),
                         CONTEXT_CONVO_CHARS=100, CONTEXT_KB_CHARS=10)
        self.assertTrue(all(c.kept for c in pull.candidates))
        self.assertIn('cite="Book, p.3"', pull.block)

    def test_a_failing_memory_store_still_yields_the_knowledge_half(self):
        def boom(*a, **k):
            raise RuntimeError("hnsw segment reader")
        chunk = Hit(0.2, "reference text", {"title": "Book"}, "kb1")
        pull = context.build_context("q", owner="alice",
                                     memory=SimpleNamespace(query_rows=boom),
                                     kb=fake_kb([chunk]), now=NOW)
        self.assertIn("reference text", pull.block)

    def test_nothing_relevant_means_no_block_at_all(self):
        pull = context.build_context("q", owner="alice", kb=fake_kb(), now=NOW,
                                     memory=fake_memory([dense("junk", "meh", 0.05)]))
        self.assertEqual(pull.block, "")
        self.assertEqual([c.note for c in pull.candidates], ["gate"])


class TestConverseSeams(unittest.TestCase):
    def test_system_is_a_cached_static_block_plus_the_background(self):
        c = make_claude()
        c.memory = fake_memory([dense("m1", "user: my cat is Mochi\n"
                                            "assistant: noted", 0.8)])
        c.converse("what is my cat called?")
        system = c.client.messages.calls[0]["system"]
        self.assertEqual(len(system), 2)
        self.assertIn("cache_control", system[0])
        self.assertIn("You are Alice", system[0]["text"])
        self.assertNotIn("cache_control", system[1])
        self.assertIn(context.HEADER, system[1]["text"])
        self.assertIn("Mochi", system[1]["text"])

    def test_no_background_means_a_single_block(self):
        c = make_claude()
        c.converse("hello")
        self.assertEqual(len(c.client.messages.calls[0]["system"]), 1)
        self.assertIn("You are Alice", system_text(c.client.messages.calls[0]))

    def test_only_the_recent_tail_is_sent_and_the_whole_turn_stays_inside(self):
        c = make_claude([tool_reply("get_current_time", {}), text_reply("ok")])
        for i in range(3):
            c.history += [{"role": "user", "content": f"q{i}", "ts": ""},
                          {"role": "assistant",
                           "content": [{"type": "text", "text": f"a{i}"}]}]
        with mock.patch.object(cfg, "CONTEXT_RECENT_EXCHANGES", 2):
            c.converse("now")
        first, second = c.client.messages.calls
        self.assertEqual(first["messages"][0]["content"], "q1")  # q0 fell off
        self.assertEqual(len(first["messages"]), 5)
        # Round two carries the tool_use/tool_result pair from the same tail.
        self.assertEqual(second["messages"][0]["content"], "q1")
        self.assertEqual(second["messages"][-2]["content"][0]["type"], "tool_use")
        self.assertEqual(second["messages"][-1]["content"][0]["type"], "tool_result")

    def test_a_retrieval_failure_never_costs_the_reply(self):
        def boom(*a, **k):
            raise RuntimeError("index gone")
        c = make_claude([text_reply("still here")])
        c.memory = SimpleNamespace(query_rows=boom, index_exchanges=boom)
        self.assertEqual(c.converse("hi"), "still here")

    def test_short_follow_ups_borrow_the_previous_user_turn(self):
        c = make_claude(history=[
            {"role": "user", "content": "how did the SPX butterfly do", "ts": ""},
            {"role": "assistant", "content": [{"type": "text", "text": "fine"}]}])
        c.memory = fake_memory()
        c.converse("and the other one?")
        c.converse("please tell me everything about the trades we placed this week")
        queries = [q for q, _, _ in c.memory.calls]
        self.assertEqual(queries[0], "how did the SPX butterfly do and the other one?")
        self.assertTrue(queries[1].startswith("please tell me everything"))

    def test_the_pull_is_scoped_to_the_active_persona(self):
        c = make_claude(active="tom")
        c.memory = fake_memory()
        c.converse("hello")
        self.assertEqual(c.memory.calls[0][2], "tom")

    def test_the_exchange_is_indexed_after_the_turn(self):
        c = make_claude([text_reply("noted")])
        written = []
        c.memory = SimpleNamespace(
            query_rows=lambda *a, **k: Rows([], [], 0, None),
            index_exchanges=lambda msgs, owner, skip_existing=True:
                written.append((msgs, owner, skip_existing)) or 1)
        c.converse("remember this")
        (msgs, owner, skip), = written
        self.assertEqual(owner, "alice")
        self.assertFalse(skip)
        self.assertEqual(msgs[0]["content"], "remember this")


class TestRender(unittest.TestCase):
    def test_an_approximate_stamp_reads_on_or_before_with_no_age(self):
        # The old staging file stamped text at flush time; the model once
        # read such a stamp as "9:43am today" for words from the previous
        # morning (2026-09-17 12:01).
        exact = dense("x", "user: a\nassistant: b", 0.9, age_h=2,
                      ts="2026-09-17T09:43:11-07:00", approx=False)
        approx = dense("y", "user: I\nassistant: morning", 0.9, age_h=2,
                       ts="2026-09-17T09:43:11", approx=True)
        pull = context.build_context("q", owner="alice", kb=fake_kb(), now=NOW,
                                     memory=fake_memory([exact, approx]))
        self.assertIn('when="9:43am 9/17/2026" age="2 hours ago"', pull.block)
        self.assertIn('when="on or before 9:43am 9/17/2026">', pull.block)


class TestRedundancy(unittest.TestCase):
    def test_the_fill_penalises_a_near_duplicate_of_what_it_already_kept(self):
        # Nine exchanges *about* keeping a list once filled the budget ahead
        # of the list itself (2026-09-17). Two near-duplicates at the top, a
        # distinct exchange below them, and room for two.
        a = dense("dup1", "user: are you keeping my list?\nassistant: yes", 0.60,
                  vec=[1.0, 0.0, 0.0])
        b = dense("dup2", "user: is my list going?\nassistant: it is", 0.58,
                  vec=[0.99, 0.14, 0.0])
        c = dense("list", "user: item two\nassistant: Your list now: one, two", 0.50,
                  vec=[0.0, 1.0, 0.0])

        def kept(lam):
            with mock.patch.multiple(cfg, CONTEXT_MMR_LAMBDA=lam,
                                     CONTEXT_CONVO_CHARS=95, CONTEXT_HIT_CHARS=800):
                pull = context.build_context("q", owner="alice", kb=fake_kb(), now=NOW,
                                             memory=fake_memory([a, b, c]))
            return [x.id for x in pull.candidates if x.kept]
        self.assertEqual(kept(1.0), ["dup1", "dup2"])  # plain score order
        self.assertEqual(kept(0.7), ["dup1", "list"])  # the twin pays for its twin


class TestTailSharing(unittest.TestCase):
    def test_the_verbatim_tail_is_shared_with_the_search_tool(self):
        c = make_claude(history=[
            {"role": "user", "content": "q0", "ts": "2026-09-17T09:00:00-07:00"},
            {"role": "assistant", "content": [{"type": "text", "text": "a0"}]}])
        c.memory = fake_memory()
        c.converse("now")
        self.assertEqual(c._ctx.tail_ids, {exchange_id("alice", c.history[0])})


if __name__ == "__main__":
    unittest.main()
