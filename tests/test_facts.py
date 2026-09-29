"""Tests for the fact store (brain/facts.py): what the extractor's operations
write, that a list stays ONE current fact across updates, that a superseded
fact leaves retrieval but not the store, that facts are read by key, and that
learn() withholds the read-mark when the extractor could not run — so the
next boot retries instead of silently forgetting the exchange.

Nothing here loads Chroma, the embedding model, or a model client.
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

import config as cfg
from brain.facts import catch_up, extract, learn
from brain.llm.plain import ask
from stores.chroma_store import Hit
from tests.store_fixtures import facts_over, memory_over

OWNER = "alice"
COL = cfg.agent_facts_collection(OWNER)
T1 = "2026-09-18T09:24:00-07:00"
T2 = "2026-09-18T09:25:30-07:00"


def add(entity, value):
    return {"op": "add", "entity": entity, "value": value}


class TestApply(unittest.TestCase):
    def setUp(self):
        self.cols = {}
        self.facts = facts_over(self.cols)

    def test_add_writes_a_current_fact_with_provenance(self):
        counts = self.facts.apply(OWNER, [add("todo:2026-09-18", "Items: one.")],
                                  xid="xc_1", ts=T1, epoch=1.0)
        self.assertEqual(counts, (1, 0, 0))
        ((fid,), (doc,), (meta,)), = self.cols[COL].upserts
        self.assertEqual(doc, "todo:2026-09-18: Items: one.")
        self.assertEqual((meta["entity"], meta["source_ids"], meta["superseded_by"]),
                         ("todo:2026-09-18", "xc_1", ""))
        self.assertIs(meta["approx"], False)
        self.assertEqual(self.facts.entities(OWNER), {"todo:2026-09-18": T1})

    def test_update_keeps_one_current_fact_and_accumulates_sources(self):
        self.facts.apply(OWNER, [add("todo:2026-09-18", "Items: one.")],
                         xid="xc_1", ts=T1, epoch=1.0)
        (old,) = self.cols[COL].rows
        self.facts.apply(OWNER, [{"op": "update", "id": old, "value": "Items: one, two."}],
                         xid="xc_2", ts=T2, epoch=2.0)
        current = self.facts.current(OWNER, ["todo:2026-09-18"])
        self.assertEqual([h.doc for h in current], ["todo:2026-09-18: Items: one, two."])
        self.assertEqual(current[0].meta["source_ids"], "xc_1,xc_2")
        _, old_meta = self.cols[COL].rows[old]
        self.assertEqual(old_meta["superseded_by"], current[0].id)
        self.assertEqual(old_meta["superseded_epoch"], 2.0)
        self.assertEqual(len(self.cols[COL].rows), 2)  # nothing deleted
        self.assertEqual(self.facts.entities(OWNER), {"todo:2026-09-18": T2})

    def test_supersede_retires_a_fact_without_deleting_it(self):
        self.facts.apply(OWNER, [add("preference:coffee", "Takes it black.")],
                         xid="xc_1", ts=T1, epoch=1.0)
        (fid,) = self.cols[COL].rows
        self.assertEqual(self.facts.apply(OWNER, [{"op": "supersede", "id": fid}],
                                          xid="xc_2", ts=T2, epoch=2.0), (0, 0, 1))
        self.assertEqual(self.facts.current(OWNER, ["preference:coffee"]), [])
        self.assertEqual(self.cols[COL].rows[fid][1]["superseded_by"], "xc_2")
        self.assertEqual(self.facts.entities(OWNER), {})

    def test_retrieval_reads_current_facts_only(self):
        self.facts.apply(OWNER, [add("todo:2026-09-18", "Items: call the dentist.")],
                         xid="xc_1", ts=T1, epoch=1.0)
        (old,) = self.cols[COL].rows
        self.facts.apply(OWNER, [{"op": "update", "id": old,
                                  "value": "Items: call the dentist, buy milk."}],
                         xid="xc_2", ts=T2, epoch=2.0)
        rows = self.facts.query_rows("dentist", owner=OWNER)
        self.assertEqual([h.meta["source_ids"] for h in rows.lexical], ["xc_1,xc_2"])
        self.assertEqual(self.cols[COL].wheres[-1], {"superseded_by": ""})

    def test_an_approximate_exchange_makes_an_approximate_fact(self):
        self.facts.apply(OWNER, [add("person:tom", "Tom is the trading persona.")],
                         xid="xc_1", ts="2026-09-17T09:43:11", approx=True)
        (_, meta), = self.cols[COL].rows.values()
        self.assertIs(meta["approx"], True)
        self.assertNotIn("epoch", meta)


class TestIsolation(unittest.TestCase):
    """The same rule as the exchange index: a persona reads its own facts
    (plus registry grants) and writes only its own; there is no parameter
    through which it can name another persona's collection."""

    def test_reads_and_writes_stay_in_the_callers_own_collection(self):
        cols = {}
        facts = facts_over(cols)
        facts.apply("alice", [add("person:user", "Prefers black coffee.")],
                    xid="xc_1", ts=T1, epoch=1.0)
        facts.query_rows("coffee", owner="tom")
        facts.current("tom", ["person:user"])
        facts.entities("tom")
        self.assertEqual(set(cols), {cfg.agent_facts_collection("alice"),
                                     cfg.agent_facts_collection("tom")})
        self.assertEqual(cols[cfg.agent_facts_collection("tom")].upserts, [])
        self.assertEqual(facts.current("tom", ["person:user"]), [])
        # An update naming a fact Tom cannot see is dropped, not applied.
        (alice_fact,) = cols[cfg.agent_facts_collection("alice")].rows
        self.assertEqual(facts.apply("tom", [{"op": "update", "id": alice_fact,
                                              "value": "x"}],
                                     xid="xc_2", ts=T2, epoch=2.0), (0, 0, 0))


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.known = [Hit(0.1, "todo:2026-09-18: Items: one.",
                          {"entity": "todo:2026-09-18", "ts": T1}, "f_1")]

    def test_operations_are_validated_against_the_candidates(self):
        reply = json.dumps({"ops": [
            {"op": "add", "entity": "Preference: Coffee", "value": " black,  no sugar "},
            {"op": "update", "id": "f_1", "value": "Items: one, two."},
            {"op": "update", "id": "f_unknown", "value": "made up"},
            {"op": "supersede", "id": "f_1"},
            {"op": "add", "entity": "", "value": "no entity"},
            "garbage",
        ]})
        ops = extract(lambda s, p: "Sure — here it is:\n" + reply,
                      "user: x\nassistant: y", T2, self.known, [])
        self.assertEqual(ops, [
            {"op": "add", "entity": "preference:-coffee", "value": "black, no sugar"},
            {"op": "update", "id": "f_1", "value": "Items: one, two."},
            {"op": "supersede", "id": "f_1"},
        ])

    def test_a_list_value_with_real_line_breaks_still_parses(self):
        reply = '{"ops": [{"op": "add", "entity": "todo:2026-09-18",\n'
        reply += ' "value": "Items:\n1) dashboard import\n2) Costa email"}]}'
        (op,) = extract(lambda s, p: reply, "d", T1, [], [])
        self.assertEqual(op["value"], "Items: 1) dashboard import 2) Costa email")

    def test_nothing_durable_is_done_but_no_answer_is_not(self):
        self.assertEqual(extract(lambda s, p: '{"ops": []}', "d", T1, [], []), [])
        self.assertIsNone(extract(lambda s, p: "I cannot help with that.", "d", T1, [], []))

        def boom(s, p):
            raise RuntimeError("timeout")
        self.assertIsNone(extract(boom, "d", T1, [], []))

    def test_the_prompt_shows_the_candidates_and_the_known_keys(self):
        seen = {}

        def ask(system, prompt):
            seen["prompt"] = prompt
            return '{"ops": []}'
        extract(ask, "user: add milk\nassistant: added", T2, self.known,
                ["person:tom"])
        self.assertIn("[f_1] todo:2026-09-18 — Items: one.", seen["prompt"])
        self.assertIn("person:tom", seen["prompt"])
        self.assertIn("user: add milk", seen["prompt"])


class TestPlainAsk(unittest.TestCase):
    def test_a_reply_cut_at_max_tokens_is_an_error_not_a_text(self):
        # Adaptive thinking shares the budget; a cut JSON object once read as
        # "the model gave no ops list" and the exchange's items went missing.
        def client(stop):
            resp = SimpleNamespace(stop_reason=stop,
                                   content=[SimpleNamespace(type="text", text='{"ops": []}')])
            c = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))
            c.with_options = lambda **kw: c
            return c
        self.assertEqual(ask(client("end_turn"), "m", "s", "p", max_tokens=10), '{"ops": []}')
        with self.assertRaises(RuntimeError):
            ask(client("max_tokens"), "m", "s", "p", max_tokens=10)


class TestLearn(unittest.TestCase):
    def setUp(self):
        self.cols = {}
        self.memory = memory_over(self.cols)
        self.facts = facts_over(self.cols)
        msgs = [{"role": "user", "content": "add call the dentist", "ts": T1},
                {"role": "assistant", "content": [{"type": "text", "text": "Added."}]}]
        self.memory.index_exchanges(msgs, OWNER)
        ((self.xid, self.doc, self.meta),) = self.memory.exchanges_of(msgs, OWNER)

    def test_a_read_exchange_is_marked_and_its_facts_land(self):
        reply = json.dumps({"ops": [add("todo:2026-09-18", "Items: call the dentist.")]})
        self.assertEqual(learn(lambda s, p: reply, self.memory, self.facts, OWNER,
                               self.xid, self.doc, self.meta), (1, 0, 0))
        self.assertEqual(self.memory.unextracted(OWNER), [])
        (h,) = self.facts.current(OWNER, ["todo:2026-09-18"])
        self.assertEqual((h.meta["source_ids"], h.meta["ts"]), (self.xid, T1))

    def test_a_failed_extraction_leaves_the_exchange_for_the_next_pass(self):
        def boom(s, p):
            raise RuntimeError("api down")
        self.assertIsNone(learn(boom, self.memory, self.facts, OWNER,
                                self.xid, self.doc, self.meta))
        self.assertEqual([x for x, _, _ in self.memory.unextracted(OWNER)], [self.xid])
        self.assertEqual(getattr(self.cols.get(COL), "upserts", []), [])  # nothing written

    def test_the_extractor_sees_the_list_being_built_however_unlike_the_new_item(self):
        # Item six, "meditate with an eyeshade", shares no words with a list
        # about dashboards and email; the search on the exchange text never
        # surfaced the list and item six became a second fact (2026-09-18).
        self.facts.apply(OWNER, [add("todo:2026-09-18", "Items: dashboard import; "
                                                         "Costa email in Thunderbird.")],
                         xid="xc_0", ts=T1, epoch=1.0)
        (fid,) = self.cols[COL].rows
        seen = {}

        def ask(system, prompt):
            seen["prompt"] = prompt
            return '{"ops": []}'
        learn(ask, self.memory, self.facts, OWNER, self.xid,
              "user: allocate a session to meditate with an eyeshade\nassistant: Sixth item.",
              self.meta)
        self.assertIn(f"[{fid}] todo:2026-09-18", seen["prompt"])
        self.assertIn("(the assistant is Alice)", seen["prompt"])  # never the user's name

    def test_catch_up_reads_oldest_first(self):
        later = [{"role": "user", "content": "and buy milk", "ts": T2},
                 {"role": "assistant", "content": [{"type": "text", "text": "Ok."}]}]
        self.memory.index_exchanges(later, OWNER)
        order = []

        def ask(system, prompt):
            order.append(prompt.split("\n", 1)[0])
            return '{"ops": []}'
        with mock.patch.object(cfg, "FACTS_CANDIDATES", 12):
            self.assertEqual(catch_up(ask, self.memory, self.facts, OWNER), (2, 0))
        self.assertEqual(order, [f"Exchange at {T1} (the assistant is Alice):",
                                 f"Exchange at {T2} (the assistant is Alice):"])
        self.assertEqual(self.memory.unextracted(OWNER), [])
        self.assertEqual(catch_up(ask, self.memory, self.facts, OWNER), (0, 0))


if __name__ == "__main__":
    unittest.main()
