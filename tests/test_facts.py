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
from brain import agents
from brain.facts import catch_up, extract, learn, skipped
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


class TestTradeStateIsNotAFact(unittest.TestCase):
    """Tom's position state belongs to the trade log: kept as facts it went
    stale and wrong ("the 7650 leg stays open", 2026-10-19) and was fed back
    as memory. Dropped on write whatever the extractor says, and on read."""

    HOLDING = add("holding:blue-diagonal-legs", "Short the Oct 22 7650 put.")
    NICKNAME = add("account:blue", "The user calls account 839 'blue'.")

    def setUp(self):
        self.cols = {}
        self.memory = memory_over(self.cols)
        self.facts = facts_over(self.cols)

    def test_learn_gives_tom_the_rule_and_drops_trade_state(self):
        msgs = [{"role": "user", "content": "what's open on blue", "ts": T1},
                {"role": "assistant", "content": [{"type": "text", "text": "x"}]}]
        self.memory.index_exchanges(msgs, "tom")
        ((xid, doc, meta),) = self.memory.exchanges_of(msgs, "tom")
        seen = {}

        def ask(system, prompt):
            seen["system"] = system
            return json.dumps({"ops": [self.HOLDING, self.NICKNAME]})
        self.assertEqual(learn(ask, self.memory, self.facts, "tom", xid, doc, meta),
                         (1, 0, 0))
        self.assertIn(agents.TRADE_STATE_RULE, seen["system"])
        self.assertEqual(list(self.facts.entities("tom")), ["account:blue"])

    def test_a_fact_naming_a_contract_is_dropped_under_any_key(self):
        """The 2026-10-22 rebuild refiled position state as "matter" and
        "plan": the text, not the key, decides."""
        dropped = (
            "Short one Oct 21 7660 put, sold to open at a 15.70 credit.",
            "The Oct 16 vertical is long the 7650/7620 puts; the leg stays open.",
            "The user holds a put calendar at the 7700 strike in account 868.",
            "Rolled the short leg from 7720P to 7790P.",
            "Decided to roll the short Oct 15 7760 put down.")
        kept = (
            "839 is 'Blue', 868 is 'Red', and 699 is 'IRA'.",
            "The user calls account 839 'blue'; the user calls 868 red.",
            "The user wants the VIX level with every morning check-in.",
            "Act on the campaign above 7,770 on SPX.",
            "The user prefers puts over calls for the quick-cycle campaign.",
            "Short puts expiring 2026-10-23 are reviewed the day before.")
        tom = {"agent": "tom", "entity": "matter:x"}
        self.assertEqual([d for d in dropped if not skipped(tom, d)], [])
        self.assertEqual([d for d in kept if skipped(tom, d)], [])
        self.assertFalse(skipped({"agent": "alice", "entity": "matter:x"}, dropped[0]))
        self.facts.apply("tom", [add("matter:blue-7660", dropped[0]), self.NICKNAME],
                         xid="xc_1", ts=T1, epoch=1.0)
        self.assertEqual(list(self.facts.entities("tom")), ["account:blue"])

    def test_old_trade_state_facts_stop_reaching_tom(self):
        with mock.patch.dict(agents.AGENTS["tom"], {"fact_skip": (),
                                                    "fact_skip_text": ""}):
            self.facts.apply("tom", [self.HOLDING, self.NICKNAME],
                             xid="xc_1", ts=T1, epoch=1.0)     # written before the rule
        rows = self.facts.query_rows("blue 7650 put", owner="tom")
        self.assertEqual({h.meta["entity"] for h in rows.dense + rows.lexical},
                         {"account:blue"})
        self.assertEqual(self.facts.current("tom", ["holding:blue-diagonal-legs"]), [])
        # Alice keeps every kind: the rule is Tom's alone.
        self.facts.apply("alice", [self.HOLDING], xid="xc_2", ts=T2, epoch=2.0)
        self.assertEqual(len(self.facts.current("alice", ["holding:blue-diagonal-legs"])), 1)


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
