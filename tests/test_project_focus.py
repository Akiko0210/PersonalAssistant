"""Project focus (#63): one open record lifted out of a persona's memory on the
user's say-so. The promises pinned here are the ones the design rests on: focus
adds to retrieval and never takes from it, it survives a restart and says so,
it can't create a duplicate record, and a deleted project's memory stays.

The label search against real Chroma, and the 2026-11-02 conversation itself,
were checked by replay (see the PR); nothing here loads Chroma or a model.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import config as cfg
from brain import agents, context
from brain.facts import learn
from brain.history import now_iso
from brain.llm.main import Claude
from brain.memory import Rows, exchange_id
from brain.query import Query
from stores import working_state as ws
from stores.chroma_store import Hit
from tests.llm_fixtures import make_claude, system_text, text_reply, tool_reply
from tests.store_fixtures import facts_over, memory_over
from tests.test_working_state import RecordBase

NOW = 1_800_000_000.0
PROJECT = {"id": 29, "title": "Rooftop Butterfly"}


def user(text="did we save it", **extra):
    return {"role": "user", "content": text, "ts": "2026-11-02T14:34:39-08:00", **extra}


def assistant(text="Yes."):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def hit(id_, doc, sim=0.8, **meta):
    return Hit(1 - sim, doc, {"epoch": NOW, **meta}, id_)


class TestStore(RecordBase):
    def setUp(self):
        super().setUp()
        self.a = ws.create("tom", "Rooftop Butterfly", kind="strategy", body="Enter at 35 DTE.")
        self.b = ws.create("tom", "Blue diagonal", kind="campaign", body="Roll at 21 DTE.")

    def test_focus_is_one_slot_and_moves(self):
        self.assertIsNone(ws.focused("tom"))
        ws.focus_on("tom", self.a)
        ws.focus_on("tom", self.b)
        self.assertEqual(ws.focused("tom")["id"], self.b)
        self.assertEqual(ws.leave_focus("tom")["id"], self.b)
        self.assertIsNone(ws.focused("tom"))
        self.assertIsNone(ws.leave_focus("tom"))

    def test_focus_belongs_to_one_persona(self):
        ws.focus_on("tom", self.a)
        self.assertIsNone(ws.focused("alice"))
        with self.assertRaises(ValueError):
            ws.focus_on("alice", self.a)

    def test_a_closed_project_cant_be_entered_and_closing_ends_focus(self):
        ws.focus_on("tom", self.a)
        ws.close("tom", self.a, "Deleted by the user")
        self.assertIsNone(ws.focused("tom"))
        with self.assertRaises(ValueError) as e:
            ws.focus_on("tom", self.a)
        self.assertIn("closed", str(e.exception))
        self.assertIn("Enter at 35 DTE", ws.render("tom", self.a))  # still readable

    def test_closing_another_record_leaves_focus_alone(self):
        ws.focus_on("tom", self.a)
        ws.close("tom", self.b, "done")
        self.assertEqual(ws.focused("tom")["id"], self.a)

    def test_the_block_says_what_is_in_focus(self):
        self.assertNotIn("In focus", ws.block("tom"))
        ws.focus_on("tom", self.a)
        self.assertIn(f"In focus: #{self.a} Rooftop Butterfly", ws.block("tom"))

    def test_no_focus_leaves_the_block_byte_for_byte(self):
        before = ws.block("tom")
        ws.focus_on("tom", self.a)
        ws.leave_focus("tom")
        self.assertEqual(ws.block("tom"), before)

    def test_the_focused_record_gets_its_whole_journal_ahead_of_others(self):
        for i in range(12):
            ws.add_entry("tom", self.a, f"rooftop decision {i} " + "x" * 300)
        ws.add_entry("tom", self.b, "newest diagonal news")  # newest, so it leads without focus
        cfg.WORKING_RECORD_CHARS = 3000
        self.assertNotIn("rooftop decision 0 ", ws.block("tom"))
        ws.focus_on("tom", self.a)
        shown = ws.block("tom")
        self.assertIn("Enter at 35 DTE", shown)
        self.assertIn("rooftop decision 11 ", shown)
        self.assertIn("Blue diagonal", shown)  # the other's title always survives
        self.assertLessEqual(len(shown), 3100)


class TestTools(RecordBase):
    def setUp(self):
        super().setUp()
        self.a = ws.create("tom", "Rooftop Butterfly", body="Enter at 35 DTE.")

    def test_entering_returns_the_record_since_this_turns_background_is_old(self):
        out = self.call("focus_project", id=self.a)
        self.assertIn("tell the user", out)
        self.assertIn("Enter at 35 DTE", out)
        self.assertEqual(ws.focused("tom")["id"], self.a)

    def test_no_id_leaves_and_says_what_was_left(self):
        self.call("focus_project", id=self.a)
        self.assertIn("Left focus on Rooftop Butterfly", self.call("focus_project"))
        self.assertIn("No project was in focus", self.call("focus_project"))

    def test_bad_targets_are_sentences_and_change_nothing(self):
        self.call("focus_project", id=self.a)
        self.assertIn("no record #999", self.call("focus_project", id=999))
        self.assertEqual(ws.focused("tom")["id"], self.a)
        self.assertIn("Which record", self.call("focus_project", id="rooftop"))
        self.assertEqual(ws.focused("tom")["id"], self.a)

    def test_an_id_less_write_in_focus_cant_make_a_duplicate(self):
        # The 2026-11-01 failure: same rules, second record, other name.
        self.call("focus_project", id=self.a)
        out = self.call("update_record", title="SPX Campaign Entry Rules", body="35 DTE")
        self.assertIn(f"pass id {self.a}", out)
        self.assertEqual(len(ws.listing("tom")), 1)
        self.assertIn("Enter at 35 DTE", ws.render("tom", self.a))  # not overwritten

    def test_writes_with_the_id_still_work_and_leaving_allows_new_ones(self):
        self.call("focus_project", id=self.a)
        self.assertIn("Saved", self.call("update_record", id=self.a, body="Enter at 40 DTE."))
        self.call("focus_project")
        self.assertIn("Saved", self.call("update_record", title="Something separate"))
        self.assertEqual(len(ws.listing("tom")), 2)

    def test_deleting_the_focused_project_says_focus_ended(self):
        self.call("focus_project", id=self.a)
        self.assertIn("Focus on it has ended",
                      self.call("close_record", id=self.a, outcome="Deleted by the user"))
        self.assertIsNone(ws.focused("tom"))
        self.assertIn("closed", self.call("focus_project", id=self.a))

    def test_a_refused_delete_doesnt_claim_focus_ended(self):
        self.call("focus_project", id=self.a)
        out = self.call("close_record", id=self.a, outcome="  ")
        self.assertNotIn("Focus on it has ended", out)
        self.assertEqual(ws.focused("tom")["id"], self.a)

    def test_only_tom_has_the_tool(self):
        self.assertIn("focus_project", agents.AGENTS["tom"]["tools"])
        for other in set(agents.AGENTS) - {"tom"}:
            self.assertNotIn("focus_project", agents.AGENTS[other]["tools"])


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.cols = {}
        self.mem = memory_over(self.cols)
        self.facts = facts_over(self.cols)

    def test_an_exchange_and_its_companion_carry_the_label(self):
        self.mem.index_exchanges([user(project=29), assistant()], "tom")
        ((_, _, metas),) = self.cols[cfg.agent_memory_collection("tom")].upserts
        self.assertEqual([m["project"] for m in metas], [29, 29])

    def test_an_unfocused_exchange_has_no_label_key(self):
        self.mem.index_exchanges([user(), assistant()], "tom")
        ((_, _, metas),) = self.cols[cfg.agent_memory_collection("tom")].upserts
        self.assertTrue(all("project" not in m for m in metas))  # Chroma rejects None

    def test_the_id_doesnt_depend_on_the_label(self):
        # Or a re-index after labeling would duplicate the exchange.
        self.assertEqual(exchange_id("tom", user()), exchange_id("tom", user(project=29)))

    def test_a_project_search_reads_only_its_label_and_skips_the_legacy_archive(self):
        self.mem.index_exchanges([user("rooftop save", project=29), assistant(),
                                  user("rooftop other"), assistant()], "tom")
        rows = self.mem.query_rows("rooftop", 8, "tom", project=29)
        self.assertEqual({h.meta["project"] for h in rows.lexical}, {29})
        name = cfg.agent_memory_collection("tom")
        self.assertEqual(self.cols[name].wheres[-1], {"project": 29})
        self.assertNotIn(cfg.MEMORY_COLLECTION, self.cols)

    def test_an_ordinary_search_is_untouched(self):
        self.mem.index_exchanges([user("rooftop save", project=29), assistant(),
                                  user("rooftop other"), assistant()], "tom")
        rows = self.mem.query_rows("rooftop", 8, "tom")
        self.assertEqual(len(rows.lexical), 2)
        self.assertIn(cfg.MEMORY_COLLECTION, self.cols)
        self.assertIsNone(self.cols[cfg.agent_memory_collection("tom")].wheres[-1])

    def _fact(self):
        col = self.cols[cfg.agent_facts_collection("tom")]
        current = [(i, m) for i, (d, m) in col.rows.items() if m["superseded_by"] == ""]
        self.assertEqual(len(current), 1)
        return current[0]

    def test_a_new_fact_takes_the_exchanges_label(self):
        self.facts.apply("tom", [{"op": "add", "entity": "plan:x", "value": "Enter at 35."}],
                         xid="x1", ts="t", epoch=1.0, project=29)
        self.assertEqual(self._fact()[1]["project"], 29)

    def test_an_update_outside_focus_keeps_the_label_and_inside_another_takes_it(self):
        self.facts.apply("tom", [{"op": "add", "entity": "plan:x", "value": "Enter at 35."}],
                         xid="x1", ts="t", epoch=1.0, project=29)
        old = self._fact()[0]
        self.facts.apply("tom", [{"op": "update", "id": old, "value": "Enter at 40."}],
                         xid="x2", ts="t", epoch=2.0)
        fid, meta = self._fact()
        self.assertEqual(meta["project"], 29)
        self.facts.apply("tom", [{"op": "update", "id": fid, "value": "Enter at 45."}],
                         xid="x3", ts="t", epoch=3.0, project=30)
        self.assertEqual(self._fact()[1]["project"], 30)

    def test_learning_passes_the_label_through(self):
        ask = lambda s, p: '{"ops": [{"op": "add", "entity": "plan:x", "value": "Enter at 35."}]}'
        self.mem.index_exchanges([user(project=29), assistant()], "tom")
        ((xid, doc, meta),) = self.mem.exchanges_of([user(project=29), assistant()], "tom")
        learn(ask, self.mem, self.facts, "tom", xid, doc, meta)
        self.assertEqual(self._fact()[1]["project"], 29)

    def test_a_fact_search_keeps_only_the_labeled_and_current(self):
        self.facts.apply("tom", [{"op": "add", "entity": "plan:a", "value": "rooftop one"},
                                 {"op": "add", "entity": "plan:b", "value": "rooftop two"}],
                         xid="x1", ts="t", epoch=1.0, project=29)
        self.facts.apply("tom", [{"op": "add", "entity": "plan:c", "value": "rooftop three"}],
                         xid="x2", ts="t", epoch=2.0)
        rows = self.facts.query_rows("rooftop", 8, "tom", project=29)
        self.assertEqual(len(rows.lexical), 2)
        self.assertEqual(len(self.facts.query_rows("rooftop", 8, "tom").lexical), 3)


class FakeMemory:
    """query_rows answering the ordinary and the project search differently."""

    def __init__(self, ordinary=(), own=(), fail_own=False):
        self.ordinary, self.own, self.fail_own = list(ordinary), list(own), fail_own
        self.calls = []

    def query_rows(self, query, n, caller, **kw):
        self.calls.append((query, kw))
        if kw.get("project") is None:
            return Rows(self.ordinary, [], 1, None)
        if self.fail_own:
            raise RuntimeError("segment reader")
        return Rows(self.own, [], 1, None)


KB = SimpleNamespace(query_rows=lambda *a, **k: [])


class TestBackground(unittest.TestCase):
    def pull(self, memory, query="did we save it", project=PROJECT, **kw):
        return context.build_context(query, owner="tom", memory=memory, kb=KB, now=NOW,
                                     project=project, **kw)

    def kept(self, pull):
        return [c.id for c in pull.candidates if c.kept]

    def test_no_focus_makes_no_second_search(self):
        memory = FakeMemory([hit("o1", "user: a")], [hit("p1", "user: b")])
        self.assertEqual(self.kept(self.pull(memory, project=None)), ["o1"])
        self.assertEqual(memory.calls, [("did we save it", {"window": None})])

    def test_it_finds_what_the_ordinary_search_misses_and_lists_it_first(self):
        # The save on 2026-11-02 was not an ordinary candidate at all.
        pull = self.pull(FakeMemory([hit("o1", "user: other")], [hit("p1", "user: the save")]))
        self.assertEqual(self.kept(pull), ["p1", "o1"])
        self.assertLess(pull.block.index("the save"), pull.block.index("other"))

    def test_the_search_uses_the_projects_title_and_label(self):
        memory = FakeMemory()
        self.pull(memory)
        self.assertEqual(memory.calls[1][0], "Rooftop Butterfly did we save it")
        self.assertEqual(memory.calls[1][1]["project"], 29)
        self.assertEqual(memory.calls[0][0], "did we save it")  # ordinary words untouched

    def test_everything_the_ordinary_pull_keeps_is_still_kept(self):
        # Even with a budget's worth of project memory competing for space.
        ordinary = [hit(f"o{i}", f"user: ordinary {i} " + "x" * 700, sim=0.3) for i in range(8)]
        own = [hit(f"p{i}", f"user: project {i} " + "y" * 700, sim=0.9) for i in range(10)]
        alone = self.kept(self.pull(FakeMemory(ordinary), project=None))
        focused = self.kept(self.pull(FakeMemory(ordinary, own)))
        self.assertTrue(alone)
        self.assertTrue(set(alone) <= set(focused))

    def test_the_addition_stays_inside_its_budget(self):
        own = [hit(f"p{i}", "user: " + "y" * 700) for i in range(20)]
        pull = self.pull(FakeMemory([], own))
        chars = sum(c.chars for c in pull.candidates if c.kept)
        self.assertLessEqual(chars, cfg.CONTEXT_PROJECT_CHARS)
        self.assertGreater(chars, 0)

    def test_a_memory_found_both_ways_is_shown_once(self):
        both = hit("x", "user: the save", project=29)
        pull = self.pull(FakeMemory([both], [both]))
        self.assertEqual(self.kept(pull), ["x"])
        self.assertEqual(pull.block.count("the save"), 1)

    def test_the_verbatim_tail_stays_out(self):
        memory = FakeMemory([], [hit("tail", "user: said just now"), hit("p1", "user: older")])
        self.assertEqual(self.kept(self.pull(memory, exclude_ids={"tail"})), ["p1"])
        self.assertEqual(memory.calls[1][0], "Rooftop Butterfly did we save it")

    def test_a_failed_project_read_costs_only_the_addition(self):
        memory = FakeMemory([hit("o1", "user: other")], fail_own=True)
        self.assertEqual(self.kept(self.pull(memory)), ["o1"])

    def test_labeled_facts_are_added_too(self):
        facts = SimpleNamespace(
            current=lambda *a: [],
            query_rows=lambda q, n, owner, window=None, project=None: Rows(
                [hit("f1", "plan:x: Enter at 35.", project=29)] if project else [], [], 1, None))
        pull = self.pull(FakeMemory(), facts=facts)
        self.assertEqual(self.kept(pull), ["f1"])

    def test_a_time_window_reaches_the_project_search_and_reads_in_order(self):
        window = (NOW - 100, NOW)
        own = [Hit(0.5, "user: later", {"epoch": NOW}, "p2"),
               Hit(0.5, "user: earlier", {"epoch": NOW - 50}, "p1")]
        memory = FakeMemory([], own)
        pull = self.pull(memory, query=Query(text="q", window=window))
        self.assertEqual(memory.calls[1][1]["window"], window)
        self.assertEqual(self.kept(pull), ["p1", "p2"])


class TestConverse(RecordBase):
    """The real store under a scripted model: the path a voice turn takes."""

    def claude(self, responses=None, **kw):
        c = make_claude(responses, active="tom", **kw)
        c._focused_project = Claude._focused_project.__get__(c)
        c._record_block = Claude._record_block.__get__(c)
        return c

    def setUp(self):
        super().setUp()
        self.rid = ws.create("tom", "Rooftop Butterfly", body="Enter at 35 DTE.")

    def test_entering_labels_that_turn_and_leaving_doesnt_label_that_one(self):
        c = self.claude([tool_reply("focus_project", {"id": self.rid}), text_reply("Focused."),
                         text_reply("Noted."),
                         tool_reply("focus_project", {}), text_reply("Back to normal."),
                         text_reply("Hi.")])
        for said in ("work on the rooftop", "enter at 40", "back to normal", "hello"):
            c.converse(said)
        labels = {m["content"]: m.get("project") for m in c.history
                  if m["role"] == "user" and isinstance(m["content"], str)}
        self.assertEqual(labels, {"work on the rooftop": self.rid, "enter at 40": self.rid,
                                  "back to normal": None, "hello": None})

    def test_the_label_never_reaches_the_api(self):
        ws.focus_on("tom", self.rid)
        c = self.claude()
        c.converse("one")
        c.converse("two")
        for call in c.client.messages.calls:
            for m in call["messages"]:
                self.assertEqual(set(m), {"role", "content"})

    def test_the_prompt_says_what_is_in_focus(self):
        ws.focus_on("tom", self.rid)
        c = self.claude()
        c.converse("hello")
        self.assertIn("In focus: #", system_text(c.client.messages.calls[0]))

    def test_a_turn_in_focus_reaches_the_index_and_facts_labeled(self):
        ws.focus_on("tom", self.rid)
        c = self.claude([text_reply("Enter at 35.")])
        cols = {}
        c.memory, c.facts = memory_over(cols), facts_over(cols)
        c._extractor = lambda: (lambda s, p: '{"ops": [{"op": "add", "entity": "plan:x", '
                                             '"value": "Enter at 35."}]}')
        c.converse("what's the rule")
        c.settle_reply("Enter at 35.", "Enter at 35.")
        ((_, _, metas),) = cols[cfg.agent_memory_collection("tom")].upserts
        self.assertTrue(all(m["project"] == self.rid for m in metas))
        ((_, _, fact_metas),) = cols[cfg.agent_facts_collection("tom")].upserts
        self.assertEqual(fact_metas[0]["project"], self.rid)

    def test_the_project_reaches_the_background(self):
        ws.focus_on("tom", self.rid)
        c = self.claude()
        with mock.patch.object(context, "build_context", wraps=context.build_context) as pull:
            c.converse("hello")
        self.assertEqual(pull.call_args.kwargs["project"]["id"], self.rid)

    def test_a_restart_says_focus_is_still_on_once(self):
        ws.focus_on("tom", self.rid)
        c = self.claude([text_reply("First."), text_reply("Second.")])  # a new process
        self.assertEqual(c.converse("hi"), "We're still focused on Rooftop Butterfly. First.")
        self.assertEqual(c.converse("and"), "Second.")

    def test_entering_during_a_session_is_not_announced_as_still_on(self):
        c = self.claude([tool_reply("focus_project", {"id": self.rid}), text_reply("Focused."),
                         text_reply("Next.")])
        self.assertEqual(c.converse("work on the rooftop"), "Focused.")
        self.assertEqual(c.converse("and"), "Next.")

    def test_another_personas_focus_is_not_announced(self):
        ws.focus_on("tom", self.rid)
        c = make_claude([text_reply("Hi.")], active="alice")
        c._focused_project = Claude._focused_project.__get__(c)
        self.assertEqual(c.converse("hi"), "Hi.")

    def test_the_announcement_doesnt_enter_history(self):
        ws.focus_on("tom", self.rid)
        c = self.claude([text_reply("First.")])
        c.converse("hi")
        self.assertEqual(c.history[-1]["content"][0]["text"], "First.")

    def test_a_broken_focus_store_costs_the_turn_nothing(self):
        c = self.claude([text_reply("Hi.")])
        with mock.patch.object(ws, "focused", side_effect=RuntimeError("locked")):
            self.assertEqual(c.converse("hello"), "Hi.")

    def test_a_delegation_reads_focus_but_cant_change_it(self):
        ws.focus_on("tom", self.rid)
        c = self.claude([text_reply("Done.")], convo_model=cfg.CONVO_MODELS["haiku"])
        with mock.patch.object(context, "build_context", wraps=context.build_context) as pull:
            c.run_delegated_task("tom", "look something up")
        self.assertEqual(pull.call_args.kwargs["project"]["id"], self.rid)
        names = {t["name"] for t in c.client.messages.calls[0]["tools"]}
        self.assertIn("update_record", names)
        self.assertNotIn("focus_project", names)


if __name__ == "__main__":
    unittest.main()
