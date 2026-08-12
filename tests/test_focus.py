"""Focus mode: retrieval narrowing that is hard on private collections, soft
on common, visible in the system prompt, and inherited by delegation."""

import unittest
from unittest import mock

from brain import agents
import config as cfg
from stores.knowledge import KnowledgeStore, _focus_where
from tests.llm_fixtures import make_claude, text_reply, tool_reply
from tests.test_knowledge import FakeCol, fake_store
from tools import ToolContext, dispatch
from tools.focus_tools import describe_focus, focus_prompt_block


class TestFocusWhere(unittest.TestCase):
    def test_single_value_is_an_equality(self):
        self.assertEqual(_focus_where({"strategy": "butterfly"}),
                         {"strategy": "butterfly"})

    def test_list_value_becomes_in(self):
        self.assertEqual(_focus_where({"strategy": ["butterfly", "diagonal"]}),
                         {"strategy": {"$in": ["butterfly", "diagonal"]}})

    def test_two_keys_are_anded(self):
        where = _focus_where({"strategy": "butterfly", "underlying": "SPX"})
        self.assertEqual(where, {"$and": [{"strategy": "butterfly"},
                                          {"underlying": "SPX"}]})

    def test_empty_focus_is_none(self):
        self.assertIsNone(_focus_where(None))
        self.assertIsNone(_focus_where({}))
        self.assertIsNone(_focus_where({"strategy": None}))


class RecordingCol(FakeCol):
    """A FakeCol that records the where= of every query."""

    def __init__(self, hits=()):
        super().__init__(hits)
        self.wheres = []

    def query(self, query_texts, n_results, where=None):
        self.wheres.append(where)
        if where is not None and not self.hits_match(where):
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        return super().query(query_texts, n_results)

    def hits_match(self, where):
        # Enough filter semantics for these tests: equality on flat keys.
        conds = where.get("$and", [where]) if isinstance(where, dict) else []
        for cond in conds:
            for key, expected in cond.items():
                for _, meta, _ in self.hits:
                    if meta.get(key) == expected:
                        break
                else:
                    return False
        return True


class TestFocusedSearch(unittest.TestCase):
    def setUp(self):
        self.cols = {}
        self.store = fake_store(self.cols)
        self.manifest = mock.patch.object(
            KnowledgeStore, "_load_manifest",
            return_value={"h1": {"source": "a"},
                          "h2": {"source": "b", "collection": "tom"}})
        self.manifest.start()
        self.addCleanup(self.manifest.stop)

    def test_focus_is_a_hard_filter_on_the_private_collection(self):
        self.cols[cfg.agent_knowledge_collection("tom")] = RecordingCol(
            [("fly note", {"title": "Fly", "strategy": "butterfly"}, 0.1),
             ("diag note", {"title": "Diag", "strategy": "diagonal"}, 0.2)])
        self.cols[cfg.KNOWLEDGE_COLLECTION] = RecordingCol()
        out = self.store.search("how did it go", caller="tom",
                                focus={"strategy": "diagonal"})
        self.assertIn("Diag", out)
        private = self.cols[cfg.agent_knowledge_collection("tom")]
        self.assertEqual(private.wheres, [{"strategy": "diagonal"}])

    def test_focus_is_soft_on_common_falls_back_when_empty(self):
        # Reference chunks carry no strategy tags: the filtered query finds
        # nothing, and the fallback must not hide the textbook.
        self.cols[cfg.KNOWLEDGE_COLLECTION] = RecordingCol(
            [("textbook passage", {"title": "Book"}, 0.4)])
        out = self.store.search("what is a diagonal", caller="tom",
                                focus={"strategy": "diagonal"})
        self.assertIn("Book", out)
        common = self.cols[cfg.KNOWLEDGE_COLLECTION]
        self.assertEqual(common.wheres, [{"strategy": "diagonal"}, None])

    def test_no_focus_queries_unfiltered(self):
        self.cols[cfg.KNOWLEDGE_COLLECTION] = RecordingCol(
            [("passage", {"title": "Book"}, 0.4)])
        self.store.search("anything", caller="tom")
        self.assertEqual(self.cols[cfg.KNOWLEDGE_COLLECTION].wheres, [None])


class TestFocusTools(unittest.TestCase):
    def setUp(self):
        self.ctx = ToolContext(active_agent="tom")

    def test_set_get_clear_roundtrip(self):
        out = dispatch(self.ctx, "set_focus", {"strategy": "double_diagonal",
                                               "underlying": "SPX"})
        self.assertIn("double_diagonal", out)
        self.assertEqual(self.ctx.focus, {"strategy": "double_diagonal",
                                          "underlying": "SPX"})
        self.assertIn("double_diagonal", dispatch(self.ctx, "get_focus", {}))
        out = dispatch(self.ctx, "clear_focus", {})
        self.assertIn("Cleared", out)
        self.assertIsNone(self.ctx.focus)

    def test_empty_set_focus_sets_nothing(self):
        out = dispatch(self.ctx, "set_focus", {})
        self.assertIsNone(self.ctx.focus)
        self.assertIn("Nothing to focus on", out)

    def test_clear_without_focus_is_honest(self):
        self.assertIn("No focus", dispatch(self.ctx, "clear_focus", {}))

    def test_focus_tools_are_toms_only(self):
        for name in ("set_focus", "clear_focus", "get_focus"):
            self.assertIn(name, agents.AGENTS["tom"]["tools"])
            self.assertNotIn(name, agents.AGENTS["alice"]["tools"])
            self.assertNotIn(name, agents.AGENTS["bob"]["tools"])

    def test_describe_focus_reads_naturally(self):
        self.assertEqual(
            describe_focus({"strategy": ["butterfly", "diagonal"],
                            "underlying": "SPX"}),
            "strategy butterfly, diagonal; underlying SPX")


class TestFocusInThePrompt(unittest.TestCase):
    def test_prompt_line_appears_exactly_when_focus_is_set(self):
        c = make_claude(active="tom")
        c.converse("hello")
        self.assertNotIn("Focus:", c.client.messages.calls[0]["system"])
        c._ctx.focus = {"strategy": "double_diagonal"}
        c.converse("how are my diagonals?")
        system = c.client.messages.calls[1]["system"]
        self.assertIn("Focus: retrieval is narrowed to strategy "
                      "double_diagonal", system)
        self.assertIn("clear_focus", system)

    def test_focus_survives_a_persona_switch(self):
        # Deliberate (TODO §1): focus is session state on the shared
        # ToolContext, so a detour through Alice keeps it.
        c = make_claude(active="tom")
        c._ctx.focus = {"strategy": "butterfly"}
        c.switch_to("alice")
        c.switch_to("tom")
        self.assertEqual(c._ctx.focus, {"strategy": "butterfly"})

    def test_delegation_inherits_the_focus(self):
        c = make_claude([text_reply("done")], active="tom")
        c._ctx.focus = {"strategy": "butterfly"}
        c.run_delegated_task("bob", "look something up")
        self.assertIn("Focus: retrieval is narrowed to strategy butterfly",
                      c.client.messages.calls[0]["system"])

    def test_empty_focus_adds_no_prompt_block(self):
        self.assertEqual(focus_prompt_block(None), "")
        self.assertEqual(focus_prompt_block({}), "")


if __name__ == "__main__":
    unittest.main()
