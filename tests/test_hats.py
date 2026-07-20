"""Persona ("hat") smoke tests against a fake client: switching agents must
change the model, the tool subset, and the system prompt — while the one
shared history keeps flowing through untouched."""

import unittest
from types import SimpleNamespace

import agents
import config as cfg
from llm import Claude
from tools import ToolContext


class FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self, exclude_none=False):
        return dict(self.__dict__)


def _reply(text="ok"):
    return SimpleNamespace(stop_reason="end_turn",
                           content=[FakeBlock(type="text", text=text)])


class CapturingMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _reply()


def make_claude():
    c = Claude.__new__(Claude)
    c.client = SimpleNamespace(messages=CapturingMessages())
    c.active = agents.DEFAULT_AGENT
    c._model_overrides = {}
    c._ctx = ToolContext(active_agent=c.active,
                         convo_model=cfg.CONVO_MODELS["haiku"])
    c.history = []
    c.idle = SimpleNamespace(start=lambda: None, stop=lambda: None)
    c.memory = SimpleNamespace(record_dropped=lambda dropped: None)
    c._save_history = lambda: None
    c._write_agent_state = lambda: None  # no disk writes from tests
    return c


class TestHats(unittest.TestCase):
    def test_alice_defaults(self):
        c = make_claude()
        c.converse("hello")
        call = c.client.messages.calls[0]
        self.assertEqual(call["model"], cfg.CONVO_MODELS["haiku"])
        names = {t["name"] for t in call["tools"]}
        self.assertEqual(names, set(agents.AGENTS["alice"]["tools"]))
        self.assertIn("You are Alice", call["system"])

    def test_switch_to_cobe_changes_model_tools_and_prompt(self):
        c = make_claude()
        c.switch_to("cobe")
        c.converse("how did SPX trades go?")
        call = c.client.messages.calls[0]
        self.assertEqual(call["model"], cfg.CONVO_MODELS["sonnet"])
        names = {t["name"] for t in call["tools"]}
        self.assertIn("search_knowledge", names)
        self.assertNotIn("search_notes", names)
        self.assertIn("You are Cobe", call["system"])

    def test_shared_history_survives_switching(self):
        c = make_claude()
        c.converse("tell alice something")
        c.switch_to("bob")
        c.converse("bob, do you remember?")
        # One list, both turns present in the second call's messages.
        texts = str(c.client.messages.calls[1]["messages"])
        self.assertIn("tell alice something", texts)

    def test_model_override_sticks_to_its_hat(self):
        c = make_claude()
        c.switch_to("cobe")
        c._ctx.convo_model = cfg.CONVO_MODELS["opus"]  # "make Cobe smarter"
        c.switch_to("alice")
        self.assertEqual(c._ctx.convo_model, cfg.CONVO_MODELS["haiku"])
        c.switch_to("cobe")
        self.assertEqual(c._ctx.convo_model, cfg.CONVO_MODELS["opus"])

    def test_pending_switch_roundtrip(self):
        c = make_claude()
        c._ctx.pending_switch = ("bob", "what's my last note?")
        self.assertEqual(c.take_pending_switch(), ("bob", "what's my last note?"))
        self.assertIsNone(c.take_pending_switch())


if __name__ == "__main__":
    unittest.main()
