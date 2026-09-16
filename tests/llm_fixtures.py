"""Shared fakes for the Claude/LLM-side tests (see trading_fixtures.py for the
trading equivalent). One definition each of the fakes that had grown a copy per
test file — a new attribute on Claude means editing ONE builder, not five.

Only genuinely shared shapes live here; a fake used by a single file stays
local to that file. Store doubles (FakeCol) live in store_fixtures.py.
"""

from types import SimpleNamespace

from brain import agents
import config as cfg
from brain.llm.main import Claude
from brain.memory import Rows
from tools import ToolContext


class FakeBlock:
    """An SDK content block: attribute bag + the model_dump the code calls."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self, exclude_none=False):
        return dict(self.__dict__)


def text_reply(text="ok"):
    return SimpleNamespace(stop_reason="end_turn",
                           content=[FakeBlock(type="text", text=text)])


def tool_reply(name, args, block_id="tu_1"):
    return SimpleNamespace(stop_reason="tool_use",
                           content=[FakeBlock(type="tool_use", name=name,
                                              input=args, id=block_id)])


class ScriptedMessages:
    """messages.create capturing every call; returns canned responses in
    order, or an endless text_reply("ok") when built with responses=None
    (the capture-only shape some tests need)."""

    def __init__(self, responses=None):
        self.calls = []
        self._responses = None if responses is None else list(responses)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses is None:
            return text_reply()
        return self._responses.pop(0)


_UNSET = object()


def make_claude(responses=None, *, active=None, convo_model=_UNSET,
                history=None):
    """A Claude shell via __new__ — no API client, stores, or embedding model.
    Covers the union of what converse / run_delegated_task / switch_to /
    flush_tool_events touch; the extras are inert for narrower tests."""
    c = Claude.__new__(Claude)
    c.client = SimpleNamespace(messages=ScriptedMessages(responses))
    c._deepseek = None
    c.active = active or agents.DEFAULT_AGENT
    c._model_overrides = {}
    c.store = None
    c.discord = None
    # Retrieval seams answer empty, so converse() builds no Background and
    # the exchange index write is a no-op — tests that care install their own.
    c.kb = SimpleNamespace(query_rows=lambda *a, **k: [])
    c.memory = SimpleNamespace(index_exchanges=lambda *a, **k: 0,
                               query_rows=lambda *a, **k: Rows([], [], 0, None))
    c._ctx = ToolContext(
        active_agent=c.active,
        convo_model=(cfg.CONVO_MODELS["haiku"] if convo_model is _UNSET
                     else convo_model))
    c.history = list(history or [])
    c.idle = SimpleNamespace(start=lambda: None, stop=lambda: None)
    c.saved = []    # snapshots of history at each persist
    c.threads = {}  # per-agent thread "files": what switch_to saves/loads

    def _save():
        c.saved.append(list(c.history))
        c.threads[c.active] = list(c.history)

    def _load():
        return list(c.threads.get(c.active, []))

    c._save_history = _save
    c._load_history = _load
    c._write_agent_state = lambda: None  # no disk writes from tests
    return c


def system_text(call):
    """The system prompt of a captured messages.create call as one string,
    whether it was sent as a str or as the block list system_blocks builds."""
    system = call["system"]
    if isinstance(system, str):
        return system
    return "\n".join(b.get("text", "") for b in system)
