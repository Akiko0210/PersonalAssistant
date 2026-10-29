"""Tool registry: one place where a tool's schema and handler live together.

Adding a capability used to mean editing three places (the TOOLS list, the
_dispatch if-chain, and remembering the wiring); now it's one decorated
function in a module under tools/:

    @tool({"name": "...", "description": "...", "input_schema": {...}})
    def my_tool(ctx, args):
        return "spoken-friendly result string"

Handlers receive the shared ToolContext (the app's stores) and the raw args
dict from the model, and return a string for the tool_result. Schemas are
kept as explicit dicts — they are prompt engineering, not boilerplate, so
they stay reviewable next to their handler.
"""

import logging
from dataclasses import dataclass, field

log = logging.getLogger("tools")


@dataclass
class ToolContext:
    """Everything a tool handler may need. Owned by the Claude wrapper; one
    instance lives for the whole session."""
    store: object = None      # NoteStore
    discord: object = None    # DiscordData
    kb: object = None         # KnowledgeStore
    memory: object = None     # ConversationMemory
    facts: object = None      # FactStore
    # Set by save_conversation_note; the agent picks it up after the reply and
    # runs the folder dialogue + save (see voice_agent._save_pending_note).
    pending_note: dict = field(default=None)
    # Active conversation model id; set by set_conversation_model and read by
    # Claude.converse each call, so the user can switch models by voice.
    convo_model: str = None
    # The conversation's provider failover (brain/llm/failover.py), shared
    # with every delegation: which model actually answers while a provider is
    # down. None runs without one.
    failover: object = None
    # Key of the persona currently answering (see agents.py); read by
    # switch_agent so "switch to Bob" while already Bob can say so.
    active_agent: str = None
    # Set by switch_agent: (agent_key, forward_text). The agent picks it up
    # after the reply and performs the actual switch + optional forwarded
    # question (see voice_agent) — the same deferred pattern as pending_note.
    pending_switch: tuple = field(default=None)
    # Appended to by ask_agent: [(agent_key, task_text), ...]. Unlike a switch,
    # a delegation never moves the user: the agent starts each task on a
    # background thread as that persona (Claude.run_delegated_task) and the
    # result comes back as a spoken interjection between utterances.
    pending_delegations: list = field(default_factory=list)
    # The user's message this turn, set by Claude.converse. ask_agent goes
    # through only when it names the other assistant: delegating on its own
    # initiative, Alice sent Bob two "confirm it" / "check your log" jobs the
    # user never asked for (session_2026-10-06.log, 14:58 and 15:00). A
    # delegated task's fresh ToolContext leaves it empty.
    user_text: str = ""
    # Per-session retrieval narrowing (set_focus): folded into store where=
    # filters, e.g. {"strategy": "double_diagonal", "underlying": "SPX"}.
    # Session state, deliberately never persisted — a stale focus outliving
    # the session is exactly the failure mode that makes retrieval look
    # broken. Values may be lists (the store turns them into $in).
    focus: dict = field(default=None)
    # Exchange ids in this turn's verbatim tail, set by Claude._pull_context.
    # recall excludes them, so a persona never reads its own current answers
    # back as corroborating "past conversations" — at
    # 12:08 on 2026-09-17 the tool returned Alice's replies from three minutes
    # earlier and she took them as evidence. A delegated task's fresh
    # ToolContext leaves it empty, which is right: it has no tail.
    tail_ids: frozenset = frozenset()
    # Factual notes about work a tool did *beyond* the string it returned — a
    # deferred save, or a sub-dialogue that ran in its own model memory. Only
    # a tool's return value lands in history automatically; anything that
    # happens after (or in separate memory) is invisible to the model unless
    # recorded here. Claude.flush_tool_events folds these into the conversation
    # so the next turn knows what its tools actually did.
    events: list = field(default_factory=list)
    # (heading, card-or-text) -> None: the live Agent's show_card, which puts
    # a tool's answer on the display window (display/). Only the conversation
    # has one; a delegated task's fresh ToolContext leaves it None, so a
    # background answer never swaps out what the user is looking at.
    show: object = None
    # The card a display= tool built for this call (display/card.py), from
    # the same figures as its returned sentence. Reset by dispatch before
    # every call; a tool that sets none (a refusal, an error) is shown as text.
    card: dict = field(default=None)

    def record_event(self, text):
        """Record what a tool actually did, for the model to see next turn.
        No-op on empty text so callers needn't guard."""
        if text:
            self.events.append(str(text))


_REGISTRY = {}  # name -> (schema, handler); insertion-ordered
_DISPLAY = {}   # name -> (ctx, args) -> heading, for tools shown on the display
_CARD_EXPORT = set()  # display tools whose export= writes their card

# Every display tool can save its table: asked for the position table as a
# CSV, Tom had to refuse while export lived on the trade log alone (#53).
_EXPORT_ARG = {"type": "boolean", "description": (
    "Also save this answer's table, as shown on the display, to a CSV in "
    "data/exports/. Only when the user asks to export or save it.")}


def tool(schema, display=None):
    """Register a tool. `schema` is the full Anthropic tool dict (name,
    description, input_schema); the decorated function is its handler.
    `display`, a (ctx, args) -> heading function, puts every answer the tool
    gives on the display window under that heading: the card it left in
    ctx.card, built from the same figures as its returned string, or that
    string itself when it built none. A display tool also takes export=,
    which saves that card as a CSV — unless its schema declares its own
    export (query_trade_log writes every row, not just the shown ones)."""
    def deco(fn):
        _REGISTRY[schema["name"]] = (schema, fn)
        if display:
            _DISPLAY[schema["name"]] = display
            props = schema["input_schema"].setdefault("properties", {})
            if "export" not in props:
                props["export"] = _EXPORT_ARG
                _CARD_EXPORT.add(schema["name"])
        return fn
    return deco


def api_tools(exclude=(), include=None):
    """The `tools=` list for a messages.create call. `exclude` drops tools
    that don't fit the current dialogue; `include` (an allowlist of names)
    keeps only those — used for per-agent tool subsets. Registry insertion
    order is preserved either way."""
    return [schema for name, (schema, _) in _REGISTRY.items()
            if name not in exclude and (include is None or name in include)]


def dispatch(ctx, name, args):
    """Run one tool call. Errors come back as strings so the model can react
    (and the conversation loop never dies on a tool bug)."""
    entry = _REGISTRY.get(name)
    if entry is None:
        return f"Unknown tool: {name}"
    _, handler = entry
    args = args or {}
    ctx.card = None
    try:
        out = handler(ctx, args)
    except Exception as e:  # surface tool errors back to the model
        return f"Tool error: {e}"
    if args.get("export") and name in _CARD_EXPORT:
        out = _export_card(name, ctx.card, out)
    heading = _DISPLAY.get(name)
    if heading and ctx.show:
        try:
            ctx.show(heading(ctx, args), ctx.card or out)
        except Exception:  # the window is an aid; it must never cost the answer
            log.exception("display card for %s failed", name)
    return out


def _export_card(name, card, out):
    """`out`, led by what the export did. A failed write is said, never
    raised: the answer itself must still reach the user."""
    if not card:
        return out + "\n(Nothing was exported: this answer has no table.)"
    from display import card as blocks   # here: display imports trading
    from lib import csv_export
    rows = blocks.table(card)
    try:
        path = csv_export.write(name.removeprefix("get_"), blocks.TABLE_HEADER,
                                rows)
    except OSError as e:
        log.exception("export of %s failed", name)
        return out + f"\n(The export failed: {e}. Nothing was saved.)"
    return (f"Exported {len(rows)} rows to {path.name} in the data/exports "
            "folder.\n" + out)


# Importing the tool modules populates the registry. Order sets the order the
# schemas are presented to the model.
from tools import note_tools      # noqa: E402,F401
from tools import discord_tools   # noqa: E402,F401
from tools import time_tools      # noqa: E402,F401
from tools import memory_tools    # noqa: E402,F401
from tools import focus_tools     # noqa: E402,F401
from tools import working_state_tools  # noqa: E402,F401
from tools import model_tools     # noqa: E402,F401
from tools import project_tools   # noqa: E402,F401
from tools import agent_tools     # noqa: E402,F401
from tools import trading_tools   # noqa: E402,F401
from tools import trade_log_tools  # noqa: E402,F401
from tools import gmail_tools     # noqa: E402,F401
