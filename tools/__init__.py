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

from dataclasses import dataclass, field


@dataclass
class ToolContext:
    """Everything a tool handler may need. Owned by the Claude wrapper; one
    instance lives for the whole session."""
    store: object = None      # NoteStore
    discord: object = None    # DiscordData
    kb: object = None         # KnowledgeStore
    memory: object = None     # ConversationMemory
    # Set by save_conversation_note; the agent picks it up after the reply and
    # runs the folder dialogue + save (see voice_agent._save_pending_note).
    pending_note: dict = field(default=None)
    # Active conversation model id; set by set_conversation_model and read by
    # Claude.converse each call, so the user can switch models by voice.
    convo_model: str = None
    # Key of the persona currently answering (see agents.py); read by
    # switch_agent so "switch to Bob" while already Bob can say so.
    active_agent: str = None
    # Set by switch_agent: (agent_key, forward_text). The agent picks it up
    # after the reply and performs the actual switch + optional forwarded
    # question (see voice_agent) — the same deferred pattern as pending_note.
    pending_switch: tuple = field(default=None)
    # Factual notes about work a tool did *beyond* the string it returned — a
    # deferred save, or a sub-dialogue that ran in its own model memory. Only
    # a tool's return value lands in history automatically; anything that
    # happens after (or in separate memory) is invisible to the model unless
    # recorded here. Claude.flush_tool_events folds these into the conversation
    # so the next turn knows what its tools actually did.
    events: list = field(default_factory=list)

    def record_event(self, text):
        """Record what a tool actually did, for the model to see next turn.
        No-op on empty text so callers needn't guard."""
        if text:
            self.events.append(str(text))


_REGISTRY = {}  # name -> (schema, handler); insertion-ordered


def tool(schema):
    """Register a tool. `schema` is the full Anthropic tool dict (name,
    description, input_schema); the decorated function is its handler."""
    def deco(fn):
        _REGISTRY[schema["name"]] = (schema, fn)
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
    try:
        return handler(ctx, args or {})
    except Exception as e:  # surface tool errors back to the model
        return f"Tool error: {e}"


# Importing the tool modules populates the registry. Order sets the order the
# schemas are presented to the model.
from tools import note_tools      # noqa: E402,F401
from tools import discord_tools   # noqa: E402,F401
from tools import time_tools      # noqa: E402,F401
from tools import memory_tools    # noqa: E402,F401
from tools import knowledge_tools # noqa: E402,F401
from tools import model_tools     # noqa: E402,F401
from tools import project_tools   # noqa: E402,F401
from tools import agent_tools     # noqa: E402,F401
