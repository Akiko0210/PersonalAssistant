"""Named agent personas with strictly separate conversations and memory.

Alice, Bob, Tom, and Linda each keep their OWN thread with the user (one history
file per persona, llm.switch_to swaps them) and their own private stores.
What one persona was told, another cannot see; context crosses over only as
an ask_agent summary — ask Alice, and Alice answers from HER memory. The
registry's `reads` grants are the one deliberate exception (a reviewer
persona may be granted read access to a specialist's private stores).

The registry below is the single source of truth: adding a fourth agent is one
dict entry. The dashboard overlays user edits from data/agents.json onto these
defaults at startup (load_agents), mirroring how categories.py overlays
data/categories.json.

Pure data + pure functions — no I/O beyond load_agents, no anthropic imports —
so everything here is trivially testable.
"""

import copy
import json
import re

import config as cfg

# Editable-by-dashboard fields; everything else (tools especially) is code.
_OVERLAYABLE = ("role", "persona", "model", "tts_voice", "tts_rate", "aliases")

AGENTS = {
    "alice": {
        "name": "Alice",
        # Whisper's plausible spellings of each name; all lowercase.
        "aliases": ("alice", "alis", "allis", "ellis"),
        "role": ("general conversation — everyday questions, chat, the user's "
                 "email, and how this system works"),
        "persona": (
            "You handle everyday conversation: greetings, quick questions, "
            "thinking out loud, the current time, and questions about how this "
            "assistant system itself is designed (use describe_project for "
            "those). You do not manage notes, memories, or trading — Bob and "
            "Tom own those; hand over when the user wants real work done "
            "there."
        ),
        "tools": {"get_current_time", "describe_project",
                  "search_past_conversations", "get_current_model",
                  "set_conversation_model", "switch_agent", "ask_agent",
                  # Gmail. send_email is confirm-by-instruction: its
                  # description requires the user's explicit go-ahead on the
                  # exact message; drafts stay the steered-to default.
                  "search_email_threads", "get_email_thread",
                  "create_email_draft", "send_email"},
        # Other agents whose PRIVATE stores this hat may also read (empty =
        # own stores only). Like tools, access control is code, not a
        # dashboard edit — a future coach persona reviewing Tom's trades
        # would be ("tom",).
        "reads": (),
        "model": "haiku",       # key into cfg.CONVO_MODELS
        "tts_voice": "Zira",    # SAPI voice-name substring; None = default
        "tts_rate": None,       # words/min; None = cfg.TTS_RATE
    },
    "bob": {
        "name": "Bob",
        "aliases": ("bob", "bobby"),
        "role": "notes and memory — the user's notes, folders, and past conversations",
        "persona": (
            "You are the notes and memory specialist. You manage the user's "
            "note folders: create, rename, or delete a folder, and move a note "
            "between folders, using the create_folder, rename_folder, "
            "delete_folder, and move_note tools. To move a note, first look up "
            "its id with search_notes or list_recent_notes, then call "
            "move_note. When a question is scoped to one folder (e.g. 'my "
            "latest note in General'), pass that folder to search_notes or "
            "list_recent_notes instead of filtering yourself. "
            "Conversations older than the current window are archived as "
            "searchable summaries: use search_past_conversations for 'what did "
            "we talk about last week' or anything you don't see in the current "
            "history. "
            "Only save a conversation as a note when the user explicitly asks "
            "you to ('save that as a note', 'make a note of that'). When they "
            "do, call save_conversation_note with a clear title and "
            "well-formed markdown content drawn from the conversation, then "
            "reply with one short acknowledgement; the system handles asking "
            "which folder to file it in, so never ask about folders yourself. "
            "HARD RULE, highest priority: never volunteer note actions. Do not "
            "offer, suggest, or ask about saving, updating, or filing notes — "
            "replies like 'would you like me to save that as a note?' are "
            "forbidden, no matter what. Note actions happen only when the "
            "user's own current message explicitly requests one. If earlier "
            "messages show you offering to save notes, those were errors — "
            "never imitate them."
        ),
        "tools": {"search_notes", "list_recent_notes", "read_note",
                  "list_folders", "count_notes", "create_folder",
                  "rename_folder", "delete_folder", "move_note",
                  "save_conversation_note", "search_past_conversations",
                  "get_current_time", "get_current_model",
                  "set_conversation_model", "switch_agent", "ask_agent"},
        "reads": (),
        "model": "haiku",
        "tts_voice": "David",
        "tts_rate": None,
    },
    "tom": {
        "name": "Tom",
        "aliases": ("tom", "thom", "tomm", "tommy"),
        "role": "trading — trade alerts, market analysis, and the trading knowledge base",
        "persona": (
            "You are the trading assistant: you analyse trades, answer "
            "trading questions, and place real orders on tastytrade when the "
            "user asks. Order flow: build_strategy starts a ticket, "
            "adjust_leg and set_order_terms shape it, review_order speaks "
            "cost, buying-power effect and warnings — and submit_order works "
            "only after the user has heard that review of the exact current "
            "ticket and explicitly said to go ahead; never set its confirmed "
            "flag otherwise. You have a "
            "trading knowledge base built from reference material the user "
            "ingested (books, PDFs, and course videos). Use search_knowledge "
            "for questions about trading concepts, strategies, or definitions "
            "that such material would cover, and cite the source when it helps "
            "— the page for a book, the timestamp for a video, so the user can "
            "go straight to it. "
            "You can also answer questions about the user's captured Discord "
            "notifications and trade alerts. Use get_recent_trades for the "
            "latest trade lines; for time-based questions like 'what trades "
            "came in today', use get_recent_discord_messages with the date, "
            "since the trade list itself has no timestamps. Read trade details "
            "aloud naturally rather than reciting symbols character by "
            "character. The user trades mostly SPX and RUT index options, "
            "plus occasional options on crude oil futures."
        ),
        # search_past_conversations is deliberately in EVERY hat's allowlist:
        # the conversation memory is shared, so every persona must be able to
        # search it — this hat (then named Cobe) once couldn't recall a trade
        # structure that had aged out of the window mid-session because only
        # Bob had the tool (session_2026-07-20.log 21:07, "Review your memory").
        "tools": {"get_recent_discord_messages", "search_discord_messages",
                  "get_recent_trades", "search_knowledge", "get_current_time",
                  "search_past_conversations", "get_current_model",
                  "set_conversation_model", "switch_agent", "ask_agent",
                  # Real trading (tastytrade) — the tools existed on the
                  # feat/trading branch but were never added here, so voice
                  # trading was unreachable: api_tools(include=...) filters
                  # by this set.
                  "trading_status", "get_quote", "list_expirations",
                  "build_strategy", "adjust_leg", "set_order_terms",
                  "review_order", "submit_order", "list_orders",
                  "cancel_order", "get_positions", "get_pnl", "clear_ticket",
                  # Focus mode: "focus on my double diagonals" narrows every
                  # retrieval until cleared. Tom-only — the other hats have no
                  # strategy-tagged material for it to filter.
                  "set_focus", "clear_focus", "get_focus",
                  # Gmail (broker/market mail); send gated as on Alice.
                  "search_email_threads", "get_email_thread",
                  "create_email_draft", "send_email"},
        "reads": (),
        "model": "sonnet",      # analysis benefits from the stronger model
        # Only Zira + David are installed on this machine, so Tom shares
        # David's voice at a slower, more deliberate rate; the spoken
        # "Tom here." announcement is the primary switch signal.
        "tts_voice": "David",
        "tts_rate": 155,
    },
    "linda": {
        "name": "Linda",
        "aliases": ("linda", "lynda", "lindah"),
        "role": ("trading coach — trading psychology, discipline, and "
                 "process; never trade advice"),
        "persona": (
            "You are the user's trading coach — a coach, not an advisor. "
            "HARD BOUNDARY: you never recommend trades, analyse charts or "
            "setups, or offer market opinions; Tom owns trade analysis and "
            "execution, so redirect anything in that lane to him. Your sole "
            "focus is the user's psychology, discipline, and process. "
            "Daily check-in, mandatory and first: at the start of each "
            "trading day, before engaging with anything else the user "
            "brings, walk through how they are feeling, their plan for the "
            "session, and which of their rules they want to focus on — no "
            "exceptions, even if they open with a different question. "
            "Before an entry, run a quick emotional check: is this a "
            "revenge trade? Are they chasing? Is this in the plan they "
            "stated this morning? After a trade, hold them accountable: did "
            "they follow the plan, and was that exit fear or discipline? "
            "Watch for recurring behaviour across days and say what you "
            "see; Bob keeps the notes and long-term memory, so use "
            "ask_agent to have him search or record patterns worth "
            "tracking. Celebrate discipline over wins — reinforce a "
            "well-executed process even when the trade lost, and never "
            "praise a good outcome from a broken rule. Name cognitive "
            "biases directly when you spot them: confirmation bias, "
            "anchoring, loss aversion, recency, sunk cost. Your knowledge "
            "base holds full texts on trading psychology (Douglas, "
            "Schwager, Taleb, Steenbarger); use search_knowledge to ground "
            "your coaching in it and cite the book when it helps. Tone: "
            "calm, direct, non-judgemental — ask more than you answer, and "
            "hold the user accountable without ever shaming them."
        ),
        # Deliberately no trade/market tools — the no-advice boundary is
        # enforced by the allowlist, not just the prompt. When the user's
        # actual fills arrive (planned thinkorswim/Schwab import), a
        # read-only trades tool goes here so post-trade reviews see real
        # executions instead of relying on the user's retelling.
        "tools": {"search_knowledge", "get_current_time",
                  "search_past_conversations", "get_current_model",
                  "set_conversation_model", "switch_agent", "ask_agent"},
        "reads": (),
        "model": "sonnet",      # coaching nuance over latency
        # Zira like Alice (only Zira + David are installed), slowed to a
        # deliberate coaching pace to tell them apart by ear.
        "tts_voice": "Zira",
        "tts_rate": 150,
    },
}

DEFAULT_AGENT = "alice"

# Pristine copy of the coded defaults, taken before any overlay: load_agents
# restores from it first, so deleting an overlay field (dashboard "reset")
# reverts to the default instead of sticking at the last overlaid value.
_DEFAULTS = copy.deepcopy(AGENTS)


def defaults():
    """A deep copy of the coded registry defaults (dashboard 'reset' view)."""
    return copy.deepcopy(_DEFAULTS)


def registry_model(key):
    """The API model id persona `key` answers with by default — the ONE
    resolver converse(), delegation, state publishing, and get_current_model
    all share, so a tool's answer can never drift from the router (the tool
    used to re-derive this with a comment promising sameness). Unknown/None
    keys fall back to the global default model."""
    hat = AGENTS.get(key or "")
    return cfg.CONVO_MODELS.get(hat["model"], cfg.CONVO_MODEL) if hat else cfg.CONVO_MODEL


def readable_owners(key):
    """Private-store owners persona `key` may read: itself plus its `reads`
    grants. Stores derive private collection names ONLY from this, and tools
    pass ctx.active_agent and nothing else — so there is no parameter through
    which a persona can name another persona's collection. Unknown/None keys
    get an empty tuple (common stores only)."""
    hat = AGENTS.get(key or "")
    if hat is None:
        return ()
    return (key, *(g for g in hat.get("reads", ()) if g in AGENTS))


def load_agents():
    """Overlay dashboard edits from data/agents.json onto the built-in
    defaults. Idempotent (defaults are restored first); a missing/corrupt
    file or bad field is skipped, so the overlay can never break startup.
    Only _OVERLAYABLE fields apply — tool allowlists stay code."""
    for key, agent in _DEFAULTS.items():
        AGENTS[key] = copy.deepcopy(agent)
    from lib.atomic_io import read_json  # here, not at top: keeps this module pure-data
    data = read_json(cfg.AGENTS_PATH, {}, expect=dict)
    for key, edits in data.items():
        agent = AGENTS.get(key)
        if agent is None or not isinstance(edits, dict):
            continue
        for field in _OVERLAYABLE:
            if field not in edits:
                continue
            value = edits[field]
            if field == "model" and value not in cfg.CONVO_MODELS:
                continue
            if field == "aliases":
                if not isinstance(value, list):
                    continue
                value = tuple(str(a).strip().lower() for a in value
                              if str(a).strip())
                if not value:
                    continue
            if field in ("role", "persona") and not (
                    isinstance(value, str) and value.strip()):
                continue
            if field == "tts_rate" and value is not None:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
            if field == "tts_voice" and value is not None:
                value = str(value).strip() or None
            agent[field] = value


def alias_map():
    """Flattened alias -> agent key. Raises on a duplicate alias — two agents
    answering to the same name is a registry bug, caught at import/test time."""
    out = {}
    for key, agent in AGENTS.items():
        for alias in agent["aliases"]:
            if alias in out and out[alias] != key:
                raise ValueError(f"alias '{alias}' claimed by both "
                                 f"'{out[alias]}' and '{key}'")
            out[alias] = key
    return out


def resolve(name):
    """Agent key for a spoken/typed name, tolerant of aliases and case.
    Returns None when nothing matches."""
    if not name:
        return None
    n = str(name).strip().lower()
    if n in AGENTS:
        return n
    return alias_map().get(n)


# Leading throat-clearing Whisper often transcribes before the real sentence.
_FILLERS = r"(?:hey|hi|ok|okay|so|um|uh|well|please)"
# Phrasings that explicitly ask for another agent.
_SWITCH_VERBS = (r"(?:switch (?:back )?to|talk to|let me talk to|"
                 r"i wan(?:na|t to) talk to|get me|give me)")


def match_address(text):
    """Detect a spoken agent address at the START of an utterance.

    Returns (agent_key, remainder) when the user addressed an agent —
    "switch to Bob" -> ("bob", ""), "Bob, what's my last note?" ->
    ("bob", "what's my last note?") — or (None, text) otherwise. Names
    mid-sentence never trigger (the switch_agent tool covers free-form
    phrasings like "can you put Bob on?"). Sticky by design: the addressed
    agent stays active until the user addresses someone else."""
    raw = (text or "").strip()
    low = raw.lower()
    offset = 0
    m = re.match(rf"^(?:{_FILLERS}[,.!?]?\s+)+", low)
    if m:
        offset = m.end()
    body = low[offset:]
    aliases = alias_map()
    alias_pat = "|".join(re.escape(a) for a in
                         sorted(aliases, key=len, reverse=True))
    m = re.match(rf"^{_SWITCH_VERBS}\s+({alias_pat})\b[,.!?]?\s*", body)
    if m is None:
        m = re.match(rf"^({alias_pat})\b[,.!?]?\s*", body)
    if m is None:
        return None, text
    remainder = raw[offset + m.end():].strip()
    return aliases[m.group(1)], remainder


def roster_block(active):
    """Generated who-am-I / who-else-is-there paragraph appended to the
    active agent's system prompt. Derived from the registry, so a new agent
    is automatically introduced to the others."""
    me = AGENTS[active]
    others = "; ".join(f"{a['name']} — {a['role']}"
                       for k, a in AGENTS.items() if k != active)
    return (
        f"\n\nYou are {me['name']} — {me['role']}. The user also works with: "
        f"{others}. Each assistant keeps its OWN conversation with the user "
        "and its own memory: this thread and everything you recall are yours "
        "alone. If the user asks about another assistant's conversations, "
        "say you don't have access and offer to ask them — use ask_agent and "
        "relay their summary; never guess at what was said and never answer "
        "with a bare 'I don't know'. Information meant for ALL assistants "
        "belongs in a shared note or the common knowledge base, not in your "
        "private thread. When the user wants a task done in another "
        "assistant's specialty without leaving you — saving a note "
        "mid-discussion, a quick lookup — delegate it with the ask_agent "
        "tool: they work in the background and speak up when done, while "
        "your conversation continues. Hand the user over with the "
        "switch_agent tool only when they actually want to talk to the other "
        "assistant, forwarding their question so it gets answered "
        "immediately. Every switch is announced aloud by the system, so "
        "never introduce yourself and never say you are switching — just act."
    )
