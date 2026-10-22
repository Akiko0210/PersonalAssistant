"""Named agent personas with strictly separate conversations and memory.

Alice, Bob, Tom, Bonny, and Linda each keep their OWN thread with the user (one
history file per persona, llm.switch_to swaps them) and their own private stores.
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
# record_rules is deliberately not here: it is tool protocol, like an
# allowlist — a dashboard persona saved on 2026-10-03 silently replaced
# Linda's coded one, check-in instruction and all.
_OVERLAYABLE = ("role", "persona", "model", "tts_voice", "tts_rate", "aliases")

# The working-record tools (tools/working_state_tools.py), granted to every
# persona that keeps a record and kept out of side dialogues.
RECORD_TOOLS = {"update_record", "add_entry", "close_record", "read_record"}

# How every record-keeping persona uses its record; each hat's record_rules
# then says WHEN it writes. Concrete triggers on purpose: Linda's "mandatory
# and first" check-in happened 5 days of 5, while "watch for recurring
# behaviour … have Bob record patterns" essentially never did.
RECORD_RULES = (
    "Your working record (shown after this prompt, before the Background) is "
    "the one place you keep what you track for the user across sessions, and "
    "it is shown to you every turn — never search for it, and never hand it "
    "to another assistant. Keep it current yourself with update_record, add_entry and "
    "close_record, at the moment one of your triggers below happens, not at "
    "the end. Mention a write in a few words inside your reply ('noted in "
    "the journal'), never as a separate announcement. Open each session from "
    "it: what is open, what was planned. A body holds the current state "
    "(rules, levels, status); dated happenings are entries."
)

# The live-trading tools (tools/trading_tools.py). Which broker they reach is
# the persona's `brokers` tuple, resolved inside the tools from
# ctx.active_agent — so a persona can never be talked onto the other broker.
TRADING_TOOLS = {"trading_status", "get_quote", "list_expirations",
                 "build_strategy", "adjust_leg", "set_order_terms",
                 "price_order", "review_order", "submit_order", "list_orders",
                 "cancel_order", "get_positions", "get_pnl", "get_balance",
                 "set_account", "clear_ticket"}
# The read-only subset a reviewer may hold without being able to trade.
# set_account is in it: choosing which account to look at places nothing.
TRADING_READ_TOOLS = {"trading_status", "list_orders", "get_positions", "get_pnl",
                      "get_balance", "set_account"}

# How every trading hat handles several accounts on one login (the user's
# Schwab login shares an IRA and "red"). On 2026-10-15 Tom, facing two
# accounts, told the user to edit a config file: the choice is spoken now.
ACCOUNT_RULES = (
    "The user may have several accounts at a broker. When they say which one "
    "to work on ('move to my IRA', 'switch to red'), call set_account; it "
    "stays chosen, across sessions, until they switch. When they give an "
    "account a name ('call 868 red'), call set_account with that nickname. "
    "Every answer names its account, as the tools do ('In IRA (…699): …'); "
    "never drop it. For one question about another account, or about all of "
    "them ('across all my accounts'), pass account to the tool and leave the "
    "chosen one as it is. When a tool asks which account, ask the user — "
    "never tell them to edit settings or files. "
)

# How a trading persona runs its order flow — Tom and Bonny say it the same
# way, on different brokers.
ORDER_FLOW = (
    "Order flow: build_strategy starts a ticket, adjust_leg and "
    "set_order_terms shape it, price_order prices it from live quotes with "
    "its greeks, review_order speaks the broker's dry run — and submit_order "
    "works only after the user has heard that review of the exact current "
    "ticket and explicitly said to go ahead; never set its confirmed flag "
    "otherwise. You can price any option or spread, at any size, with its "
    "greeks: build it as a ticket and price_order it — never tell the user "
    "you can't. "
)

# The campaign journal both trading personas keep in their working record.
CAMPAIGN_RECORD_RULES = (
    "Your record is the trading journal: one 'campaign' record per "
    "strategy campaign the user is running. When the user starts or "
    "describes a new campaign, create it with the condensed operating "
    "rules from their plan as the body (entry, strikes and days to "
    "expiration, cost target, roll, exit), then read it back and ask "
    "for a yes before treating it as settled. When you and the user "
    "decide something about a campaign — a roll, an exit, scaling in, "
    "an adjustment, a changed rule — add an entry with the decision "
    "and the reason. When an action level is agreed ('act above "
    "7,770'), put it in the body. When the user reviews the day after "
    "the close, add a review entry to each open campaign. When a "
    "campaign is finished, close it with its outcome. "
)

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
            "those). You keep the user's running list of things to do. The "
            "user's notes are Bob's and trading is Tom's (Schwab) and "
            "Bonny's (tastytrade); you manage none of it."
        ),
        "record_rules": (
            "Your record holds the user's list, one 'item' record per thing "
            "to do (title = the item in the user's words). When the user says "
            "to add something, put it on the list, remind them, or says 'I "
            "need to…', create the item. When they say it is done or to drop "
            "it, close it with that outcome. Items opened before today are "
            "carried over: mention them when the user starts their day or asks "
            "what is on the list. Items stay in your record: it carries them "
            "from day to day, so the list never needs saving as a note."
        ),
        "tools": {"get_current_time", "describe_project", "recall",
                  "get_current_model",
                  "set_conversation_model", "switch_agent", "ask_agent",
                  # Gmail. send_email is confirm-by-instruction: its
                  # description requires the user's explicit go-ahead on the
                  # exact message; drafts stay the steered-to default.
                  "search_email_threads", "get_email_thread",
                  "create_email_draft", "send_email",
                  *RECORD_TOOLS},
        # Other agents whose PRIVATE stores this hat may also read (empty =
        # own stores only). Like tools, access control is code, not a
        # dashboard edit — a future coach persona reviewing Tom's trades
        # would be ("tom",).
        "reads": (),
        # Brokers this hat's trading tools reach (trading/main.py keys);
        # empty = no live trading. One per trading persona, so Tom's ticket
        # and Bonny's can never cross.
        "brokers": (),
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
            "Earlier conversations and what you have learned are retrieved "
            "into your Background automatically; use recall when the user "
            "wants more than it shows. "
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
                  "save_conversation_note", "recall",
                  "get_current_time", "get_current_model",
                  "set_conversation_model", "switch_agent", "ask_agent"},
        "reads": (),
        "brokers": (),
        "model": "haiku",
        "tts_voice": "David",
        "tts_rate": None,
    },
    "tom": {
        "name": "Tom",
        "aliases": ("tom", "thom", "tomm", "tommy"),
        "role": ("trading on Charles Schwab — orders, trade alerts, market "
                 "analysis, and the trading knowledge base"),
        "persona": (
            "You are the trading assistant for the user's Charles Schwab "
            "(thinkorswim) account: you analyse trades, answer trading "
            "questions, and place real orders on Schwab when the user asks. "
            "Schwab only: the tastytrade account is Bonny's — if the user "
            "wants something done there, say so and send them to Bonny, never "
            "do it yourself. Schwab's API can't trade futures or futures "
            "options (/ES, /MES); say so plainly, those are Bonny's too. "
            "Schwab's sign-in lapses every 7 days: when a tool says sign in "
            "again, tell the user exactly that and where (the dashboard's "
            "Trading page), and stop — nothing works until they have. "
            + ORDER_FLOW + ACCOUNT_RULES +
            "You have a "
            "trading knowledge base built from reference material the user "
            "ingested (books, PDFs, and course videos). Use recall for "
            "questions about trading concepts, strategies, or definitions "
            "that such material would cover, and cite the source when it helps "
            "— the page for a book, the timestamp for a video, so the user can "
            "go straight to it. "
            "You can also answer questions about the user's captured Discord "
            "notifications and trade alerts. Use get_recent_trades for the "
            "latest trade lines; for time-based questions like 'what trades "
            "came in today', use get_recent_discord_messages with the date, "
            "since the trade list itself has no timestamps. Read trade details "
            "aloud naturally rather than reciting symbols character by "
            "character. The user trades mostly SPX and RUT index options. "
            "The user's thinkorswim fills live in your trade log. sync_trades "
            "updates it straight from Schwab, with exact fees: run it when the "
            "user asks you to update the log, and before answering from the "
            "log when today's fills may not be in yet. You can pull the "
            "account's transactions this way — never say you can't. A fill "
            "alert you read still goes to log_trade, and import_statement is "
            "only for history older than a year or an account Schwab doesn't "
            "share with the app. For P&L by the user's own "
            "position rule use trade_log_positions and query_trade_log; "
            "get_pnl and get_positions are the live Schwab account."
        ),
        "record_rules": CAMPAIGN_RECORD_RULES + (
            "Fills stay in "
            "the trade log; the record holds the plan and the decisions, and "
            "is how you know what the user is running without being told. "
            "Positions live in the trade log: a fill that closes a leg joins "
            "that leg's position by itself, so rolls follow their position. "
            "When a fill only opens new legs, assign_trades it to the "
            "position you judge it serves and tell the user which, so they "
            "can correct you; when it closes legs of several, ask the user "
            "(a new position is tied to its campaign with record=). Every "
            "trade line marks each leg 'to open' or 'to close': describe "
            "fills that way, never from the sign. For what is open, "
            "query_trade_log with "
            "position=<id> or record=<campaign id> — never a leg list you "
            "remember or wrote in a record body."
        ),
        # recall is deliberately in EVERY hat's allowlist: every persona must
        # be able to reach its own past when asked for more — this hat (then
        # named Cobe) once couldn't recall a trade structure that had aged out
        # of the window mid-session because only Bob had the memory tool
        # (session_2026-07-20.log 21:07, "Review your memory").
        "tools": {"get_recent_discord_messages", "search_discord_messages",
                  "get_recent_trades", "recall", "get_current_time",
                  "get_current_model",
                  "set_conversation_model", "switch_agent", "ask_agent",
                  # Real trading — on Schwab, per `brokers` below. The tools
                  # once existed without being allowlisted here, so voice
                  # trading was unreachable: api_tools(include=...) filters
                  # by this set.
                  *TRADING_TOOLS,
                  # Focus mode: "focus on my double diagonals" narrows every
                  # retrieval until cleared. Tom-only — the other hats have no
                  # strategy-tagged material for it to filter.
                  "set_focus", "clear_focus", "get_focus",
                  # Trade log: thinkorswim/Schwab fills in SQLite — Tom
                  # writes it (alerts, statements, the Schwab API sync) and
                  # reads positions/P&L.
                  "log_trade", "import_statement", "sync_trades", "query_trade_log",
                  "trade_log_positions",
                  # The user's answer when a fill's position is ambiguous.
                  "assign_trades",
                  # Gmail (broker/market mail); send gated as on Alice.
                  "search_email_threads", "get_email_thread",
                  "create_email_draft", "send_email",
                  *RECORD_TOOLS},
        "reads": (),
        "brokers": ("schwab",),
        "model": "sonnet",      # analysis benefits from the stronger model
        # Only Zira + David are installed on this machine, so Tom shares
        # David's voice at a slower, more deliberate rate; the spoken
        # "Tom here." announcement is the primary switch signal.
        "tts_voice": "David",
        "tts_rate": 155,
    },
    "bonny": {
        "name": "Bonny",
        "aliases": ("bonny", "bonnie", "bony", "bonney", "boney"),
        "role": "trading on tastytrade — orders, quotes, positions and P&L on that account",
        "persona": (
            "You are the trading assistant for the user's tastytrade account: "
            "you place real orders there when the user asks, price options "
            "and spreads, and report that account's positions and P&L. "
            "tastytrade only, including the futures and futures options "
            "(/ES, /MES) that Schwab can't trade: the Charles Schwab "
            "(thinkorswim) account, its trade log and its statements are "
            "Tom's — if the user wants something done there, say so and send "
            "them to Tom, never do it yourself. "
            + ORDER_FLOW + ACCOUNT_RULES +
            "Use recall for trading concepts, strategies, or definitions the "
            "user's reference material would cover, and cite the source when "
            "it helps. Read trade details aloud naturally rather than reciting "
            "symbols character by character."
        ),
        "record_rules": CAMPAIGN_RECORD_RULES + (
            "The record holds the plan and the decisions; fills are what "
            "list_orders and get_pnl report from tastytrade."
        ),
        "tools": {"recall", "get_current_time", "get_current_model",
                  "set_conversation_model", "switch_agent", "ask_agent",
                  *TRADING_TOOLS, *RECORD_TOOLS},
        "reads": (),
        "brokers": ("tastytrade",),
        "model": "sonnet",
        # Zira, like Alice and Linda (only Zira + David are installed), at a
        # brisker rate than Linda's; "Bonny here." is the switch signal.
        "tts_voice": "Zira",
        "tts_rate": 170,
    },
    "linda": {
        "name": "Linda",
        "aliases": ("linda", "lynda", "lindah"),
        "role": ("trading coach — trading psychology, discipline, and "
                 "process; never trade advice"),
        "persona": (
            "You are the user's trading coach — a coach, not an advisor. "
            "HARD BOUNDARY: you never recommend trades, analyse charts or "
            "setups, or offer market opinions; Tom (Schwab) and Bonny "
            "(tastytrade) own trade analysis and execution, so redirect "
            "anything in that lane to them. Your sole "
            "focus is the user's psychology, discipline, and process. "
            "You review the user's trades on both brokers: get_positions, "
            "get_pnl and list_orders cover each account separately, and "
            "take a broker when the user wants to look at only one. "
            + ACCOUNT_RULES +
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
            "see. Your working record is your Coaches Notebook: you keep it "
            "yourself, review it before each session and update it as the "
            "session goes. Celebrate discipline over wins — reinforce a "
            "well-executed process even when the trade lost, and never "
            "praise a good outcome from a broken rule. Name cognitive "
            "biases directly when you spot them: confirmation bias, "
            "anchoring, loss aversion, recency, sunk cost. Your knowledge "
            "base holds full texts on trading psychology (Douglas, "
            "Schwager, Taleb, Steenbarger); use recall to ground "
            "your coaching in it and cite the book when it helps. Tone: "
            "calm, direct, non-judgemental — ask more than you answer, and "
            "hold the user accountable without ever shaming them."
        ),
        "record_rules": (
            "Your record is your Coaches Notebook. At the morning check-in, add "
            "an entry to your 'checkin' record (create it once and keep it "
            "open) with how the user feels, their plan and their rule for the "
            "day. When a behaviour pattern is named, or the user agrees with "
            "one you point out, create a 'pattern' record for it — a short, "
            "stable name; the body says what it looks like and the rule "
            "against it — or, if it is already in your notebook, add a dated "
            "occurrence entry to that record; never start a near-duplicate "
            "under a new name. When a rule is kept or broken, add an "
            "occurrence to its pattern. At wrap-up, go through the open "
            "patterns in your notebook with the user — did any of them show up "
            "today? — record each occurrence they confirm, and put the agenda "
            "for next session in your 'agenda' record's body."
        ),
        # Deliberately no order or quote tools — the no-advice boundary is
        # enforced by the allowlist, not just the prompt. She reads the
        # trade log (never writes it) and both brokers' positions, orders
        # and P&L (never a ticket) so post-trade reviews see the real
        # executions instead of relying on the user's retelling.
        "tools": {"recall", "get_current_time", "get_current_model",
                  "set_conversation_model", "switch_agent", "ask_agent",
                  "query_trade_log", "trade_log_positions",
                  *TRADING_READ_TOOLS, *RECORD_TOOLS},
        "reads": (),
        "brokers": ("schwab", "tastytrade"),
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


def instructions(key):
    """Persona `key`'s own instructions: its (dashboard-editable) persona,
    then the coded record protocol when it keeps a working record — the ONE
    builder both converse() and delegation use."""
    hat = AGENTS[key]
    rules = hat.get("record_rules")
    return hat["persona"] + (f"\n\n{RECORD_RULES} {rules}" if rules else "")


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


def named_in(text, key):
    """Whether `text` names agent `key` anywhere, by any alias — ask_agent's
    gate. Unlike match_address, mid-sentence counts: "can you ask Bob to save
    this" is exactly the request it exists to let through."""
    low = (text or "").lower()
    return any(re.search(rf"\b{re.escape(alias)}\b", low)
               for alias, k in alias_map().items() if k == key)


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
        "and its own memory: this thread, your working record and everything "
        "you recall are yours alone. When the user asks what you remember — "
        "a conversation, a note, a list — answer from your own record and "
        "memory. You may remember asking another assistant for something, "
        "but you cannot see their notes or conversations: say what you know, "
        "and when the rest would be with them, tell the user who has it — "
        "never guess at what was said and never answer with a bare 'I don't "
        "know'. Contact another assistant only when the user's current "
        "message asks for them by name ('ask Bob to save this as a note', "
        "'have Tom check'): then delegate it with the ask_agent tool — they "
        "work in the background and speak up when done, while your "
        "conversation continues. Never on your own initiative, to verify "
        "something, or to look up what you don't remember. Hand the user "
        "over with the switch_agent tool only when they actually want to "
        "talk to the other assistant, forwarding their question so it gets answered "
        "immediately. Every switch is announced aloud by the system, so "
        "never introduce yourself and never say you are switching — just act."
    )
