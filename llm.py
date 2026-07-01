"""Claude integration: conversation (with note-access tools) and summarisation."""

import logging
from datetime import datetime

import anthropic

import config as cfg
from discord_data import DiscordData

log = logging.getLogger("llm")

TOOLS = [
    {
        "name": "search_notes",
        "description": (
            "Semantic search across all saved notes. Use for questions like "
            "'what did I say about X' or to find notes on a topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_recent_notes",
        "description": (
            "List the most recent notes, newest first. Use for 'what's my latest "
            "note' or 'what have I recorded recently'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many to list (default 5)"}
            },
        },
    },
    {
        "name": "read_note",
        "description": "Read the full saved summary for a note id (e.g. note_2026-06-22_141500).",
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "description": "The note id to read"}
            },
            "required": ["note_id"],
        },
    },
    {
        "name": "get_recent_discord_messages",
        "description": (
            "List recent Discord notifications the user has captured (newest at the "
            "end). Use for 'what came in today', 'latest Discord messages', or any "
            "time-based question. trades.txt has no timestamps, so use this with a "
            "date to answer 'what trades came in today'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many to list (default 10)"},
                "date": {
                    "type": "string",
                    "description": "Optional day filter: 'today', 'yesterday', or YYYY-MM-DD",
                },
            },
        },
    },
    {
        "name": "search_discord_messages",
        "description": (
            "Search captured Discord notifications by keyword/topic and/or sender. "
            "Use for 'what was said about SPX', 'any iron condors', or "
            "'what did Dan Sheridan say'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword or topic to search for"},
                "sender": {
                    "type": "string",
                    "description": "Optional: only messages from this sender (substring match)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_recent_trades",
        "description": (
            "List the most recent trade-ready lines captured from Discord trade "
            "alerts. Use for 'what are the latest trades' or 'recent Discord trades'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many to list (default 10)"}
            },
        },
    },
    {
        "name": "get_current_time",
        "description": (
            "Get the current date, time, and day of the week. Use whenever the user "
            "asks what time or day it is, or when you need today's date to reason "
            "about recency (e.g. 'today', 'this week', 'how long ago')."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_folders",
        "description": (
            "List the note folders/categories available to file notes into, with a "
            "short description of what belongs in each. Use for 'what folders do I "
            "have', 'where can I put my notes', or 'what categories are there'."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "count_notes",
        "description": (
            "Count saved notes, optionally within one category folder. Use for "
            "'how many notes do I have' or 'how many notes are in my trading folder'. "
            "Omit folder for a total with a per-category breakdown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "Category/folder name, e.g. 'Trading'. Omit to count all.",
                }
            },
        },
    },
]

SUMMARY_PROMPT = """You are summarising a spoken notetaking session. The text below is an \
automatic transcript and may contain disfluencies or recognition errors — clean it up \
sensibly without inventing content.

Respond in EXACTLY this format:

TITLE: <a short descriptive title, max ~8 words>
SPOKEN: <a 2-3 sentence spoken recap that will be read aloud to the user; plain sentences, no markdown>
CATEGORY: <the single best-fitting category slug from the list below>
---
## Summary
<a tight prose summary>

## Key Points
- <point>

## Action Items
- <action item, or "None">

Categories (choose exactly one slug for CATEGORY):
{categories}

Transcript:
"""


class _NullIdle:
    """No-op stand-in so Claude runs without an idle-sound controller (selftest)."""

    def start(self):
        pass

    def stop(self):
        pass


class Claude:
    def __init__(self, store, idle=None):
        self.client = anthropic.Anthropic()
        self.store = store
        self.discord = DiscordData()
        self.history = []
        # Looped while we wait on the model, so the user hears the agent thinking.
        self.idle = idle if idle is not None else _NullIdle()

    # --- conversation --------------------------------------------------------
    def _dispatch(self, name, args):
        try:
            if name == "search_notes":
                return self.store.search_notes(args["query"])
            if name == "list_recent_notes":
                return self.store.list_recent_notes(int(args.get("n", 5)))
            if name == "read_note":
                return self.store.read_note(args["note_id"])
            if name == "get_recent_discord_messages":
                return self.discord.recent_messages(
                    int(args.get("n", 10)), args.get("date")
                )
            if name == "search_discord_messages":
                return self.discord.search_messages(
                    args["query"], args.get("sender")
                )
            if name == "get_recent_trades":
                return self.discord.recent_trades(int(args.get("n", 10)))
            if name == "get_current_time":
                return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
            if name == "list_folders":
                return self.store.list_folders()
            if name == "count_notes":
                return self.store.count_notes(args.get("folder"))
        except Exception as e:  # surface tool errors back to the model
            return f"Tool error: {e}"
        return f"Unknown tool: {name}"

    def converse(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        self.idle.start()  # thinking — keep it looping across the whole tool loop
        try:
            while True:
                resp = self.client.messages.create(
                    model=cfg.CONVO_MODEL,
                    max_tokens=cfg.CONVO_MAX_TOKENS,
                    system=cfg.CONVO_SYSTEM,
                    tools=TOOLS,
                    thinking={"type": "disabled"},
                    messages=self.history,
                )
                self.history.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason == "tool_use":
                    results = []
                    for block in resp.content:
                        if block.type == "tool_use":
                            log.info("tool_use %s %s", block.name, block.input)
                            out = self._dispatch(block.name, block.input)
                            results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": out,
                            })
                    self.history.append({"role": "user", "content": results})
                    continue

                return "".join(b.text for b in resp.content if b.type == "text").strip()
        finally:
            self.idle.stop()

    # --- summarisation -------------------------------------------------------
    @staticmethod
    def _category_guidance() -> str:
        return "\n".join(
            f"- {slug}: {meta['description']}"
            for slug, meta in cfg.NOTE_CATEGORIES.items()
        )

    def summarize(self, transcript: str):
        prompt = SUMMARY_PROMPT.format(categories=self._category_guidance())
        self.idle.start()
        try:
            resp = self.client.messages.create(
                model=cfg.SUMMARY_MODEL,
                max_tokens=cfg.SUMMARY_MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt + transcript}],
            )
        finally:
            self.idle.stop()
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return self._parse_summary(text)

    @staticmethod
    def _parse_summary(text: str):
        title = "Untitled note"
        spoken = ""
        category = cfg.DEFAULT_CATEGORY
        full = text
        if "---" in text:
            head, full = text.split("---", 1)
            full = full.strip()
        else:
            head = text
        for line in head.splitlines():
            if line.upper().startswith("TITLE:"):
                title = line.split(":", 1)[1].strip() or title
            elif line.upper().startswith("SPOKEN:"):
                spoken = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CATEGORY:"):
                slug = line.split(":", 1)[1].strip().lower()
                if slug in cfg.NOTE_CATEGORIES:
                    category = slug
        if not spoken:
            spoken = "I've saved your note."
        if not full:
            full = f"## Summary\n{spoken}"
        return title, spoken, full, category

    def choose_folder_via_dialogue(self, title, summary, suggested, ask_fn, max_turns=6):
        """Decide a note's folder through a short spoken conversation. Proposes the
        suggested folder, answers any questions the user asks (what folders exist,
        how many notes are in one, etc.) using tools, and finalizes only when the
        user clearly commits. `ask_fn(text)` speaks `text` aloud and returns the
        user's transcribed reply ("" if silent). Returns the chosen category slug."""
        folders = "\n".join(
            f"- {slug} ({m['display']}): {m['description']}"
            for slug, m in cfg.NOTE_CATEGORIES.items()
        )
        system = (
            "You are helping the user decide which folder to file a note they just "
            "recorded into. Your words are read aloud, so keep every reply to one or "
            "two short spoken sentences, no markdown. Start by telling them your "
            "suggested folder and asking whether to use it, pick another, or if they "
            "have questions. The user may chat or ask things before deciding — answer "
            "them, using tools when helpful, and do NOT finalize yet. Only when the "
            "user clearly commits to a folder, call choose_folder with its slug. If "
            "they simply agree, use your suggested folder. IMPORTANT: the ONLY way to "
            "file the note is to call choose_folder — never claim in plain text that "
            "the note is filed, saved, or done without calling the tool, and do not "
            "announce the result yourself (the system says it aloud after the tool "
            "call).\n\n"
            f"Note title: {title}\n"
            f"Note summary: {summary}\n"
            f"Suggested folder: {suggested} ({cfg.NOTE_CATEGORIES[suggested]['display']}).\n\n"
            f"Available folders:\n{folders}"
        )
        choose_tool = {
            "name": "choose_folder",
            "description": (
                "Finalize the folder for this note. Only call this once the user has "
                "clearly decided. Pass the chosen folder's slug."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "The chosen folder slug, one of the available slugs.",
                    }
                },
                "required": ["folder"],
            },
        }
        tools = TOOLS + [choose_tool]
        history = [{"role": "user",
                    "content": "I just finished recording a note. Where should it go?"}]
        try:
            for _ in range(max_turns):
                self.idle.start()  # thinking; stopped below before we speak/listen
                resp = self.client.messages.create(
                    model=cfg.CONVO_MODEL,
                    max_tokens=cfg.CONVO_MAX_TOKENS,
                    system=system,
                    tools=tools,
                    thinking={"type": "disabled"},
                    messages=history,
                )
                history.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason == "tool_use":
                    results = []
                    chosen = None
                    for block in resp.content:
                        if block.type != "tool_use":
                            continue
                        if block.name == "choose_folder":
                            chosen = block.input.get("folder", "")
                            results.append({"type": "tool_result", "tool_use_id": block.id,
                                            "content": "ok"})
                        else:
                            log.info("tool_use %s %s", block.name, block.input)
                            results.append({"type": "tool_result", "tool_use_id": block.id,
                                            "content": self._dispatch(block.name, block.input)})
                    if chosen is not None:
                        if chosen in cfg.NOTE_CATEGORIES:
                            return chosen
                        return self.store._match_category(chosen) or suggested
                    history.append({"role": "user", "content": results})
                    continue  # still thinking — keep the loop playing

                # The model wants to talk to the user: silence the cue so it
                # doesn't bleed into the spoken question or the mic.
                self.idle.stop()
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                reply = ask_fn(text or "Which folder should this go in?")
                history.append({"role": "user", "content": reply or "(no answer)"})
        finally:
            self.idle.stop()

        log.info("folder dialogue hit max turns; using suggested %s", suggested)
        return suggested
