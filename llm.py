"""Claude integration: conversation (with note-access tools) and summarisation."""

import json
import logging
from datetime import datetime

import anthropic

import config as cfg
from discord_data import DiscordData
from knowledge import KnowledgeStore
from memory import ConversationMemory

log = logging.getLogger("llm")

TOOLS = [
    {
        "name": "search_notes",
        "description": (
            "Semantic search across saved notes. Use for questions like 'what did "
            "I say about X' or to find notes on a topic. Pass folder to limit the "
            "search to one folder (e.g. 'what did I note about spreads in my "
            "Trading folder'); omit it to search everywhere."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "folder": {
                    "type": "string",
                    "description": "Optional: only search this folder, e.g. 'Trading'.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_recent_notes",
        "description": (
            "List the most recent notes, newest first. Use for 'what's my latest "
            "note' or 'what have I recorded recently'. Pass folder to scope it "
            "(e.g. 'what's the latest note in my General folder'); omit it for "
            "all folders."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many to list (default 5)"},
                "folder": {
                    "type": "string",
                    "description": "Optional: only list notes from this folder, e.g. 'General'.",
                },
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
        "name": "search_past_conversations",
        "description": (
            "Search archived summaries of older conversations — ones that have "
            "aged out of the current chat window. Use for 'what did we talk about "
            "last week/month', 'didn't we discuss X before', or any question about "
            "a past conversation you don't see in the current history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic to look for in past conversations"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_knowledge",
        "description": (
            "Search the user's ingested trading reference material (PDFs and text "
            "files) for relevant passages. Use for questions about trading concepts, "
            "strategies, definitions, or 'what does my trading book say about X'. "
            "Returns excerpts with their source (and page, for PDFs) so you can cite them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up in the trading material"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_folder",
        "description": (
            "Create a new note folder/category. Use when the user asks to make, "
            "add, or create a folder (e.g. 'create a folder called Recipes'). Pass "
            "the spoken name; optionally a short description of what belongs in it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name for the new folder, e.g. 'Recipes'"},
                "description": {
                    "type": "string",
                    "description": "Optional: what kind of notes belong in this folder.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "rename_folder",
        "description": (
            "Rename an existing note folder/category. Use when the user asks to "
            "rename or change a folder's name (e.g. 'rename Ideas to Brainstorms'). "
            "Notes already filed there are kept. Pass the current name and the new name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "current": {"type": "string", "description": "The current folder name, e.g. 'Ideas'"},
                "new_name": {"type": "string", "description": "The new name, e.g. 'Brainstorms'"},
            },
            "required": ["current", "new_name"],
        },
    },
    {
        "name": "save_conversation_note",
        "description": (
            "Save something from this conversation as a note. ONLY call this when "
            "the user explicitly asks (e.g. 'save that as a note', 'make a note of "
            "that'); never call it proactively or suggest saving on your own. Write "
            "the note content yourself from the conversation — clean markdown with a "
            "Summary section and Key Points. After calling this, reply with one "
            "short acknowledgement only; the system will ask the user which "
            "folder to file it in, so never ask about folders yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short descriptive title, max ~8 words"},
                "content": {
                    "type": "string",
                    "description": "The full note body in markdown, drawn from the conversation.",
                },
                "spoken_summary": {
                    "type": "string",
                    "description": "1-2 plain sentences recapping the note, to be read aloud after saving.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional: the folder slug that seems the best fit.",
                },
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "delete_folder",
        "description": (
            "Delete a note folder/category. Use when the user asks to delete or "
            "remove a folder. Notes in it are never lost — they're moved to General "
            "by default, or to 'move_notes_to' if the user names a destination. The "
            "General folder itself can't be deleted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The folder to delete, e.g. 'Recipes'"},
                "move_notes_to": {
                    "type": "string",
                    "description": "Optional folder to move any notes into before deleting (defaults to General).",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "move_note",
        "description": (
            "Move a single saved note into a different folder. First find the note's "
            "id with search_notes or list_recent_notes (they show it in [brackets]), "
            "then call this with that id and the destination folder. Use for 'move my "
            "last note to Ideas' or 'put the grocery note in Recipes'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "description": "The note id, e.g. note_2026-06-22_141500"},
                "to_folder": {"type": "string", "description": "Destination folder name, e.g. 'Ideas'"},
            },
            "required": ["note_id", "to_folder"],
        },
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
    def __init__(self, store, idle=None, kb=None):
        self.client = anthropic.Anthropic()
        self.store = store
        self.discord = DiscordData()
        # The agent shares its single KnowledgeStore (same one used for boot-time
        # ingestion) so the embedding model loads at most once per process; selftest
        # passes none, so fall back to a fresh instance.
        self.kb = kb if kb is not None else KnowledgeStore()
        # Long-term memory must exist before the history loads: anything the
        # rolling window drops is staged into it rather than lost.
        self.memory = ConversationMemory()
        # Conversation memory: restore the last conversation (trimmed) so the agent
        # remembers it across restarts; saved back to disk after every turn.
        self.history = self._load_history()
        # Set by the save_conversation_note tool; the agent picks it up after the
        # reply and runs the folder dialogue + save (see voice_agent).
        self.pending_note = None
        # Looped while we wait on the model, so the user hears the agent thinking.
        self.idle = idle if idle is not None else _NullIdle()

    # --- conversation --------------------------------------------------------
    def _dispatch(self, name, args):
        try:
            if name == "search_notes":
                return self.store.search_notes(args["query"], folder=args.get("folder"))
            if name == "list_recent_notes":
                return self.store.list_recent_notes(
                    int(args.get("n", 5)), folder=args.get("folder")
                )
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
            if name == "search_knowledge":
                return self.kb.search(args["query"])
            if name == "search_past_conversations":
                return self.memory.search(args["query"])
            if name == "create_folder":
                return self.store.create_folder(args["name"], args.get("description"))
            if name == "rename_folder":
                return self.store.rename_folder(args["current"], args["new_name"])
            if name == "delete_folder":
                return self.store.delete_folder(args["name"], args.get("move_notes_to"))
            if name == "move_note":
                return self.store.move_note(args["note_id"], args["to_folder"])
            if name == "count_notes":
                return self.store.count_notes(args.get("folder"))
            if name == "save_conversation_note":
                title = (args.get("title") or "").strip() or "Conversation note"
                content = (args.get("content") or "").strip()
                if not content:
                    return "No content provided — include the note body in 'content'."
                self.pending_note = {
                    "title": title,
                    "content": content,
                    "spoken": (args.get("spoken_summary") or "").strip(),
                    "category": args.get("category"),
                }
                return ("Note prepared. Acknowledge briefly; the system will now ask "
                        "the user which folder to file it in.")
        except Exception as e:  # surface tool errors back to the model
            return f"Tool error: {e}"
        return f"Unknown tool: {name}"

    def discard_last_turn(self):
        """Erase the most recent exchange — the last plain-text user message and
        everything after it (assistant reply, tool calls/results). Used when the
        user kept talking while the model was thinking: that reply was never
        spoken, and the combined utterance replaces the whole turn. Any note the
        discarded reply prepared is dropped with it."""
        for i in range(len(self.history) - 1, -1, -1):
            m = self.history[i]
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                del self.history[i:]
                break
        self.pending_note = None
        self._save_history()

    def take_pending_note(self):
        """Hand the pending conversation note (if any) to the agent, clearing it."""
        pending, self.pending_note = self.pending_note, None
        return pending

    def consolidate_memory(self):
        """Fold staged (aged-out) conversation text into long-term memory. Run at
        boot; a no-op unless enough has accumulated. Failures keep the staging
        file intact, so nothing is lost when offline."""
        try:
            return self.memory.consolidate(self.client)
        except Exception as e:
            log.warning("memory consolidation failed (will retry next boot): %s", e)
            return None

    # --- persistent conversation memory ---------------------------------------
    @staticmethod
    def _dump_block(block):
        """Content blocks from the SDK are pydantic models; store them as plain
        dicts so the history is JSON-serializable (the API accepts dicts back)."""
        return block if isinstance(block, dict) else block.model_dump(exclude_none=True)

    @staticmethod
    def _sanitize(history):
        """Make a history safe to send. Drops any tool_use whose tool_result never
        arrived (and any tool_result with no matching tool_use), then merges
        adjacent same-role turns. Without this, a conversation persisted
        mid-tool-loop — e.g. a turn abandoned between the tool call and its result
        by the background-thread/barge-in path — replays into a 400 'tool_use ids
        were found without tool_result blocks'. Because the bad history reloads on
        every launch, that 400 bricks the app until the file is repaired; this
        makes it self-heal instead."""
        result_ids, use_ids = set(), set()
        for m in history:
            c = m.get("content")
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict):
                        if b.get("type") == "tool_result":
                            result_ids.add(b.get("tool_use_id"))
                        elif b.get("type") == "tool_use":
                            use_ids.add(b.get("id"))
        cleaned = []
        for m in history:
            c = m.get("content")
            if isinstance(c, list):
                blocks = []
                for b in c:
                    if isinstance(b, dict):
                        if b.get("type") == "tool_use" and b.get("id") not in result_ids:
                            continue  # unanswered tool call — its result was lost
                        if b.get("type") == "tool_result" and b.get("tool_use_id") not in use_ids:
                            continue  # result with no surviving tool call
                    blocks.append(b)
                if not blocks:
                    continue  # message had only orphaned blocks — drop it whole
                m = {**m, "content": blocks}
            cleaned.append(m)
        # Dropping a message can leave two same-role turns adjacent, which the API
        # also rejects; fold them together so roles keep alternating.
        to_list = lambda c: [{"type": "text", "text": c}] if isinstance(c, str) else list(c)
        coalesced = []
        for m in cleaned:
            if coalesced and coalesced[-1]["role"] == m["role"]:
                prev = coalesced[-1]
                pc, mc = prev["content"], m["content"]
                if isinstance(pc, str) and isinstance(mc, str):
                    prev["content"] = f"{pc}\n\n{mc}"
                else:
                    prev["content"] = to_list(pc) + to_list(mc)
            else:
                content = m["content"] if isinstance(m["content"], str) else list(m["content"])
                coalesced.append({**m, "content": content})
        return coalesced

    @staticmethod
    def _trim_history(history):
        """Cap the history and make sure it starts on a plain user message — a
        trim boundary must never orphan a tool_result from its tool_use, or the
        API rejects the conversation."""
        h = list(history[-cfg.HISTORY_MAX_MESSAGES:])
        while h and not (h[0].get("role") == "user" and isinstance(h[0].get("content"), str)):
            h.pop(0)
        return h

    def _trim_and_archive(self, history):
        """Trim to the rolling window, staging whatever falls off into long-term
        memory instead of discarding it. The kept part is always a contiguous
        suffix, so the dropped prefix is everything before it."""
        history = self._sanitize(history)  # never carry an orphaned tool call forward
        kept = self._trim_history(history)
        dropped = history[:len(history) - len(kept)]
        if dropped:
            try:
                self.memory.record_dropped(dropped)
            except Exception as e:  # staging must never break the conversation
                log.warning("could not stage dropped history: %s", e)
        return kept

    def _load_history(self):
        try:
            if cfg.HISTORY_PATH.exists():
                h = json.loads(cfg.HISTORY_PATH.read_text(encoding="utf-8"))
                if isinstance(h, list) and h:
                    h = self._trim_and_archive(h)
                    log.info("restored %d message(s) of conversation history", len(h))
                    return h
        except (OSError, ValueError) as e:
            log.warning("could not load conversation history: %s", e)
        return []

    def _save_history(self):
        # Saved untrimmed: trimming happens on load / at each turn, where the
        # dropped part is staged into long-term memory. Trimming here instead
        # would silently discard the overflow on quit. Sanitized, though, so a
        # turn abandoned mid-tool-loop can never persist an orphaned tool_use
        # that would 400 (and brick) the next launch.
        try:
            cfg.HISTORY_PATH.write_text(
                json.dumps(self._sanitize(self.history), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, TypeError) as e:
            log.warning("could not save conversation history: %s", e)

    def converse(self, user_text: str) -> str:
        # Trim in memory too, so a long-running session doesn't grow unbounded;
        # whatever falls off is staged into long-term memory, not lost.
        self.history = self._trim_and_archive(self.history)
        self.history.append({"role": "user", "content": user_text})
        # If the previous turn was abandoned right after its user message (leaving
        # history ending on a user turn), this new message would be a second
        # consecutive user turn — which the API also rejects. Fold them together.
        self.history = self._sanitize(self.history)
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
                self.history.append(
                    {"role": "assistant",
                     "content": [self._dump_block(b) for b in resp.content]}
                )

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
            self._save_history()  # every turn — survives crashes and quits alike

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
        # save_conversation_note is excluded here: we're already filing a note,
        # so triggering another pending note mid-dialogue would be circular.
        tools = [t for t in TOOLS if t["name"] != "save_conversation_note"] + [choose_tool]
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
