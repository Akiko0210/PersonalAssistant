"""Claude integration: conversation (with note-access tools) and summarisation."""

import logging

import anthropic

import categories
import config as cfg
import history as hist
from discord_data import DiscordData
from knowledge import KnowledgeStore
from memory import ConversationMemory
from tools import ToolContext, api_tools, dispatch

log = logging.getLogger("llm")


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
        # Everything tool handlers may touch (see tools/); also carries the
        # pending conversation note that save_conversation_note prepares.
        self._ctx = ToolContext(store=self.store, discord=self.discord,
                                kb=self.kb, memory=self.memory)
        # Conversation memory: restore the last conversation (trimmed) so the agent
        # remembers it across restarts; saved back to disk after every turn.
        self.history = self._load_history()
        # Looped while we wait on the model, so the user hears the agent thinking.
        self.idle = idle if idle is not None else _NullIdle()

    # Set by the save_conversation_note tool; the agent picks it up after the
    # reply and runs the folder dialogue + save (see voice_agent).
    @property
    def pending_note(self):
        return self._ctx.pending_note

    @pending_note.setter
    def pending_note(self, value):
        self._ctx.pending_note = value

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

    def _trim_and_archive(self, history):
        """Sanitize + trim to the rolling window, staging whatever falls off into
        long-term memory instead of discarding it. The kept part is always a
        contiguous suffix, so the dropped prefix is everything before it."""
        history = hist.sanitize(history)  # never carry an orphaned tool call forward
        kept = hist.trim(history, cfg.HISTORY_MAX_MESSAGES)
        dropped = history[:len(history) - len(kept)]
        if dropped:
            try:
                self.memory.record_dropped(dropped)
            except Exception as e:  # staging must never break the conversation
                log.warning("could not stage dropped history: %s", e)
        return kept

    def _load_history(self):
        h = hist.load(cfg.HISTORY_PATH)
        if h:
            h = self._trim_and_archive(h)
            log.info("restored %d message(s) of conversation history", len(h))
        return h

    def _save_history(self):
        # Saved untrimmed: trimming happens on load / at each turn, where the
        # dropped part is staged into long-term memory. Trimming here instead
        # would silently discard the overflow on quit.
        hist.save(cfg.HISTORY_PATH, self.history)

    def converse(self, user_text: str) -> str:
        # Trim in memory too, so a long-running session doesn't grow unbounded;
        # whatever falls off is staged into long-term memory, not lost.
        self.history = self._trim_and_archive(self.history)
        self.history.append({"role": "user", "content": user_text})
        # If the previous turn was abandoned right after its user message (leaving
        # history ending on a user turn), this new message would be a second
        # consecutive user turn — which the API also rejects. Fold them together.
        self.history = hist.sanitize(self.history)
        self.idle.start()  # thinking — keep it looping across the whole tool loop
        try:
            while True:
                resp = self.client.messages.create(
                    model=cfg.CONVO_MODEL,
                    max_tokens=cfg.CONVO_MAX_TOKENS,
                    system=cfg.CONVO_SYSTEM,
                    tools=api_tools(),
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
                            out = dispatch(self._ctx, block.name, block.input)
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
            for slug, meta in categories.NOTE_CATEGORIES.items()
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
        category = categories.DEFAULT_CATEGORY
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
                if slug in categories.NOTE_CATEGORIES:
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
            for slug, m in categories.NOTE_CATEGORIES.items()
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
            f"Suggested folder: {suggested} ({categories.NOTE_CATEGORIES[suggested]['display']}).\n\n"
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
        tools = api_tools(exclude={"save_conversation_note"}) + [choose_tool]
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
                                            "content": dispatch(self._ctx, block.name, block.input)})
                    if chosen is not None:
                        if chosen in categories.NOTE_CATEGORIES:
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
