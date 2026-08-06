"""Claude integration: conversation (with note-access tools) and summarisation.

DeepSeek models ride the same code path: their Anthropic-compatible endpoint
speaks the Messages API, so `client_for` just picks which anthropic-SDK client
a model id goes to. Nothing downstream — tool loop, history invariants, error
handling — knows more than one provider exists."""

import logging
import os
from datetime import datetime

import anthropic

import agents
import categories
import config as cfg
import history as hist
from atomic_io import write_json_atomic
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

DELEGATION_PROMPT = """You are {name} — {role}. Another assistant handed you the task below to do in \
the BACKGROUND: the user is still talking to them and cannot hear you work. Complete the task with \
your tools, then reply with the outcome in plain sentences — the assistant the user is talking to \
reads your report to them aloud at the next pause in their conversation, so write it to be read \
aloud (no markdown). Do not greet or introduce yourself; the report is announced as yours. If the \
task is to save a note, call save_conversation_note and reply with a brief acknowledgement — the \
system will ask the user which folder itself, so never mention folders."""


_CACHE_CONTROL = {"type": "ephemeral"}


def cached(history):
    """Copy of `history` carrying a prompt-cache breakpoint on its last block.

    Prompt caching is a prefix match, so one breakpoint at the end of the history
    caches everything rendered before it — tool schemas, system prompt, and the
    whole conversation so far. The breakpoint has to go here rather than on the
    system block because the static part alone is too small to cache: tools plus
    system measure 1,967-3,492 tokens depending on the agent, all under Haiku
    4.5's 4096-token minimum, below which the API caches nothing and says so only
    by leaving cache_creation_input_tokens at 0.

    This is the payoff for the tool loop in particular: one turn can make up to
    CONVO_MAX_TOOL_ROUNDS calls that each resend the entire prefix, so rounds 2..n
    read at a tenth of the input price instead of paying full freight again.

    Returns a copy — self.history is written to disk every turn, and persisting
    cache_control markers would leave stale breakpoints scattered through the
    restored history."""
    if not history:
        return history
    content = history[-1].get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content = [dict(b) if isinstance(b, dict) else b for b in content]
    else:
        return history
    if not content or not isinstance(content[-1], dict):
        return history  # SDK block objects are immutable; skip rather than crash
    content[-1] = {**content[-1], "cache_control": _CACHE_CONTROL}
    return history[:-1] + [{**history[-1], "content": content}]


class _NullIdle:
    """No-op stand-in so Claude runs without an idle-sound controller (selftest)."""

    def start(self):
        pass

    def stop(self):
        pass


class Claude:
    def __init__(self, store, idle=None, kb=None):
        self.client = anthropic.Anthropic()
        self._deepseek = None  # lazily built by client_for; needs DEEPSEEK_API_KEY
        self.store = store
        self.discord = DiscordData()
        # The agent shares its single KnowledgeStore (same one used for boot-time
        # ingestion) so the embedding model loads at most once per process; selftest
        # passes none, so fall back to a fresh instance.
        self.kb = kb if kb is not None else KnowledgeStore()
        # Long-term memory must exist before the history loads: anything the
        # rolling window drops is staged into it rather than lost.
        self.memory = ConversationMemory()
        # Personas ("hats") over the one shared conversation: per-agent system
        # prompt, tools, model, and voice — but a single history. The registry
        # defaults are overlaid with any dashboard edits first.
        agents.load_agents()
        self.active = agents.DEFAULT_AGENT
        # Per-hat snapshots of a mid-session set_conversation_model choice, so
        # "make Tom smarter" survives switching away and back but never
        # bleeds into the other personas.
        self._model_overrides = {}
        # Tools that make their own model calls (staged-memory recall,
        # consolidation) always run on cfg.CONVO_MODEL, so hand them the client
        # that serves it. If that model's key is missing, fall back to the
        # Anthropic client rather than refuse to boot — recall then degrades to
        # its keyword scan instead of taking the whole agent down.
        try:
            tool_client = self.client_for(cfg.CONVO_MODEL)
        except RuntimeError as e:
            log.warning("memory calls fall back to the Anthropic client: %s", e)
            tool_client = self.client
        # Everything tool handlers may touch (see tools/); also carries the
        # pending conversation note and the active conversation model (which the
        # set_conversation_model tool can switch mid-session).
        self._ctx = ToolContext(store=self.store, discord=self.discord,
                                kb=self.kb, memory=self.memory,
                                client=tool_client,
                                convo_model=self._registry_model(self.active),
                                active_agent=self.active)
        self._write_agent_state()
        # Conversation memory: restore the last conversation (trimmed) so the agent
        # remembers it across restarts; saved back to disk after every turn.
        self.history = self._load_history()
        # Looped while we wait on the model, so the user hears the agent thinking.
        self.idle = idle if idle is not None else _NullIdle()

    def client_for(self, model_id):
        """The API client serving `model_id`: the shared Anthropic client, or a
        lazily created one for DeepSeek's Anthropic-compatible endpoint. The
        set_conversation_model tool refuses a DeepSeek switch when the key is
        missing, so by voice this can't raise; the RuntimeError covers config
        edits (dashboard model dropdowns) that sidestep the tool."""
        if cfg.model_provider(model_id) != "deepseek":
            return self.client
        if self._deepseek is None:
            key = os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY is not set — add it to .env to use "
                    f"{model_id}, or switch back to a Claude model.")
            self._deepseek = anthropic.Anthropic(
                base_url=cfg.DEEPSEEK_BASE_URL, api_key=key)
        return self._deepseek

    # Set by the save_conversation_note tool; the agent picks it up after the
    # reply and runs the folder dialogue + save (see voice_agent).
    @property
    def pending_note(self):
        return self._ctx.pending_note

    @pending_note.setter
    def pending_note(self, value):
        self._ctx.pending_note = value

    def take_pending_note(self):
        """Hand the pending conversation note (if any) to the agent, clearing it."""
        pending, self.pending_note = self.pending_note, None
        return pending

    def conversation_excerpt(self) -> str:
        """Plain-text flatten of the current history window — 'user: …' /
        'assistant: …' lines, tool traffic skipped (same flattening the
        long-term memory staging uses). This is the SOURCE MATERIAL a
        conversation note is drawn from: it becomes the note's transcript,
        preserving what was actually said rather than a second copy of the
        model's own summary."""
        lines = [t for m in self.history
                 if (t := ConversationMemory._message_text(m))]
        return "\n\n".join(lines)

    def take_pending_switch(self):
        """Hand a pending agent switch (if any) to the agent, clearing it.
        Returns (agent_key, forward_text) or None."""
        pending, self._ctx.pending_switch = self._ctx.pending_switch, None
        return pending

    def take_pending_delegations(self):
        """Hand over (and clear) the background tasks ask_agent queued this
        turn, as a list of (agent_key, task_text)."""
        pending, self._ctx.pending_delegations = self._ctx.pending_delegations, []
        return pending

    # --- background delegation ------------------------------------------------
    def run_delegated_task(self, key, task):
        """Run `task` as persona `key` in an isolated mini-conversation, for
        the ask_agent tool. This runs on a worker thread while the main
        conversation continues, so it must not touch shared mutable state: it
        gets its own ToolContext (same stores, fresh pending/event slots) and
        its own message list. The shared history never sees this exchange —
        the agent folds the outcome back in when the result is finally spoken
        (see voice_agent's interjection delivery).

        Returns (spoken_text, pending_note): the persona's short spoken
        report, plus the prepared note dict when the task ended in
        save_conversation_note (the caller owns the folder dialogue, exactly
        as for a foreground save).

        Deliberately runs on the persona's registry-default model: a
        mid-session set_conversation_model choice belongs to the audible
        conversation, and a background task must never silently bill whatever
        expensive model a hat happens to be parked on. No idle "thinking"
        sound either — the user may be mid-sentence with the foreground
        persona."""
        hat = agents.AGENTS[key]
        model = self._registry_model(key)
        client = self.client_for(model)
        sub_ctx = ToolContext(store=self.store, discord=self.discord,
                              kb=self.kb, memory=self.memory,
                              client=self._ctx.client,
                              convo_model=model, active_agent=key)
        system = (DELEGATION_PROMPT.format(name=hat["name"], role=hat["role"])
                  + "\n\n" + hat["persona"])
        # No switching tools in the background: the worker has no user to hand
        # over or re-route, and a nested delegation could chain unboundedly.
        tools = api_tools(include=hat["tools"] - {
            "switch_agent", "ask_agent", "set_conversation_model"})
        messages = [{"role": "user", "content": task}]
        log.info("delegated task -> %s (%s)", key, model)
        for _ in range(cfg.DELEGATION_MAX_TOOL_ROUNDS):
            resp = client.messages.create(
                model=model,
                max_tokens=cfg.CONVO_MAX_TOKENS,
                system=system,
                tools=tools,
                messages=cached(messages),
                **cfg.thinking_kwargs(model),
            )
            messages.append(
                {"role": "assistant",
                 "content": [self._dump_block(b) for b in resp.content]})
            if resp.stop_reason == "tool_use":
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        log.info("delegated tool_use %s %s",
                                 block.name, block.input)
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": dispatch(sub_ctx, block.name,
                                                block.input),
                        })
                messages.append({"role": "user", "content": results})
                continue
            text = "".join(b.text for b in resp.content
                           if b.type == "text").strip()
            return (text or "the task is done.", sub_ctx.pending_note)
        log.warning("delegated task hit DELEGATION_MAX_TOOL_ROUNDS (%d); "
                    "reporting as incomplete", cfg.DELEGATION_MAX_TOOL_ROUNDS)
        return ("I ran out of steps before finishing that task — it may be "
                "incomplete.", sub_ctx.pending_note)

    # --- personas ("hats") ----------------------------------------------------
    @staticmethod
    def _registry_model(key):
        """The API model id an agent's registry entry names."""
        return cfg.CONVO_MODELS.get(agents.AGENTS[key]["model"], cfg.CONVO_MODEL)

    @property
    def active_model(self):
        """API model id the active persona answers with — its registry default
        unless a mid-session set_conversation_model override is in play. Same
        expression converse() resolves each call, exposed so callers (the switch
        announcement) can report the model without reaching into _ctx."""
        return self._ctx.convo_model or self._registry_model(self.active)

    @property
    def active_model_label(self):
        """Spoken name of the active model, e.g. "DeepSeek V4 Pro"."""
        return cfg.convo_model_label(self.active_model)

    def switch_to(self, key):
        """Make `key` the active persona: its system prompt, tool allowlist,
        and model apply from the next converse() on. The history carries over
        whole — hats share memory by design — with one marker appended naming
        who took over.

        That marker is the ONLY record of who spoke what. A message carries
        just {role, content}, and role is only ever user/assistant, so an
        assistant turn is anonymous; nothing else about a switch reaches the
        model either — the spoken "Tom here." announcement goes to the TTS and
        never to history, agent_state.json is write-only telemetry, and a
        regex-routed switch (agents.match_address) leaves no tool call behind
        the way switch_agent does. Without this, "was that Tom or Bob?" can
        only be guessed at, and the model guesses confidently: it claimed
        Haiku while Cobe was on Sonnet, off nothing but stale history
        (session_2026-07-25.log 18:08). Same failure the model-identity rules
        in cfg.model_identity_block were written for."""
        if key == self.active:
            return
        # Preserve a mid-session "switch to opus" for the hat it was made in.
        self._model_overrides[self.active] = self._ctx.convo_model
        self.active = key
        self._ctx.active_agent = key
        self._ctx.convo_model = (self._model_overrides.get(key)
                                 or self._registry_model(key))
        self._write_agent_state()
        # An assistant turn, like flush_tool_events' self-notes: the system's
        # own record of what it did, never a fabricated user utterance.
        # sanitize() folds it into the preceding reply so roles keep
        # alternating; a marker left leading after a trim is dropped by
        # hist.trim, which is correct — there is no prior speaker to attribute.
        self.history.append(
            {"role": "assistant",
             "content": f"({agents.AGENTS[key]['name']} took over the "
                        f"conversation here.)"})
        self.history = hist.sanitize(self.history)
        self._save_history()
        log.info("active agent -> %s (%s)", key, self._ctx.convo_model)

    def _write_agent_state(self):
        """Tell the dashboard who is talking. Telemetry only — never read back,
        and never allowed to break a switch."""
        try:
            write_json_atomic(cfg.AGENT_STATE_PATH, {
                "active": self.active,
                "name": agents.AGENTS[self.active]["name"],
                "since": datetime.now().isoformat(timespec="seconds"),
            })
        except OSError as e:
            log.warning("could not write agent state: %s", e)

    def record_tool_event(self, text):
        """Record a factual note about work a tool did beyond its return string
        (a deferred save, a sub-dialogue that ran in separate model memory), so
        flush_tool_events can fold it into the conversation. See ToolContext."""
        self._ctx.record_event(text)

    def flush_tool_events(self, persist=False):
        """Fold any recorded tool-activity notes into the conversation as an
        assistant self-note, so the next turn's model knows what actually
        happened inside its tool calls.

        A tool's return string is the only thing that reaches history on its
        own. Work that finishes *after* the tool returns — a note filed through
        the spoken folder dialogue — or that runs in a *separate* model memory
        never lands there, so the model keeps answering from the stale
        placeholder the tool returned mid-turn (which is how it reported a note
        as still "pending" after it had been filed). Recorded events close that
        gap.

        Called at the end of converse() for synchronous tools; called with
        persist=True from the agent after a deferred flow that finishes past
        converse()'s own save. The note is an assistant turn — it is the agent's
        own record of what it did, not a fabricated user utterance — and
        sanitize() coalesces it into the reply that precedes it."""
        events, self._ctx.events = self._ctx.events, []
        if not events:
            return
        note = ("(Note to self — what my tool actions just did, for when the "
                "user asks about them: " + " ".join(events) + ")")
        self.history.append({"role": "assistant", "content": note})
        self.history = hist.sanitize(self.history)
        if persist:
            self._save_history()

    def record_unanswered(self, user_text: str):
        """Keep a transcribed utterance in history when a hotkey cut the turn
        short before the model was called — the words must not vanish just
        because the user clicked mute mid-settle. The next turn's sanitize
        folds consecutive user turns together, so the model still sees them."""
        self.history.append({"role": "user", "content": user_text})
        self._save_history()

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
            for _ in range(cfg.CONVO_MAX_TOOL_ROUNDS):
                # Read the model fresh each pass: if set_conversation_model runs
                # during this turn's tool loop, the follow-up call (and every
                # later turn) uses the newly chosen model.
                model = self._ctx.convo_model or self._registry_model(self.active)
                hat = agents.AGENTS[self.active]
                system = (cfg.CONVO_SYSTEM_BASE + "\n\n" + hat["persona"]
                          + agents.roster_block(self.active)
                          + cfg.model_identity_block(
                              cfg.convo_model_label(model)))
                try:
                    resp = self.client_for(model).messages.create(
                        model=model,
                        max_tokens=cfg.CONVO_MAX_TOKENS,
                        system=system,
                        tools=api_tools(include=hat["tools"]),
                        messages=cached(self.history),
                        **cfg.thinking_kwargs(model),
                    )
                except anthropic.NotFoundError:
                    # A switched-to model id the API no longer serves must not
                    # brick every later turn — the voice fix (switching back)
                    # itself needs a working model call. Revert and retry.
                    fallback = self._registry_model(self.active)
                    if model != fallback:
                        log.warning("model %s rejected (not found); reverting "
                                    "to %s's default %s", model,
                                    hat["name"], fallback)
                        self._ctx.convo_model = fallback
                        continue
                    raise
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

                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                if resp.stop_reason == "max_tokens":
                    # The reply hit CONVO_MAX_TOKENS. If it died inside a tool
                    # call, that call was never dispatched — the action did NOT
                    # happen, and the dangling tool_use will be sanitized away,
                    # so the model won't even remember trying. Say so out loud;
                    # a silent empty return here cost a whole note save while
                    # the model kept announcing "saving now" (2026-07-19 log).
                    truncated_tool = any(b.type == "tool_use" for b in resp.content)
                    log.warning("reply truncated at CONVO_MAX_TOKENS "
                                "(tool call cut off: %s)", truncated_tool)
                    if truncated_tool:
                        notice = ("Sorry — that action needed a longer reply than "
                                  "I'm allowed, so it did not complete. Try asking "
                                  "for a shorter version.")
                        return f"{text} {notice}".strip()
                    if not text:
                        return ("My reply hit its length limit before I could say "
                                "anything. Try asking for a shorter version.")
                return text
            # The model kept calling tools without ever answering. Bail out with
            # an honest line rather than billing API calls forever; the next
            # turn's sanitize repairs whatever the loop left mid-flight.
            log.warning("tool loop hit CONVO_MAX_TOOL_ROUNDS (%d); bailing out",
                        cfg.CONVO_MAX_TOOL_ROUNDS)
            return ("I got stuck repeating tool calls and stopped myself. "
                    "Could you ask that again, maybe more specifically?")
        finally:
            self.idle.stop()
            # Fold in anything a synchronous tool recorded this turn, then save.
            # Deferred flows (the conversation-note save) flush themselves later
            # with persist=True, since they finish after this point.
            self.flush_tool_events()
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
            # No cache breakpoint: one call per note against a transcript that is
            # never seen twice, so a write would be paid and never read.
            resp = self.client_for(cfg.SUMMARY_MODEL).messages.create(
                model=cfg.SUMMARY_MODEL,
                max_tokens=cfg.SUMMARY_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt + transcript}],
                **cfg.thinking_kwargs(cfg.SUMMARY_MODEL, cfg.SUMMARY_EFFORT),
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
        # save_conversation_note is excluded (we're already filing a note, so a
        # second pending note mid-dialogue would be circular); so are
        # set_conversation_model, switch_agent, and ask_agent — switching
        # models or personas, or spawning background work, is out of scope for
        # the focused "where does this go" exchange.
        tools = api_tools(exclude={"save_conversation_note",
                                   "set_conversation_model",
                                   "switch_agent", "ask_agent"}) + [choose_tool]
        history = [{"role": "user",
                    "content": "I just finished recording a note. Where should it go?"}]
        try:
            for _ in range(max_turns):
                self.idle.start()  # thinking; stopped below before we speak/listen
                model = self._ctx.convo_model or cfg.CONVO_MODEL
                resp = self.client_for(model).messages.create(
                    model=model,
                    max_tokens=cfg.CONVO_MAX_TOKENS,
                    system=system,
                    tools=tools,
                    messages=cached(history),
                    **cfg.thinking_kwargs(model),
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
                        # Re-validate before falling back: delete_folder may
                        # have removed `suggested` from the registry during
                        # this very dialogue.
                        return (self.store._match_category(chosen)
                                or categories.valid_slug(suggested))
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
        return categories.valid_slug(suggested)  # it may have been deleted mid-dialogue
