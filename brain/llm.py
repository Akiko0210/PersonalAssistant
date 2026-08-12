"""Claude integration: conversation (with note-access tools) and summarisation.

DeepSeek models ride the same code path: their Anthropic-compatible endpoint
speaks the Messages API, so `client_for` just picks which anthropic-SDK client
a model id goes to. Nothing downstream — tool loop, history invariants, error
handling — knows more than one provider exists."""

import logging
import os
from datetime import datetime

import anthropic

from brain import agents
from stores import categories
import config as cfg
from brain import history as hist
from lib.atomic_io import write_json_atomic
from stores.discord_data import DiscordData
from stores.knowledge import KnowledgeStore
from brain.memory import ConversationMemory
from tools import ToolContext, api_tools, dispatch
from tools.focus_tools import focus_prompt_block

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
        # Personas: per-agent system prompt, tools, model, voice — and since
        # the memory split, each keeps its OWN history thread (self.history is
        # always the ACTIVE persona's; switch_to swaps files). The registry
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
                                convo_model=agents.registry_model(self.active),
                                active_agent=self.active)
        self._active_since = datetime.now()
        self._write_agent_state()
        # Conversation memory: restore the active persona's own thread (trimmed)
        # so it remembers its last conversation across restarts; saved back to
        # disk after every turn. The pre-isolation shared file is parked first.
        self._migrate_legacy_history()
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

    # --- the tool loop --------------------------------------------------------
    def _tool_loop(self, messages, *, ctx, tools, system_for, default_model,
                   max_rounds, on_tool=None, stop=None):
        """The one create -> tool_use -> tool_result engine behind converse(),
        run_delegated_task(), and choose_folder_via_dialogue(). It existed as
        three hand-written copies that drifted: only converse() had the
        NotFoundError revert and the max_tokens honesty (_final_text) — the
        others silently lacked both.

        messages       mutated in place (converse passes self.history).
        ctx            ToolContext for dispatch(); ctx.convo_model is read
                       fresh each round, so set_conversation_model mid-loop
                       applies to the next call — and is reverted to
                       default_model() if the API no longer serves it.
        system_for     (model_id) -> system str, rebuilt per round because the
                       identity block names the model.
        default_model  () -> model id when ctx.convo_model is unset, and the
                       NotFoundError fallback (re-raises if already on it).
        on_tool        optional (block) -> result str for a caller-private
                       tool (the folder dialogue's choose_folder); None means
                       "not mine", and the block goes through dispatch().
        stop           optional () -> bool, checked after each tool round, so
                       on_tool can end the loop through closure state.

        Returns the final response (stop_reason != "tool_use"), or None when
        the rounds ran out or `stop` fired. The idle cue is the caller's:
        converse holds it across the whole loop, the folder dialogue per
        exchange."""
        for _ in range(max_rounds):
            model = ctx.convo_model or default_model()
            try:
                resp = self.client_for(model).messages.create(
                    model=model,
                    max_tokens=cfg.CONVO_MAX_TOKENS,
                    system=system_for(model),
                    tools=tools,
                    messages=cached(messages),
                    **cfg.thinking_kwargs(model),
                )
            except anthropic.NotFoundError:
                # A switched-to model id the API no longer serves must not
                # brick every later turn — the voice fix (switching back)
                # itself needs a working model call. Revert and retry.
                fallback = default_model()
                if model != fallback:
                    log.warning("model %s rejected (not found); reverting "
                                "to the default %s", model, fallback)
                    ctx.convo_model = fallback
                    continue
                raise
            messages.append(
                {"role": "assistant",
                 "content": [self._dump_block(b) for b in resp.content]})
            if resp.stop_reason != "tool_use":
                return resp
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                out = on_tool(block) if on_tool else None
                if out is None:
                    log.info("tool_use %s %s", block.name, block.input)
                    out = dispatch(ctx, block.name, block.input)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id, "content": out})
            messages.append({"role": "user", "content": results})
            if stop is not None and stop():
                return None
        return None

    @staticmethod
    def _final_text(resp):
        """Text of a final response, with the max_tokens honesty applied. If
        the reply died inside a tool call, that call was never dispatched —
        the action did NOT happen, and the dangling tool_use will be sanitized
        away, so the model won't even remember trying. Say so out loud; a
        silent empty return here cost a whole note save while the model kept
        announcing "saving now" (2026-07-19 log)."""
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if resp.stop_reason == "max_tokens":
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

        Runs on the model the persona is parked on (model_for): "Bob is on
        DeepSeek" must mean Bob answers on DeepSeek everywhere, or identity
        answers lie — a delegated Alice on her registry default said "Haiku"
        while parked Alice was on DeepSeek Flash, seconds before the switch
        announcement contradicted her (session_2026-08-10.log 18:22; the
        user's call, 2026-08-10). The cost consequence is accepted and
        visible: the delegation log line names the model, and a hat parked on
        an expensive model bills it for background asks too. No idle
        "thinking" sound — the user may be mid-sentence with the foreground
        persona."""
        hat = agents.AGENTS[key]
        model = self.model_for(key)
        sub_ctx = ToolContext(store=self.store, discord=self.discord,
                              kb=self.kb, memory=self.memory,
                              client=self._ctx.client,
                              convo_model=model, active_agent=key,
                              # Focus is session state, not persona state: a
                              # delegated lookup honours the same narrowing
                              # the foreground conversation is under.
                              focus=self._ctx.focus)
        system = (DELEGATION_PROMPT.format(name=hat["name"], role=hat["role"])
                  + "\n\n" + hat["persona"] + focus_prompt_block(self._ctx.focus))
        # No switching tools in the background: the worker has no user to hand
        # over or re-route, and a nested delegation could chain unboundedly.
        # No order mutation either — review-before-submit means the USER hears
        # the review and says go, and a background worker is out of earshot by
        # definition; it could otherwise satisfy the review gate by talking to
        # itself. Read-only trading tools stay available.
        tools = api_tools(include=hat["tools"] - {
            "switch_agent", "ask_agent", "set_conversation_model",
            "submit_order", "cancel_order"})
        messages = [{"role": "user", "content": task}]
        log.info("delegated task -> %s (%s)", key, model)
        resp = self._tool_loop(
            messages, ctx=sub_ctx, tools=tools,
            system_for=lambda m: system, default_model=lambda: model,
            max_rounds=cfg.DELEGATION_MAX_TOOL_ROUNDS,
        )
        if resp is None:
            log.warning("delegated task hit DELEGATION_MAX_TOOL_ROUNDS (%d); "
                        "reporting as incomplete",
                        cfg.DELEGATION_MAX_TOOL_ROUNDS)
            return ("I ran out of steps before finishing that task — it may "
                    "be incomplete.", sub_ctx.pending_note)
        return (self._final_text(resp) or "the task is done.",
                sub_ctx.pending_note)

    # --- personas ("hats") ----------------------------------------------------
    def model_for(self, key):
        """API model id persona `key` answers with right now: the live setting
        when it's the active hat, its remembered mid-session choice when
        parked, else its registry default. The ONE resolver behind converse(),
        delegation, and the switch announcement — when these disagreed, a
        delegated Alice said "Haiku" seconds before the switch announcement
        said "DeepSeek V4 Flash" (session_2026-08-10.log 18:22)."""
        if key == self.active:
            return self._ctx.convo_model or agents.registry_model(key)
        return self._model_overrides.get(key) or agents.registry_model(key)

    @property
    def active_model(self):
        """API model id the active persona answers with. Same expression
        converse() resolves each call, exposed so callers (the switch
        announcement) can report the model without reaching into _ctx."""
        return self.model_for(self.active)

    @property
    def active_model_label(self):
        """Spoken name of the active model, e.g. "DeepSeek V4 Pro"."""
        return cfg.convo_model_label(self.active_model)

    def switch_to(self, key):
        """Make `key` the active persona: its system prompt, tool allowlist,
        model, AND history thread apply from the next converse() on. The
        departing persona's thread is saved first, then the target's own
        thread is loaded — nothing of the old conversation crosses over.
        Attribution is structural now (a thread has exactly one persona), so
        the old "(Tom took over...)" marker — once the ONLY record of who
        spoke what — has no job left and is gone. Context transfers between
        personas only as ask_agent summaries, by design: strict isolation is
        the feature, not a limitation."""
        if key == self.active:
            return
        # Save the departing thread BEFORE flipping self.active — after the
        # flip, _save_history would write it over the target's file.
        self._save_history()
        # Preserve a mid-session "switch to opus" for the hat it was made in.
        self._model_overrides[self.active] = self._ctx.convo_model
        self.active = key
        self._ctx.active_agent = key
        self._ctx.convo_model = (self._model_overrides.get(key)
                                 or agents.registry_model(key))
        self._active_since = datetime.now()
        self._write_agent_state()
        self.history = self._load_history()
        log.info("active agent -> %s (%s)", key, self._ctx.convo_model)

    def _write_agent_state(self):
        """Tell the dashboard who is talking and what each persona will answer
        with. Telemetry only — never read back, and never allowed to break a
        switch or a turn.

        The models have to be published because they live ONLY in this
        process: a mid-session set_conversation_model lands in
        _ctx.convo_model, and the per-hat memory of it in _model_overrides.
        The dashboard can otherwise see just the configured default (registry
        + agents.json), so its Agents page said Haiku while the conversation
        was going to DeepSeek. `since` tracks the last persona change, not
        this write, so a per-turn refresh doesn't keep resetting it."""
        try:
            live = {}
            for key in agents.AGENTS:
                chosen = (self._ctx.convo_model if key == self.active
                          else self._model_overrides.get(key))
                live[key] = chosen or agents.registry_model(key)
            write_json_atomic(cfg.AGENT_STATE_PATH, {
                "active": self.active,
                "name": agents.AGENTS[self.active]["name"],
                "since": self._active_since.isoformat(timespec="seconds"),
                "models": live,
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
                self.memory.record_dropped(dropped, self.active)
            except Exception as e:  # staging must never break the conversation
                log.warning("could not stage dropped history: %s", e)
        return kept

    @staticmethod
    def _migrate_legacy_history():
        """One-shot: park the pre-isolation shared history.json as .bak. Its
        turns are not staged anywhere — the session logs hold the same turns
        WITH speaker attribution, and scripts/seed_agent_memory.py mines those
        into each persona's own archive instead.

        Never overwrites an existing backup. Path.replace is an atomic
        rename that clobbers its destination silently, and this machine
        already had a hand-made history.json.bak from a month earlier —
        migrating over it would have destroyed the only copy."""
        if not cfg.HISTORY_PATH.exists():
            return
        target = cfg.HISTORY_PATH.with_suffix(".json.bak")
        n = 2
        while target.exists():
            target = cfg.HISTORY_PATH.with_suffix(f".json.bak{n}")
            n += 1
        try:
            cfg.HISTORY_PATH.replace(target)
            log.info("shared history.json parked as %s (threads are per-agent "
                     "now; seed_agent_memory.py mines the logs)", target.name)
        except OSError as e:
            log.warning("could not park legacy history.json: %s", e)

    def _load_history(self):
        h = hist.load(cfg.history_path(self.active))
        if h:
            h = self._trim_and_archive(h)
            log.info("restored %d message(s) of %s's thread", len(h), self.active)
        return h

    def _save_history(self):
        # Saved untrimmed: trimming happens on load / at each turn, where the
        # dropped part is staged into long-term memory. Trimming here instead
        # would silently discard the overflow on quit.
        hist.save(cfg.history_path(self.active), self.history)

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
            # The hat is stable for the whole turn (switch_agent defers to a
            # pending slot); the MODEL is not — set_conversation_model mid-loop
            # applies to the next round, which is why the system prompt is a
            # per-round closure: its identity block names the model.
            hat = agents.AGENTS[self.active]
            resp = self._tool_loop(
                self.history, ctx=self._ctx,
                tools=api_tools(include=hat["tools"]),
                system_for=lambda model: (
                    cfg.CONVO_SYSTEM_BASE + "\n\n" + hat["persona"]
                    + agents.roster_block(self.active)
                    + cfg.model_identity_block(cfg.convo_model_label(model))
                    # Focus lives in the cached prefix, so set/clear costs one
                    # prompt-cache miss — rare and user-initiated; the model
                    # KNOWING its retrieval is narrowed is worth more.
                    + focus_prompt_block(self._ctx.focus)),
                default_model=lambda: agents.registry_model(self.active),
                max_rounds=cfg.CONVO_MAX_TOOL_ROUNDS,
            )
            if resp is None:
                # The model kept calling tools without ever answering. Bail out
                # with an honest line rather than billing API calls forever; the
                # next turn's sanitize repairs whatever the loop left mid-flight.
                log.warning("tool loop hit CONVO_MAX_TOOL_ROUNDS (%d); bailing "
                            "out", cfg.CONVO_MAX_TOOL_ROUNDS)
                return ("I got stuck repeating tool calls and stopped myself. "
                        "Could you ask that again, maybe more specifically?")
            return self._final_text(resp)
        finally:
            self.idle.stop()
            # Fold in anything a synchronous tool recorded this turn, then save.
            # Deferred flows (the conversation-note save) flush themselves later
            # with persist=True, since they finish after this point.
            self.flush_tool_events()
            self._save_history()  # every turn — survives crashes and quits alike
            # set_conversation_model changes _ctx.convo_model without going
            # through switch_to, so the state file would otherwise go stale the
            # moment the user says "switch to DeepSeek". One atomic write a
            # turn, next to the history save that already happens here.
            self._write_agent_state()

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
        chosen = None

        def on_tool(block):
            """Intercept choose_folder — it's this dialogue's private tool,
            not a registry one — and end the loop through `chosen`."""
            nonlocal chosen
            if block.name != "choose_folder":
                return None
            chosen = block.input.get("folder", "")
            return "ok"

        try:
            # max_turns bounds the spoken exchanges; each may spend tool
            # rounds up to the loop's usual cap.
            for _ in range(max_turns):
                self.idle.start()  # thinking; stopped below before we speak/listen
                resp = self._tool_loop(
                    history, ctx=self._ctx, tools=tools,
                    system_for=lambda m: system,
                    default_model=lambda: cfg.CONVO_MODEL,
                    max_rounds=cfg.CONVO_MAX_TOOL_ROUNDS,
                    on_tool=on_tool, stop=lambda: chosen is not None,
                )
                if chosen is not None:
                    if chosen in categories.NOTE_CATEGORIES:
                        return chosen
                    # Re-validate before falling back: delete_folder may have
                    # removed `suggested` from the registry during this very
                    # dialogue.
                    return (self.store._match_category(chosen)
                            or categories.valid_slug(suggested))
                if resp is None:
                    break  # stuck in tool calls — fall back to the suggestion

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
