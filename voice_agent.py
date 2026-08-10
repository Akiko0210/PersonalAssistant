"""Voice notetaking agent — entry point and orchestration.

Conversation mode is the default: listen for an utterance, answer with Claude,
speak the reply. Hotkeys switch into a silent notetaking mode and back, and
toggle a global mute (which stops the microphone, not a reply in progress).
See README.md for setup and the hotkey list.
"""

import argparse
import collections
import logging
import queue
import re
import sys
import threading
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()  # load ANTHROPIC_API_KEY (and any overrides) from a .env file
except ImportError:
    pass  # python-dotenv not installed yet; env vars still work

import anthropic

import agents
import categories
import config as cfg
import dashboard
from audio import AudioEngine
from barge_in import BargeInDetector
from gestures import ClickGestureDecoder
from stt import Transcriber
from tts import Announcer, Speaker
from notes import NoteStore
from knowledge import KnowledgeStore
from llm import Claude
from sound import IdleSound
from single_instance import SingleInstance, AlreadyRunning


def explain_error(e: Exception) -> str:
    """Turn a turn-level failure into a spoken sentence naming the real cause.

    A generic "let's try that again" once sent the user chasing a phantom bug
    for a whole session while the actual problem was an empty credit balance —
    so name the cause whenever the exception identifies one, and say whether
    retrying can help.
    """
    # Dispatch on the SDK's typed exceptions, not message substrings: the turn
    # wraps audio capture, STT, TTS, and the note stores too, and a local fault
    # whose message happens to contain "timeout" or "connection" (sounddevice,
    # CUDA, SQLite) must never be announced as an Anthropic network problem —
    # that's the same misdirection this function exists to prevent.
    if not isinstance(e, anthropic.APIError):
        return "Sorry, I hit an error. Let's try that again."
    if isinstance(e, anthropic.APIConnectionError):  # includes APITimeoutError
        return ("I couldn't reach the Anthropic API — this looks like a "
                "network problem. Check the internet connection.")
    if "credit balance is too low" in str(e).lower():
        return ("My Anthropic credit balance is too low — please add API "
                "credits. Retrying won't help until you do.")
    if isinstance(e, anthropic.AuthenticationError):
        return ("My API key was rejected — it may be missing, expired, or "
                "revoked. Retrying won't help until the key is fixed.")
    if isinstance(e, anthropic.RateLimitError):
        return ("I'm being rate limited by the API. Give it a minute, "
                "then try again.")
    if isinstance(e, anthropic.NotFoundError):
        return ("The model I tried to use wasn't found by the API. If this "
                "keeps happening, check the model ids in config.py.")
    status = getattr(e, "status_code", None)
    if status == 529:
        return ("The Anthropic API is overloaded right now. This usually "
                "clears quickly — try again shortly.")
    if status is not None and status >= 500:
        return ("The Anthropic API returned a server error. That's on their "
                "end — try again shortly.")
    return "Sorry, I hit an error. Let's try that again."


def is_backchannel(text: str) -> bool:
    """True when a transcript is nothing but listener filler ("yeah", "uh-huh",
    "oh okay") — an acknowledgement, not a request to stop talking. Used after
    a barge-in fires to decide between resuming the reply and yielding the
    floor. Empty text is NOT a backchannel (there was nothing to classify)."""
    words = re.findall(r"[a-z']+", text.lower())
    return (0 < len(words) <= cfg.BACKCHANNEL_MAX_WORDS
            and all(w in cfg.BACKCHANNEL_WORDS for w in words))


def setup_logging():
    cfg.ensure_dirs()
    logfile = cfg.LOG_DIR / f"session_{datetime.now():%Y-%m-%d}.log"
    handlers = [logging.FileHandler(logfile, encoding="utf-8"),
                logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-8s %(levelname)-7s %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("agent")


class Agent:
    def __init__(self):
        self.log = logging.getLogger("agent")
        t0 = time.monotonic()
        self.audio = AudioEngine()
        self.tts = Speaker()
        # Second voice, for notices that must be heard over a playing reply.
        self.announcer = Announcer()
        # Which voice the reply is in, read on the main thread and cached so the
        # button thread can pick a contrasting one without touching COM.
        self._speaking_voice = self.tts.current_voice()
        self.store = NoteStore()
        # Absorb any newly added trading PDFs before we start listening. This is an
        # idempotent scan: unchanged files are skipped by content hash without
        # loading the embedding model, so it's near-instant unless there's a genuinely
        # new book (which is embedded once, here, blocking startup that one time).
        # Video/audio is excluded: a lecture takes ~20 min per hour to transcribe,
        # which is too long to hold startup open from in here. run.bat does that
        # pass as a separate `--ingest` process first, where the console can show
        # progress; this scan just names anything still waiting.
        self.kb = KnowledgeStore()
        self.log.info("scanning knowledge base...")
        self.log.info(self.kb.ingest_folder(include_media=False))
        self.idle = IdleSound()  # "thinking" cue, looped during model calls
        self.llm = Claude(self.store, self.idle, self.kb)
        # Fold any conversation text that aged out of the rolling window into
        # long-term memory. No-op on most boots; one quick model call otherwise.
        archived = self.llm.consolidate_memory()
        if archived:
            self.log.info(archived)
        self.log.info("loading speech model...")
        self.stt = Transcriber()
        self.log.info("startup took %.1fs", time.monotonic() - t0)
        self.status = "conversation_mode"

        self.cmds: "queue.Queue[str]" = queue.Queue()
        self.interrupt = threading.Event()
        # Set by a raw button click to silence playback IMMEDIATELY — before the
        # multi-click window decides which command it was. The Yealink dongle
        # only transmits a press reliably when the host actually pauses on the
        # previous one (see MEDIA_CONTROL.md), so the 2nd/3rd clicks of a
        # gesture would be swallowed if speech played on through the window.
        self.hush = threading.Event()
        # Set once the gesture behind that click turns out to be one that must
        # stop the reply for good (note-taking, quit). Mute is deliberately not
        # one of them: it deafens the microphone, it doesn't gag the agent.
        self.silence = threading.Event()
        # ...and set when the gesture was a non-silencing one, releasing the
        # paused reply to finish.
        self.resume_speech = threading.Event()
        # Raw button presses (from either listener channel) are decoded into
        # single/double/triple gestures here; see gestures.py.
        self._gesture = ClickGestureDecoder(
            on_gesture=self._on_media_gesture, on_press=self._on_media_press
        )
        self.running = True
        self._interrupted_reply = None   # full text of the last interrupted reply
        self._interrupted_remaining = None  # unsaid portion after barge-in
        # Background delegation (ask_agent): finished tasks queue their spoken
        # result here; it is delivered only at utterance boundaries, in the
        # working persona's own voice (see _deliver_interjections). The event
        # is a gentle wake-up for the idle listener — collect_utterance honours
        # it only between utterances, so a result never cuts the user off.
        self.interjections: "queue.Queue[dict]" = queue.Queue()
        self.interject = threading.Event()
        self._delegation_threads = []
        # Messages typed into the dashboard (queue_typed_message). Drained one
        # per turn at utterance boundaries, same as interjections.
        self.typed: "queue.Queue[str]" = queue.Queue()

    # --- command plumbing ----------------------------------------------------
    def _push(self, cmd):
        self.cmds.put(cmd)
        self.interrupt.set()

    def _drain(self):
        """Process queued hotkey commands. Returns the set of high-level signals
        ('start_note', 'stop_note', 'quit'); the mute acknowledgement is spoken
        inline. Mute itself was already applied when the gesture resolved (see
        _on_media_gesture) — this only voices it."""
        signals = set()
        announce = False
        while not self.cmds.empty():
            cmd = self.cmds.get()
            if cmd == "announce_mute":
                # Collapsed, not spoken per command: toggling twice while a
                # reply plays would otherwise announce the same final state
                # twice.
                announce = True
            else:
                signals.add(cmd)
        if announce:
            # Fallback path only: with a second voice available the notice was
            # already spoken, over the reply, when the gesture resolved.
            self.say("Muted." if self.audio.muted.is_set() else "Listening.",
                     voice=False, commands=False)
        self.interrupt.clear()
        self.silence.clear()
        return signals

    def set_mute(self, muted: bool):
        """Put the microphone into `muted` and say so — the shared body of
        every mute, whatever asked for it: the headset click (via _toggle_mute)
        or the dashboard's button, arriving on a web handler thread.

        A *state*, not a toggle, because the dashboard names the state it
        wants and two sources flipping a shared toggle would race. A request
        for the state we're already in is applied silently: there is nothing
        to announce, and re-announcing would talk over the reply for no
        reason. A dashboard mute skips the headset's click dance entirely —
        there is no dongle here waiting to see a pause honoured, so nothing
        touches playback; a playing reply carries on."""
        changed = muted != self.audio.muted.is_set()
        if muted:
            self.audio.muted.set()
        else:
            self.audio.muted.clear()
        if not changed:
            return
        self.log.info("muted — not listening" if muted else "unmuted — listening")
        # The acknowledgement goes out on the *second* voice, so it lands with
        # the button press instead of queueing behind a reply that may have
        # twenty seconds left to run. Only if that voice is unavailable does it
        # fall back to the main one, spoken at the next drain.
        notice = "Muted." if muted else "Listening."
        if not self.announcer.announce(notice, avoid_voice=self._speaking_voice):
            self._push("announce_mute")

    def _toggle_mute(self):
        """Flip the microphone, from the button thread, the instant the gesture
        resolves — not when the main loop next drains. The reply is still
        playing (mute no longer cuts it off), and the microphone must already
        be deaf while it plays: otherwise the tail of the reply stays
        interruptible by the very person who just asked not to be heard."""
        self.set_mute(not self.audio.muted.is_set())

    def queue_typed_message(self, text: str):
        """A message typed into the dashboard, arriving on a web handler
        thread. Queued, not answered here: the turn loop picks it up at the
        next utterance boundary, so it never cuts off speech in progress. The
        wake event is honoured while muted — typing is exactly what a muted
        user does — and honoured only between utterances, so it cannot
        truncate one."""
        self.typed.put(text)
        self.interject.set()

    def _handle_typed_message(self, text: str):
        """Answer one typed message as a spoken turn. Same path as a spoken
        utterance except for the mic-specific parts: no continuation settle
        (that window listens to the microphone and would merge room noise into
        typed text — so llm.converse directly, the forwarded-turn precedent in
        _after_reply), and no backchannel/resume handling (fillers are things
        the mic hears). Persona addressing still applies, so "Tom, ..." typed
        works like "Tom, ..." spoken."""
        self.log.info("you (typed): %s", text)
        target, remainder = agents.match_address(text)
        if target and target != self.llm.active:
            self._switch_agent(target)
            if not remainder:
                return
            text = remainder
        reply = self.llm.converse(text)
        if not reply:
            if not self.silence.is_set():
                self.log.warning("model returned an empty reply for: %r", text)
                self.say("I came back with an empty reply — something went "
                         "wrong. Try that again.", voice=False, commands=False)
            return
        self.log.info("agent: %s", reply)
        interrupted = self.say(reply, save_resume=True)
        self._after_reply(interrupted)

    def _toggle_note(self):
        self.silence.set()  # a new note (or the end of one) stops the reply
        if self.status == "conversation_mode":
            self.status = "note_taking"
            if self.audio.muted.is_set():
                self.audio.muted.clear()
                self.log.info("auto-unmuted for notetaking")
            self._push("start_note")
        else:
            self.status = "conversation_mode"
            self._push("stop_note")

    def _quit(self):
        self.silence.set()
        self._push("quit")

    def _on_media_press(self):
        """Every accepted click: go quiet the moment the press lands — stop the
        thinking cue here, and flag say() to *pause* speech on its next poll
        (~100 ms; hush avoids touching the SAPI COM object cross-thread). This
        keeps a state-tracking dongle (Yealink) in sync: it swallows the next
        press whenever the host keeps playing through a "pause", which is how
        the 2nd/3rd clicks of a gesture were getting eaten (MEDIA_CONTROL.md).

        Pausing rather than purging is what lets a mute click leave the reply
        intact — at this point the gesture hasn't resolved, so we don't yet
        know whether the reply should die (note-taking, quit) or pick up where
        it stopped (mute). _hold_for_gesture waits for that verdict."""
        self.idle.stop()
        self.hush.set()
        # Obediently pause the silent keepalive stream too, so the dongle sees
        # its "pause" honoured no matter what was (or wasn't) playing.
        media = getattr(self, "_media", None)
        if media is not None:
            media.duck()

    def _on_media_gesture(self, count):
        if count == 1:
            self.log.info("media hotkey: single click -> toggle mute")
            self._toggle_mute()
            self.resume_speech.set()  # mute stops listening, not speaking
        elif count == 2:
            self.log.info("media hotkey: double click -> toggle note-taking")
            self._toggle_note()
        else:
            self.log.info("media hotkey: triple click -> quit")
            self._quit()

    def start_hotkeys(self):
        # The button is listened to on BOTH channels at once, because different
        # headsets deliver presses differently (see MEDIA_CONTROL.md):
        #   - keyboard hook: wired headsets and USB wireless dongles (media-key
        #     events);
        #   - SMTC media session: Bluetooth-native headsets (AVRCP — their
        #     presses never appear as key events).
        # A press that arrives on both channels within MEDIA_CLICK_DEDUPE_S is
        # counted once by _media_click.
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.media_play_pause:
                self.log.info("media key received (speaking=%s)", self.tts.is_busy())
                self._gesture.click()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

        try:
            from media_control import MediaButtonListener

            def on_play_pause():
                self.log.info("media button (SMTC) received (speaking=%s)",
                              self.tts.is_busy())
                self._gesture.click()

            self._media = MediaButtonListener(
                on_play_pause=on_play_pause,
                # Headsets that decode multi-press in firmware (e.g. AirPods)
                # deliver double/triple as Next/Previous — map them to the same
                # actions as counted double/triple clicks.
                on_next=self._toggle_note,
                on_previous=self._quit,
                # Short debounce: real double-clicks arrive ~200 ms apart and
                # must get through; cross-channel dedupe lives in _media_click.
                debounce_s=0.08,
                keepalive=cfg.MEDIA_KEEPALIVE,
            )
            self._media.start()
        except Exception as e:  # noqa: BLE001 - any winrt/SMTC failure
            self.log.warning(
                "SMTC media session unavailable (%s); Bluetooth-native headset "
                "buttons won't be received (keyboard hook still active)", e
            )

    # --- modes ---------------------------------------------------------------
    def run_conversation_turn(self):
        # Background results that arrived while we were away (or that just
        # ended the idle wait below) go out first — between utterances is
        # exactly where an interjection belongs.
        self._deliver_interjections()
        utt = self.audio.collect_utterance(
            interrupt=self.interrupt, endpoint_ms=cfg.CONVO_ENDPOINT_MS,
            wake=self.interject,
        )
        signals = self._drain()
        if "quit" in signals:
            self.running = False
            return
        if "start_note" in signals:
            self.run_notetaking()
            return
        if utt is None or utt.size == 0:
            # No speech — but perhaps a typed message ended the idle wait (its
            # arrival sets self.interject, the wake event). One per turn, so a
            # burst of messages interleaves with the mic instead of locking
            # the floor.
            try:
                typed = self.typed.get_nowait()
            except queue.Empty:
                return
            self._handle_typed_message(typed)
            return
        text = self.stt.transcribe(utt)
        if not text:
            return
        self.log.info("you: %s", text)

        # Resume the interrupted reply — either the user asked to ("continue"),
        # or the "interruption" was just a listener filler ("yeah", "uh-huh"):
        # an acknowledgement means keep talking, not stop. Neither costs a
        # model call.
        if self._interrupted_remaining:
            asked = text.strip().lower() in (
                "continue", "go on", "keep going", "go ahead",
            )
            if asked or is_backchannel(text):
                self.log.info("resuming interrupted reply"
                              if asked else
                              "(just an acknowledgement — carrying on)")
                interrupted = self.say(self._interrupted_remaining, save_resume=True)
                if not interrupted:
                    self._interrupted_reply = None
                    self._interrupted_remaining = None
                    self.audio.flush()
                return

        self._interrupted_reply = None
        self._interrupted_remaining = None

        # Spoken agent addressing: "Bob, what's my last note?" or "switch to
        # Tom". Detected on the transcript directly — zero model calls; any
        # free-form phrasing the regex misses is covered by the switch_agent
        # tool the active persona can call.
        target, remainder = agents.match_address(text)
        if target and target != self.llm.active:
            self._switch_agent(target)
            if not remainder:
                return  # a pure switch — back to listening in the new voice
            text = remainder

        reply = self._converse_with_followups(text)
        if not reply:
            # Expected only when note-taking or quit cut the turn short. An
            # empty reply with nothing silencing it means the model produced
            # nothing — that must never pass silently again (a truncated tool
            # call once died here unnoticed, and the user heard nothing for 12
            # minutes).
            if not self.silence.is_set():
                self.log.warning("model returned an empty reply for: %r", text)
                self.say("I came back with an empty reply — something went "
                         "wrong. Try that again.", voice=False, commands=False)
            return
        self.log.info("agent: %s", reply)
        interrupted = self.say(reply, save_resume=True)
        self._after_reply(interrupted)

    def _after_reply(self, interrupted: bool):
        """Everything a turn leaves behind once its reply is spoken: background
        delegations to start, a prepared note to file, a persona switch to
        perform, and queued background results to deliver. One method so every
        deferred slot is drained in one place, in a fixed order — a note save
        that ran inside the switch's forwarded turn once escaped this
        accounting and its folder question fired a turn later, eating the
        user's next command (session_2026-07-31.log, 11:31)."""
        # Delegations first: they run in the background, so starting them
        # before the (possibly slow) note dialogue costs nothing and lets the
        # work overlap it.
        for key, task in self.llm.take_pending_delegations():
            self._start_delegation(key, task)

        # The user asked to save something from the conversation as a note: the
        # model prepared it via save_conversation_note; now run the same folder
        # dialogue + save flow a finished note-taking session gets.
        pending = self.llm.take_pending_note()
        if pending:
            self._save_pending_note(pending)
            dropped = self.llm.take_pending_switch()
            if dropped:
                # Both in one turn is a model overreach; the note save owns the
                # turn, and a stale switch must not fire minutes later. Say so
                # — a silent drop once left the user talking to the wrong
                # persona for 16 minutes without knowing it
                # (session_2026-07-31.log, 11:32).
                name = agents.AGENTS[dropped[0]]["name"]
                self.log.info("dropped pending agent switch "
                              "(note save took the turn)")
                self.say(f"By the way, I didn't switch you to {name} — "
                         f"ask again if you still want {name}.",
                         voice=False, commands=False)
            return

        # The model handed the conversation to another persona (switch_agent):
        # the goodbye above came out in the old voice; switch now and, if the
        # user's question was forwarded, let the new persona answer it.
        switch = self.llm.take_pending_switch()
        if switch:
            key, forward = switch
            self._switch_agent(key)
            if forward:
                forwarded = self.llm.converse(forward)
                if forwarded:
                    self.log.info("agent: %s", forwarded)
                    interrupted = self.say(forwarded, save_resume=True)
            # The forwarded turn ran INSIDE this one — past the pending-note
            # check above. Drain whatever it queued here, where it happened,
            # instead of letting it leak into the next turn.
            pending = self.llm.take_pending_note()
            if pending:
                self._save_pending_note(pending, saver=agents.AGENTS[key]["name"])
            for dkey, dtask in self.llm.take_pending_delegations():
                self._start_delegation(dkey, dtask)
            if self.llm.take_pending_switch():
                # A switch queued by the forwarded turn itself. One hand-off
                # per user turn: honouring a chain here could ping-pong
                # personas without the user ever speaking. Dropping silently
                # is safe — the user is exactly where the announced switch
                # put them.
                self.log.info("dropped chained agent switch "
                              "(one hand-off per turn)")

        if not interrupted:
            self.audio.flush()
            # A background result that finished while we were talking goes out
            # now — right after the reply, before the user takes the floor.
            # (After a barge-in the user is already talking, so it waits for
            # the next between-turns gap instead.)
            self._deliver_interjections()

    def _switch_agent(self, key):
        """Flip the active persona everywhere it shows: model/tools/prompt
        (llm.switch_to), the speaking voice, and a short spoken announcement —
        the announcement is the load-bearing signal on machines where two
        personas share a SAPI voice.

        The announcement names the model too. Each persona remembers its own
        mid-session model choice (Claude.switch_to), so switching back to a hat
        left on an expensive or external model is otherwise silent — and the
        conversation model is exactly the thing you can't hear."""
        self.llm.switch_to(key)
        hat = agents.AGENTS[key]
        self.tts.set_voice(hat["tts_voice"], hat["tts_rate"])
        self._speaking_voice = self.tts.current_voice()
        model = self.llm.active_model_label
        self.log.info("=== talking to %s (%s) ===", hat["name"], model)
        self.say(f"{hat['name']} here, running on {model}.",
                 voice=False, commands=False)

    # --- background delegation (ask_agent) ------------------------------------
    def _start_delegation(self, key, task):
        """Run a task as another persona on a worker thread (see
        llm.run_delegated_task); the user keeps talking to the active persona
        meanwhile. The result — or an honest failure — is queued as an
        interjection and spoken at the next utterance boundary; the worker
        pokes self.interject so an idle wait ends promptly."""
        hat = agents.AGENTS[key]
        self.log.info("delegating to %s: %.120s", hat["name"], task)

        def work():
            try:
                text, note = self.llm.run_delegated_task(key, task)
                self.interjections.put({"agent": key, "text": text,
                                        "note": note})
            except Exception as e:  # noqa: BLE001 - a lost task must be reported, not raised
                self.log.exception("delegated task for %s failed", hat["name"])
                self.interjections.put({
                    "agent": key, "note": None,
                    "text": ("I couldn't finish the task that was sent to "
                             "me. " + explain_error(e)),
                })
            finally:
                self.interject.set()

        t = threading.Thread(target=work, daemon=True, name=f"delegate-{key}")
        self._delegation_threads.append(t)
        t.start()

    def _deliver_interjections(self):
        """Speak queued background-task results. Called only at utterance
        boundaries — after a reply finishes, or when a finished worker ends
        the idle wait — and it stands down whenever the user already holds
        the floor: buffered speech or a muted microphone leaves the queue
        untouched for the next boundary. A voice barge-in mid-delivery does
        the same: the user took the floor, the rest of the queue waits."""
        self.interject.clear()
        if self.audio.muted.is_set() or self.audio.has_buffered_speech():
            return
        while True:
            try:
                item = self.interjections.get_nowait()
            except queue.Empty:
                return
            if self._speak_interjection(item):
                return  # barged in — remaining items wait for the next gap

    def _speak_interjection(self, item) -> bool:
        """One background result. Returns True when the user's voice cut it
        short. The two item shapes get deliberately different treatment:

        - A prepared note (`note` set): the worker persona takes the floor —
          its voice, a one-line announcement, then the folder dialogue, which
          is a real exchange with that persona. NOT voice-interruptible: the
          announcement is a single sentence, and a barge-in would strand the
          prepared note invisibly.

        - A plain text result: the delegate hands the report back and the
          ACTIVE persona reads it, in its own voice, as a normal reply —
          barge-in and "continue" both work. This used to run in the worker's
          voice with barge-in disabled, on the theory that the report is
          short and delivery is its only chance to reach the user; a note
          read aloud broke both halves of that — minutes of unstoppable
          speech, with everything the user said meanwhile piling up as
          buffered input for the next turn. Now the result is folded into the
          shared history BEFORE speaking, so an interruption can never strand
          it: the conversation already knows, even if the user never hears
          the end."""
        hat = agents.AGENTS[item["agent"]]
        note = item.get("note")
        if note:
            active = agents.AGENTS[self.llm.active]
            self.tts.set_voice(hat["tts_voice"], hat["tts_rate"])
            self._speaking_voice = self.tts.current_voice()
            try:
                self.say(f"{hat['name']} here — your note is ready to file.",
                         voice=False, commands=False)
                self._save_pending_note(note, saver=hat["name"])
            finally:
                self.tts.set_voice(active["tts_voice"], active["tts_rate"])
                self._speaking_voice = self.tts.current_voice()
            return False
        self.llm.record_tool_event(
            f"{hat['name']} finished a background task (ask_agent) and "
            f"reported: {item['text']}")
        self.llm.flush_tool_events(persist=True)
        return self.say(f"{hat['name']} came back — {item['text']}",
                        save_resume=True)

    def _converse_with_followups(self, text: str) -> str:
        """Collect the *whole* turn before calling the model, so a pause
        mid-sentence never swallows words — and never costs a wasted API call.

        After the utterance endpoints, keep listening for a short settle window
        (CONTINUATION_SETTLE_MS). If the user resumes talking, capture that
        continuation, merge it, and settle again. Only once they've truly
        finished — the window elapses in silence — do we call converse() once,
        with the complete utterance.

        (The previous design fired a speculative converse() the instant the
        utterance ended and threw the reply away whenever the user kept talking;
        each mid-thought pause billed a full model call — see PROJECT.md §4.)"""
        rounds = 0
        while self._await_continuation(cfg.CONTINUATION_SETTLE_MS):
            if self.silence.is_set():
                # Note-taking or quit arrived — abandon the turn without calling
                # the model, but keep the words: the user said them, and they
                # must not vanish. Muting is NOT one of these: it stops the
                # microphone, so the question already asked still gets answered
                # (and spoken) as normal.
                self.llm.record_unanswered(text)
                return ""
            rounds += 1
            if rounds > cfg.MAX_CONTINUATION_ROUNDS:
                # Something keeps re-triggering the settle window (a TV, a
                # second voice). Answer what we have instead of holding the
                # turn hostage and merging noise into the question; whatever
                # keeps talking is handled by the normal turn machinery.
                self.log.info("continuation cap reached — answering now")
                break
            # The user kept talking. Their opening frames were pushed back, so
            # collect the rest of this chunk, merge it, and settle again.
            more = self.audio.collect_utterance(
                interrupt=self.interrupt, endpoint_ms=cfg.CONVO_ENDPOINT_MS
            )
            if self.silence.is_set():
                self.llm.record_unanswered(text)
                return ""
            if more is not None and more.size > 0:
                addition = self.stt.transcribe(more)
                if addition:
                    self.log.info("you (continued): %s", addition)
                    text = f"{text} {addition}"
            # else: a cough / room noise triggered the settle — nothing to add,
            # and (unlike before) nothing was spent. Loop and keep settling.
        if self.silence.is_set():
            self.llm.record_unanswered(text)
            return ""
        return self.llm.converse(text)

    def _await_continuation(self, settle_ms) -> bool:
        """After an utterance ends, listen for up to `settle_ms` in case the user
        was only pausing mid-thought. Returns True the moment they audibly resume
        — the triggering frames are pushed back so collect_utterance captures the
        words from the very beginning — or False if the window elapses quietly
        (or a hotkey command arrives). If a speech onset is mid-flight at the
        deadline (voiced frames consumed but not yet a trigger), the window is
        extended once (CONTINUATION_GRACE_MS) so the onset can become a real
        trigger instead of having its opening frames dropped at the boundary;
        everything else consumed here is silence or sub-threshold noise."""
        pad_frames = max(1, cfg.SPEECH_PAD_MS // cfg.FRAME_MS)
        ring = collections.deque(maxlen=pad_frames)
        deadline = time.monotonic() + settle_ms / 1000
        graced = False
        while True:
            if time.monotonic() >= deadline:
                if not graced and any(q for _, q in ring):
                    graced = True
                    deadline = time.monotonic() + cfg.CONTINUATION_GRACE_MS / 1000
                else:
                    return False
            if self.interrupt.is_set():
                return False
            res = self.audio.poll_speech(timeout=0.05, return_frame=True)
            if res is None:
                continue
            is_speech, rms, frame = res
            # The energy floor keeps room noise (and any lingering echo) from
            # counting as the user talking.
            qualifies = is_speech and rms >= cfg.BARGE_IN_ENERGY
            ring.append((frame, qualifies))
            voiced = sum(1 for _, q in ring if q)
            if voiced > cfg.TRIGGER_RATIO * ring.maxlen:
                self.audio.pushback(f for f, _ in ring)
                self.log.info("(you kept talking — waiting for the rest)")
                return True

    def say(self, text: str, *, voice: bool = True, commands: bool = True,
            save_resume: bool = False) -> bool:
        """Single entry point for everything the agent speaks aloud.

        While speaking, watch for interruptions so the user is never stuck waiting
        for the agent to finish. Returns True only when interrupted by the user's
        *voice* — in that case the captured audio is left buffered for the next
        collect_utterance to pick up. Returns False when the speech finishes
        normally or is stopped by an action command (the command stays queued for
        the main loop to drain).

        voice:    stop when the user starts talking (voice barge-in).
        commands: stop when a *silencing* action command (note-taking, quit)
                  arrives. Mute is not one — it deafens the microphone and
                  leaves the reply to finish (see _hold_for_gesture). Turned off
                  only for the folder-destination question, so those commands
                  don't disrupt that exchange — voice barge-in still works there.
        save_resume: remember the unsaid tail for the "continue" command.

        Falls back to plain blocking speech for short status acks (voice=False) or
        when the TTS backend can't speak asynchronously."""
        self.idle.stop()  # never let the thinking cue overlap spoken output
        if not (voice and cfg.BARGE_IN and self.tts.supports_async):
            self.tts.speak(text)
            return False
        # Audio buffered while the model was thinking may be the user talking —
        # they resumed after the settle window closed, mid-model-call. A blind
        # flush() here silently deleted those words. Scan the buffer instead:
        # a speech onset means the user is already talking, so hold the reply
        # and hand the frames back for capture — exactly a barge-in at t=0.
        # (This also keeps their live speech out of the barge-in detector's
        # echo calibration, which would otherwise lock the threshold above
        # their own voice and make the reply uninterruptible.) Pure silence or
        # noise is discarded, as flush() always did.
        onset = self._drain_buffered_speech()
        if onset is not None:
            self.audio.pushback(onset)
            if save_resume:
                self._save_interrupted(text, 0.0)
            self.log.info("(you were already talking — holding the reply)")
            return True
        # Only clicks during *this* utterance may act on it, and a verdict left
        # over from an earlier one must not release the first pause of this one.
        self.hush.clear()
        self.resume_speech.clear()
        start = time.monotonic()
        self.tts.begin(text)
        detector = BargeInDetector()

        while self.tts.is_busy():
            # A raw click just landed: pause playback NOW, before the
            # multi-click window even resolves into a command — the dongle
            # swallows the gesture's next click if playback runs on (see
            # MEDIA_CONTROL.md). Whether the reply then dies or carries on
            # depends on which gesture it turns out to be.
            if commands and self.hush.is_set():
                self.hush.clear()
                if not self._hold_for_gesture():
                    return False
                continue
            if commands and self.silence.is_set():
                # A silencing command with no click behind it — an AirPods
                # Next/Previous, already decoded in firmware.
                self.tts.stop()
                return False
            res = self.audio.poll_speech(timeout=0.1, return_frame=True)
            if res is None:
                continue
            if detector.feed(*res):
                self.tts.stop()
                self.audio.pushback(detector.run)  # give the words back to capture
                if save_resume:
                    self._save_interrupted(text, time.monotonic() - start)
                self.log.info("(interrupted — listening)")
                return True

        detector.log_summary()  # finished uninterrupted; surface tuning numbers
        return False

    def _hold_for_gesture(self) -> bool:
        """A click landed mid-reply. Pause playback at once — the dongle only
        transmits the gesture's next click if the host really stops — then wait
        for the multi-click window to say what the gesture was:

          - mute (single click): the microphone goes deaf, the reply resumes
            and finishes. Muting means "stop listening to me", not "stop
            talking", and cutting the sentence off was the old behaviour this
            replaces;
          - note-taking / quit (double / triple): the reply ends here.

        Returns True if speech resumed. A pause rather than a purge is the
        whole trick — SAPI can only continue an utterance it hasn't discarded
        — so a backend that can't pause falls back to the old all-or-nothing
        stop."""
        if not self.tts.pause():
            self.tts.stop()
            return False
        deadline = time.monotonic() + cfg.GESTURE_VERDICT_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.silence.is_set():
                self.tts.stop()
                self.log.info("(gesture ends the reply)")
                return False
            if self.resume_speech.is_set():
                self.resume_speech.clear()
                self.tts.resume()
                self.log.info("(muted — the reply carries on)"
                              if self.audio.muted.is_set() else
                              "(unmuted — the reply carries on)")
                return True
            time.sleep(0.02)
        # No gesture resolved behind that press (a stray or swallowed click):
        # never leave a reply stuck mid-word.
        self.tts.resume()
        self.log.info("(no gesture followed the click — resuming the reply)")
        return True

    def _drain_buffered_speech(self):
        """Everything recorded while the model was thinking is sitting in the
        mic buffer, each frame already carrying the VAD's speech/silence
        verdict. Read the buffer back and ask one question: did the user
        speak? If yes, return the frames from where their speech starts (the
        caller pushes them back so no words are lost). If it was only
        silence/noise, discard it — exactly what flush() did — and return
        None so the reply plays.

        Reads are non-blocking (timeout=0): we take only what is already
        recorded. The always-on stream adds a frame every FRAME_MS and a
        non-blocking read is instant, so this loop always finishes — unlike
        a timed poll, which loses the race against the frame cadence and
        consumes the live stream forever (session_2026-07-17.log, where
        every reply went silent). Speech starting at this exact moment is
        the barge-in detector's job, not ours."""
        pad_frames = max(1, cfg.SPEECH_PAD_MS // cfg.FRAME_MS)
        ring = collections.deque(maxlen=pad_frames)
        while True:
            res = self.audio.poll_speech(timeout=0, return_frame=True)
            if res is None:
                return None  # buffer fully read — the VAD saw no speech in it
            is_speech, rms, frame = res
            qualifies = is_speech and rms >= cfg.BARGE_IN_ENERGY
            ring.append((frame, qualifies))
            voiced = sum(1 for _, q in ring if q)
            if voiced > cfg.TRIGGER_RATIO * ring.maxlen:
                return [f for f, _ in ring]

    def _save_interrupted(self, full_text: str, elapsed_s: float):
        words = full_text.split()
        words_spoken = int(elapsed_s * cfg.TTS_RATE / 60)
        remaining_words = words[max(0, words_spoken - 2):]  # overlap a couple for context
        if remaining_words:
            self._interrupted_reply = full_text
            self._interrupted_remaining = " ".join(remaining_words)
        else:
            self._interrupted_reply = None
            self._interrupted_remaining = None

    def run_notetaking(self):
        note_id = self.store.new_session()
        self.log.info("=== notetaking started (%s) — recording silently ===", note_id)
        self.say("Note taking started.", voice=False, commands=False)

        self.audio.flush()
        stopped = False
        while not stopped:
            utt = self.audio.collect_utterance(
                interrupt=self.interrupt, endpoint_ms=cfg.NOTE_ENDPOINT_MS
            )
            signals = self._drain()
            if "quit" in signals:
                self.running = False
                stopped = True
            if "stop_note" in signals:
                stopped = True
            if utt is not None and utt.size > 0:
                text = self.stt.transcribe(utt)
                if text:
                    self.store.append_transcript(note_id, text)
                    self.log.info("note: %s", text)

        self.say("Note taking stopped.", voice=False, commands=False)
        self.log.info("=== notetaking stopped (%s) — summarising ===", note_id)
        transcript = self.store.read_transcript(note_id)
        interrupted = False
        if transcript.strip():
            # summarize() and the folder dialogue play the idle "thinking" cue
            # themselves, around their model calls.
            title, spoken, full, category = self.llm.summarize(transcript)
            category = self._confirm_category(category, title, spoken)
            self.store.save_summary(note_id, title, full, category)
            self.log.info("saved '%s' -> %s", title, category)
            # Read the recap fully interruptible: the user can cut in by voice, or
            # use the headset button to mute/unmute or start a new note mid-summary.
            # A voice barge-in leaves the captured speech buffered so the next
            # conversation turn picks it up; a hotkey leaves self.interrupt set so
            # the main loop drains and acts on it (mute, new note, quit) — though
            # only a new note or quit cuts the recap short, since muting stops the
            # microphone, not the recap.
            # save_resume so a filler "yeah" mid-recap resumes it (and "continue"
            # works) instead of the recap being lost.
            interrupted = self.say(f"Notes saved. {spoken}", save_resume=True)
        else:
            interrupted = self.say("No speech was recorded, so nothing was saved.")
        if not interrupted:
            self.audio.flush()

    def _save_pending_note(self, pending: dict, saver: str = None):
        """Save a note the model prepared from the conversation (via the
        save_conversation_note tool): confirm the folder through the usual spoken
        dialogue, then file it exactly like a recorded note. `saver` names the
        persona that prepared it when that wasn't the active one (a delegated
        background save), so the shared history records who actually did it."""
        title = pending["title"]
        content = pending["content"]
        spoken = pending.get("spoken") or f"I've saved a note called {title}."
        suggested = self.store._match_category(pending.get("category")) or categories.DEFAULT_CATEGORY

        note_id = self.store.new_session()
        # The transcript preserves the SOURCE conversation, not a second copy
        # of the note body: the user's own words are the part that can't be
        # regenerated, and the summary file already holds the note. (Early
        # versions wrote `content` here too, so transcript == summary and the
        # actual spoken exchange was silently lost.)
        excerpt = self.llm.conversation_excerpt()
        self.store.append_transcript(
            note_id,
            "(Saved from conversation — the recent exchange this note was "
            "drawn from)\n\n" + (excerpt or content),
        )
        category = self._confirm_category(suggested, title, content[:300])
        self.store.save_summary(note_id, title, content, category)
        self.log.info("saved conversation note '%s' -> %s", title, category)
        # The folder dialogue and this save both happen after converse() returned,
        # in separate model memory the main history never sees — so on its own the
        # model still thinks the note is "pending" (the placeholder the tool
        # returned mid-turn). Record what actually happened and fold it in.
        display = categories.NOTE_CATEGORIES[category]["display"]
        actor = (f"{saver} (working in the background) completed a note save"
                 if saver else
                 "Completed the note the user asked me to save")
        # Deliberately no note id here. flush_tool_events files this sentence
        # under role=assistant, so whatever it contains becomes a pattern the
        # model reproduces in its *spoken* replies — and it reproduces the shape,
        # not the value. With an id in the template the model emitted invented
        # ids mid-turn, before the save had even run (2026-08-02 log: it spoke
        # note_2026-08-02_202012 sixty-six seconds before note_2026-08-02_202355
        # was minted), which TTS then read out digit by digit. Title and folder
        # are what the model actually needs to answer "did that save?"; when it
        # genuinely needs an id it has list_recent_notes.
        self.llm.record_tool_event(
            f"{actor} (save_conversation_note): "
            f"after a spoken folder-choice dialogue it was filed into the {display} "
            f"folder, titled '{title}'."
        )
        self.llm.flush_tool_events(persist=True)
        interrupted = self.say(f"Notes saved. {spoken}", save_resume=True)
        if not interrupted:
            self.audio.flush()

    # --- categorisation (spoken conversation) --------------------------------
    def _confirm_category(self, suggested: str, title: str, summary: str) -> str:
        """Decide the note's folder via a short back-and-forth: the agent proposes a
        folder and answers any questions until the user commits. `_ask` provides the
        speak-and-listen turn the dialogue drives."""
        # A brief confirmation ack; the saved-note recap that immediately follows
        # is fully interruptible, so this stays a short blocking line rather than
        # barge-in (whose retained audio the recap's flush would discard anyway).
        final = self.llm.choose_folder_via_dialogue(title, summary, suggested, self._ask)
        # Belt and braces: the dialogue validates its return, but the registry
        # is mutable (delete_folder) — never let a stale slug KeyError here and
        # lose the note.
        final = categories.valid_slug(final)
        self.say(f"Putting it into {categories.NOTE_CATEGORIES[final]['display']}.",
                 voice=False, commands=False)
        return final

    def _ask(self, prompt: str, endpoint_ms: int = None) -> str:
        """Speak a prompt, capture one spoken reply, and return its transcript.
        Interruptible by voice (start answering and your speech becomes the reply),
        but NOT by action commands — mute / note-taking shouldn't disrupt the
        folder-destination question, so commands=False keeps it playing through."""
        interrupted = self.say(prompt, commands=False, save_resume=False)
        if not interrupted:
            self.audio.flush()
        utt = self.audio.collect_utterance(
            interrupt=self.interrupt, endpoint_ms=endpoint_ms or cfg.CONVO_ENDPOINT_MS
        )
        signals = self._drain()
        if "quit" in signals:
            self.running = False
        if utt is None or utt.size == 0:
            return ""
        text = self.stt.transcribe(utt) or ""
        if text:
            self.log.info("you (folder choice): %s", text)
        return text

    # --- main loop -----------------------------------------------------------
    def run(self):
        self.audio.start()
        self.start_hotkeys()
        # The dashboard — live controls included — is served from inside this
        # process, so its buttons call straight into the methods that own the
        # state. Fails soft if the port is taken: a web page must never stop
        # the voice agent from starting.
        self._web = dashboard.serve_embedded(self)
        self.log.info(
            "Ready. Headset button: 1-click=mute  2-click=note  3-click=quit"
        )
        hat = agents.AGENTS[self.llm.active]
        self.tts.set_voice(hat["tts_voice"], hat["tts_rate"])
        self.say(f"Voice agent ready. {hat['name']} speaking.",
                 voice=False, commands=False)
        try:
            while self.running:
                try:
                    self.run_conversation_turn()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    # A model/API error (e.g. a transient 400/500) must not kill
                    # the whole session. Log it, tell the user, and carry on — the
                    # next turn re-sanitizes history so it self-heals.
                    self.log.exception("conversation turn failed; continuing")
                    # Speak the real cause when we can identify it: a billing
                    # error is not transient, and "let's try that again" once
                    # sent the user chasing a phantom bug for a whole session.
                    self.say(explain_error(e), voice=False, commands=False)
        except KeyboardInterrupt:
            pass
        finally:
            self._rescue_background_notes()
            # The port closes with the process anyway, but stopping cleanly
            # means an in-flight dashboard request gets a response, not a
            # reset.
            if getattr(self, "_web", None) is not None:
                self._web.stop()
            self.audio.stop()
            if getattr(self, "_media", None) is not None:
                self._media.stop()
            self.log.info("shut down")

    def _rescue_background_notes(self):
        """Quit safety: a note a background task prepared but never got to ask
        about must not evaporate with the process — file it into its suggested
        folder silently (we're shutting down; there is no one to ask). Tasks
        still mid-flight can't be rescued; their loss is at least logged."""
        alive = [t.name for t in self._delegation_threads if t.is_alive()]
        if alive:
            self.log.warning("quitting with background task(s) still running "
                             "(%s) — their results are lost",
                             ", ".join(alive))
        while True:
            try:
                item = self.interjections.get_nowait()
            except queue.Empty:
                return
            note = item.get("note")
            if not note:
                continue
            category = (self.store._match_category(note.get("category"))
                        or categories.DEFAULT_CATEGORY)
            note_id = self.store.new_session()
            self.store.append_transcript(
                note_id,
                "(Saved at shutdown from a background task — quit arrived "
                "before the folder question could be asked)\n\n"
                + note["content"],
            )
            self.store.save_summary(note_id, note["title"], note["content"],
                                    category)
            self.log.info("rescued background note '%s' -> %s "
                          "(quit before the folder question)",
                          note["title"], category)


def selftest():
    log = logging.getLogger("selftest")
    audio = AudioEngine()
    audio.start()
    try:
        log.info("[1/4] Speak now — recording 3 seconds...")
        clip = audio.record_seconds(3)
        stt = Transcriber()
        log.info("    heard: %r", stt.transcribe(clip))

        log.info("[2/4] Testing speech...")
        Speaker().speak("Self test. Text to speech is working.")

        log.info("[3/4] Testing Claude...")
        store = NoteStore()
        llm = Claude(store)
        log.info("    Claude says: %s", llm.converse("Say hello in five words."))

        log.info("[4/4] Testing note save + search...")
        nid = store.new_session()
        store.append_transcript(nid, "Testing the grocery list: milk, eggs, and bread.")
        store.save_summary(nid, "Grocery test", "## Summary\nA test grocery list.")
        log.info("    search('groceries'): %s", store.search_notes("groceries"))
        log.info("Self test complete.")
    finally:
        audio.stop()


def miccheck(seconds=20):
    """Print mic loudness so barge-in thresholds can be tuned to your setup.
    Stay silent for a few seconds, then speak normally, and compare the numbers."""
    import time as _time
    log = logging.getLogger("miccheck")
    audio = AudioEngine()
    audio.start()
    log.info("Mic check for %ds. Be SILENT first, then SPEAK. Watch the numbers.", seconds)
    log.info("Set BARGE_IN_ENERGY in config.py to roughly halfway between your "
             "silent RMS and your speaking RMS.")
    end = _time.monotonic() + seconds
    try:
        while _time.monotonic() < end:
            window = []  # ~300 ms of frames
            for _ in range(10):
                res = audio.poll_speech(timeout=1.0)
                if res is not None:
                    window.append(res)
            if not window:
                continue
            rms_vals = [r for _, r in window]
            voiced = sum(1 for s, _ in window if s)
            log.info("rms avg=%5.0f  max=%5.0f  voiced=%d/%d",
                     sum(rms_vals) / len(rms_vals), max(rms_vals), voiced, len(window))
    finally:
        audio.stop()


def main():
    parser = argparse.ArgumentParser(description="Local voice notetaking agent")
    parser.add_argument("--selftest", action="store_true",
                        help="Run component smoke tests and exit")
    parser.add_argument("--miccheck", action="store_true",
                        help="Print mic loudness to tune barge-in thresholds, then exit")
    parser.add_argument("--ingest", action="store_true",
                        help="Ingest PDFs/text/video from the knowledge/ folder into "
                             "the knowledge base, then exit (video is transcribed, "
                             "which can take minutes per hour of material)")
    parser.add_argument("--kb-list", action="store_true",
                        help="List ingested knowledge sources, then exit")
    parser.add_argument("--resync", action="store_true",
                        help="Repair note folder/frontmatter/Chroma inconsistencies, then exit")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("agent")
    # Read-only / mic-only modes need no lock.
    if args.miccheck:
        miccheck()
        return
    if args.kb_list:
        print(KnowledgeStore().list_sources())
        return
    # Everything below writes the shared state (history.json, index.json, the
    # Chroma stores), so it must hold the same single-instance lock as the live
    # agent — running --resync or --selftest beside a running agent would
    # race-corrupt exactly the data the lock exists to protect.
    try:
        lock = SingleInstance(cfg.LOCK_PATH).acquire()
    except AlreadyRunning:
        log.error("A voice agent is already running (lock: %s). Exiting.",
                  cfg.LOCK_PATH)
        if not (args.selftest or args.ingest or args.resync):
            try:  # a spoken heads-up too, since the agent is a screenless tool
                Speaker().speak("A voice agent is already running, so this copy will exit.")
            except Exception:  # noqa: BLE001 - never let the notice mask the exit
                pass
        sys.exit(1)
    try:
        if args.selftest:
            selftest()
        elif args.ingest:
            print(KnowledgeStore().ingest_folder())
        elif args.resync:
            print(NoteStore().resync())
        else:
            Agent().run()
    finally:
        lock.release()


if __name__ == "__main__":
    main()
