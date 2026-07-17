"""Voice notetaking agent — entry point and orchestration.

Conversation mode is the default: listen for an utterance, answer with Claude,
speak the reply. Hotkeys switch into a silent notetaking mode and back, and
toggle a global mute. See README.md for setup and the hotkey list.
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

import categories
import config as cfg
from audio import AudioEngine
from barge_in import BargeInDetector
from gestures import ClickGestureDecoder
from stt import Transcriber
from tts import Speaker
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
        self.store = NoteStore()
        # Absorb any newly added trading PDFs before we start listening. This is an
        # idempotent scan: unchanged files are skipped by content hash without
        # loading the embedding model, so it's near-instant unless there's a genuinely
        # new book (which is embedded once, here, blocking startup that one time).
        self.kb = KnowledgeStore()
        self.log.info("scanning knowledge base...")
        self.log.info(self.kb.ingest_folder())
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
        # Raw button presses (from either listener channel) are decoded into
        # single/double/triple gestures here; see gestures.py.
        self._gesture = ClickGestureDecoder(
            on_gesture=self._on_media_gesture, on_press=self._on_media_press
        )
        self.running = True
        self._interrupted_reply = None   # full text of the last interrupted reply
        self._interrupted_remaining = None  # unsaid portion after barge-in

    # --- command plumbing ----------------------------------------------------
    def _push(self, cmd):
        self.cmds.put(cmd)
        self.interrupt.set()

    def _drain(self):
        """Process queued hotkey commands. Returns the set of high-level signals
        ('start_note', 'stop_note', 'quit'); mute is handled inline."""
        signals = set()
        while not self.cmds.empty():
            cmd = self.cmds.get()
            if cmd == "toggle_mute":
                if self.audio.muted.is_set():
                    self.audio.muted.clear()
                    self.log.info("unmuted — listening")
                    self.say("Listening.", voice=False, commands=False)
                else:
                    self.audio.muted.set()
                    self.log.info("muted — not listening")
                    self.say("Muted.", voice=False, commands=False)
            else:
                signals.add(cmd)
        self.interrupt.clear()
        return signals

    def _toggle_note(self):
        if self.status == "conversation_mode":
            self.status = "note_taking"
            if self.audio.muted.is_set():
                self.audio.muted.clear()
                self.log.info("auto-unmuted for notetaking")
            self._push("start_note")
        else:
            self.status = "conversation_mode"
            self._push("stop_note")


    def _on_media_press(self):
        """Every accepted click: go silent the moment the press lands — stop the
        thinking cue here, and flag say() to stop speech on its next poll
        (~100 ms; hush avoids touching the SAPI COM object cross-thread). Every
        gesture's action implies silence anyway — and it keeps a state-tracking
        dongle (Yealink) in sync: the dongle swallows the next press whenever
        the host keeps playing through a "pause", which is how the 2nd/3rd
        clicks of a gesture were getting eaten. See MEDIA_CONTROL.md."""
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
            self._push("toggle_mute")
        elif count == 2:
            self.log.info("media hotkey: double click -> toggle note-taking")
            self._toggle_note()
        else:
            self.log.info("media hotkey: triple click -> quit")
            self._push("quit")

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
                on_previous=lambda: self._push("quit"),
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
        utt = self.audio.collect_utterance(
            interrupt=self.interrupt, endpoint_ms=cfg.CONVO_ENDPOINT_MS
        )
        signals = self._drain()
        if "quit" in signals:
            self.running = False
            return
        if "start_note" in signals:
            self.run_notetaking()
            return
        if utt is None or utt.size == 0:
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

        reply = self._converse_with_followups(text)
        if not reply:
            return  # a hotkey command cut the turn short; main loop handles it
        self.log.info("agent: %s", reply)
        interrupted = self.say(reply, save_resume=True)

        # The user asked to save something from the conversation as a note: the
        # model prepared it via save_conversation_note; now run the same folder
        # dialogue + save flow a finished note-taking session gets.
        pending = self.llm.take_pending_note()
        if pending:
            self._save_pending_note(pending)
            return

        if not interrupted:
            self.audio.flush()

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
            if self.interrupt.is_set():
                # A hotkey command (mute / note-taking / quit) arrived — don't
                # call the model, but keep the words: the user said them, and
                # they must not vanish because of a mute click mid-settle.
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
            if self.interrupt.is_set():
                self.llm.record_unanswered(text)
                return ""
            if more is not None and more.size > 0:
                addition = self.stt.transcribe(more)
                if addition:
                    self.log.info("you (continued): %s", addition)
                    text = f"{text} {addition}"
            # else: a cough / room noise triggered the settle — nothing to add,
            # and (unlike before) nothing was spent. Loop and keep settling.
        if self.interrupt.is_set():
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
        commands: stop when an action command (mute / note-taking) arrives. Turned
                  off only for the folder-destination question, so those commands
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
        self.hush.clear()  # only clicks during *this* utterance may hush it
        start = time.monotonic()
        self.tts.begin(text)
        detector = BargeInDetector()

        while self.tts.is_busy():
            # A raw click just landed: stop speaking NOW, before the multi-click
            # window even resolves into a command — the dongle swallows the
            # gesture's next click if playback runs on (see MEDIA_CONTROL.md).
            # The command itself arrives via _push/interrupt moments later.
            if commands and self.hush.is_set():
                self.hush.clear()
                self.tts.stop()
                return False
            if self.interrupt.is_set():   # a hotkey command arrived while speaking
                if commands:
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

    def _drain_buffered_speech(self):
        """Drain the mic buffer, watching for a speech onset. Returns the
        onset's frames (pre-roll included) if the user is/was talking — the
        caller pushes them back so no words are lost — or None once the buffer
        runs dry having held only silence/noise (discarded, as flush() would).
        Frames after a detected onset stay queued for collect_utterance."""
        pad_frames = max(1, cfg.SPEECH_PAD_MS // cfg.FRAME_MS)
        ring = collections.deque(maxlen=pad_frames)
        while True:
            res = self.audio.poll_speech(timeout=0.05, return_frame=True)
            if res is None:
                return None
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
            # the main loop drains and acts on it (mute, new note, quit).
            # save_resume so a filler "yeah" mid-recap resumes it (and "continue"
            # works) instead of the recap being lost.
            interrupted = self.say(f"Notes saved. {spoken}", save_resume=True)
        else:
            interrupted = self.say("No speech was recorded, so nothing was saved.")
        if not interrupted:
            self.audio.flush()

    def _save_pending_note(self, pending: dict):
        """Save a note the model prepared from the conversation (via the
        save_conversation_note tool): confirm the folder through the usual spoken
        dialogue, then file it exactly like a recorded note."""
        title = pending["title"]
        content = pending["content"]
        spoken = pending.get("spoken") or f"I've saved a note called {title}."
        suggested = self.store._match_category(pending.get("category")) or categories.DEFAULT_CATEGORY

        note_id = self.store.new_session()
        # The "transcript" of a conversation note is the note body itself, so the
        # note keeps the same two-file layout as recorded notes.
        self.store.append_transcript(note_id, f"(Saved from conversation)\n\n{content}")
        category = self._confirm_category(suggested, title, content[:300])
        self.store.save_summary(note_id, title, content, category)
        self.log.info("saved conversation note '%s' -> %s", title, category)
        # The folder dialogue and this save both happen after converse() returned,
        # in separate model memory the main history never sees — so on its own the
        # model still thinks the note is "pending" (the placeholder the tool
        # returned mid-turn). Record what actually happened and fold it in.
        display = categories.NOTE_CATEGORIES[category]["display"]
        self.llm.record_tool_event(
            f"Completed the note the user asked me to save (save_conversation_note): "
            f"after a spoken folder-choice dialogue it was filed into the {display} "
            f"folder and saved as note {note_id} — titled '{title}'."
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
        self.log.info(
            "Ready. Headset button: 1-click=mute  2-click=note  3-click=quit"
        )
        self.say("Voice agent ready. Conversation mode.", voice=False, commands=False)
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
            self.audio.stop()
            if getattr(self, "_media", None) is not None:
                self._media.stop()
            self.log.info("shut down")


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
                        help="Ingest PDFs/text from the knowledge/ folder into the knowledge base, then exit")
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
