"""Voice notetaking agent — entry point and orchestration.

Conversation mode is the default: listen for an utterance, answer with Claude,
speak the reply. Hotkeys switch into a silent notetaking mode and back, and
toggle a global mute. See README.md for setup and the hotkey list.
"""

import argparse
import collections
import logging
import queue
import sys
import threading
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()  # load ANTHROPIC_API_KEY (and any overrides) from a .env file
except ImportError:
    pass  # python-dotenv not installed yet; env vars still work

import config as cfg
from audio import AudioEngine
from stt import Transcriber
from tts import Speaker
from notes import NoteStore
from llm import Claude
from sound import IdleSound


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
        self.idle = IdleSound()  # "thinking" cue, looped during model calls
        self.llm = Claude(self.store, self.idle)
        self.log.info("loading speech model...")
        self.stt = Transcriber()
        self.log.info("startup took %.1fs", time.monotonic() - t0)
        self.status = "conversation_mode"

        self.cmds: "queue.Queue[str]" = queue.Queue()
        self.interrupt = threading.Event()
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


    def _media_click(self):
        import threading
        import time

        now = time.monotonic()

        # Ignore accidental duplicate events from one physical press
        last = getattr(self, "_last_media_click", 0)
        if now - last < 0.08:
            return

        self._last_media_click = now

        if not hasattr(self, "_media_click_lock"):
            self._media_click_lock = threading.Lock()
            self._media_click_count = 0
            self._media_click_timer = None

        with self._media_click_lock:
            self._media_click_count += 1

            if self._media_click_timer:
                self._media_click_timer.cancel()

            self._media_click_timer = threading.Timer(
                0.45,
                self._finish_media_clicks,
            )
            self._media_click_timer.daemon = True
            self._media_click_timer.start()


    def _finish_media_clicks(self):
        with self._media_click_lock:
            count = self._media_click_count
            self._media_click_count = 0
            self._media_click_timer = None

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
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.media_play_pause:
                # Diagnostic: if this line is absent from the log while the agent
                # is speaking but present when it's idle, the OS/headset is eating
                # the button during audio playback (it never reaches us).
                self.log.info("media key received (speaking=%s)", self.tts.is_busy())
                self._media_click()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

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

        # "continue" after an interruption — resume where we left off
        if self._interrupted_remaining and text.strip().lower() in (
            "continue", "go on", "keep going", "go ahead",
        ):
            self.log.info("resuming interrupted reply")
            interrupted = self.say(self._interrupted_remaining, save_resume=True)
            if not interrupted:
                self._interrupted_reply = None
                self._interrupted_remaining = None
                self.audio.flush()
            return

        self._interrupted_reply = None
        self._interrupted_remaining = None

        reply = self.llm.converse(text)
        self.log.info("agent: %s", reply)
        interrupted = self.say(reply, save_resume=True)
        if not interrupted:
            self.audio.flush()

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
        self.audio.flush()
        start = time.monotonic()
        self.tts.begin(text)

        voiced_ms = 0.0
        calib_ms = 0
        echo_samples = []
        threshold = float(cfg.BARGE_IN_ENERGY)  # until calibration finishes
        peak_speech_rms = 0.0  # loudest voiced frame seen — logged for tuning

        # Retain the audio we consume while deciding this is a real barge-in, so
        # the opening words aren't lost. `recent` keeps a short pre-roll; once a
        # qualifying voiced run starts, `run` accumulates it (pre-roll included)
        # and is pushed back to the mic when the interruption fires.
        pad_frames = max(1, cfg.SPEECH_PAD_MS // cfg.FRAME_MS)
        recent = collections.deque(maxlen=pad_frames)
        run = None

        while self.tts.is_busy():
            if self.interrupt.is_set():   # a hotkey command arrived while speaking
                # Diagnostic: shows the flag DID reach the speaking loop. If a
                # failed click never logs this, the interrupt was never set (the
                # press didn't get through on_press -> _media_click -> _push).
                if commands:
                    self.tts.stop()
                    return False
            res = self.audio.poll_speech(timeout=0.1, return_frame=True)
            if res is None:
                continue
            is_speech, rms, frame = res
            recent.append(frame)

            # Calibrate the echo floor from the first part of playback (the user
            # almost never barges in this quickly), then lock the threshold.
            if calib_ms < cfg.BARGE_IN_CALIB_MS:
                echo_samples.append(rms)
                calib_ms += cfg.FRAME_MS
                if calib_ms >= cfg.BARGE_IN_CALIB_MS and echo_samples:
                    echo_samples.sort()
                    # A low percentile rather than the median: if the user starts
                    # talking *during* calibration, their loud frames shouldn't
                    # inflate the echo baseline. An inflated baseline pushes the
                    # threshold so high the rest of this utterance can't be
                    # interrupted — the "sometimes I can't barge in" failure.
                    baseline = echo_samples[len(echo_samples) // 3]
                    threshold = max(cfg.BARGE_IN_ENERGY,
                                    baseline * cfg.BARGE_IN_ENERGY_RATIO)
                    self.log.info("barge-in armed (echo baseline=%.0f, threshold=%.0f)",
                                  baseline, threshold)
                continue  # don't allow interruption during calibration

            if is_speech:
                peak_speech_rms = max(peak_speech_rms, rms)

            # Only loud, voiced audio counts — this rejects the agent's own echo.
            # The counter leaks (rather than hard-resetting) on non-qualifying
            # frames so brief VAD/energy dropouts mid-speech don't wipe progress.
            if is_speech and rms > threshold:
                run = list(recent) if run is None else run + [frame]
                voiced_ms += cfg.FRAME_MS
                if voiced_ms >= cfg.BARGE_IN_MS:
                    self.tts.stop()
                    self.audio.pushback(run)  # give the spoken words back to capture
                    if save_resume:
                        self._save_interrupted(text, time.monotonic() - start)
                    self.log.info("(interrupted — listening)")
                    return True
            else:
                voiced_ms = max(0.0, voiced_ms - cfg.FRAME_MS * cfg.BARGE_IN_DECAY)
                if run is not None:
                    run.append(frame)        # keep brief gaps within the run
                    if voiced_ms <= 0:
                        run = None            # run fizzled — it was a false start

        # Finished speaking without an interruption. If the user clearly spoke but
        # we never triggered, the threshold is probably too high — surface the
        # numbers so BARGE_IN_ENERGY can be tuned.
        if peak_speech_rms > 0:
            self.log.info("reply finished; loudest speech rms=%.0f vs threshold=%.0f"
                          " (no barge-in)", peak_speech_rms, threshold)
        return False

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
            interrupted = self.say(f"Notes saved. {spoken}")
        else:
            interrupted = self.say("No speech was recorded, so nothing was saved.")
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
        self.say(f"Putting it into {cfg.NOTE_CATEGORIES[final]['display']}.",
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
                self.run_conversation_turn()
        except KeyboardInterrupt:
            pass
        finally:
            self.audio.stop()
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
    args = parser.parse_args()

    setup_logging()
    if args.selftest:
        selftest()
    elif args.miccheck:
        miccheck()
    else:
        Agent().run()


if __name__ == "__main__":
    main()
