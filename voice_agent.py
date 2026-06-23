"""Voice notetaking agent — entry point and orchestration.

Conversation mode is the default: listen for an utterance, answer with Claude,
speak the reply. Hotkeys switch into a silent notetaking mode and back, and
toggle a global mute. See README.md for setup and the hotkey list.
"""

import argparse
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
        self.llm = Claude(self.store)
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
                    self.tts.speak("Listening.")
                else:
                    self.audio.muted.set()
                    self.log.info("muted — not listening")
                    self.tts.speak("Muted.")
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


    def start_controls(self):
        """Bluetooth/headset media buttons via Windows SMTC (PC-free):
          play/pause (single press)  -> toggle notetaking
          next       (double press)  -> toggle mute
          previous   (triple press)  -> quit
        """
        from media_control import MediaButtonListener
        self._media = MediaButtonListener(
            on_play_pause=self._toggle_note,
            on_next=lambda: self._push("toggle_mute"),
            on_previous=lambda: self._push("quit"),
            session_title="Voice Agent",
        )
        self._media.start()

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
            interrupted = self.speak_with_barge_in(self._interrupted_remaining)
            if not interrupted:
                self._interrupted_reply = None
                self._interrupted_remaining = None
                self.audio.flush()
            return

        self._interrupted_reply = None
        self._interrupted_remaining = None

        reply = self.llm.converse(text)
        self.log.info("agent: %s", reply)
        interrupted = self.speak_with_barge_in(reply)
        if not interrupted:
            self.audio.flush()

    def speak_with_barge_in(self, text: str) -> bool:
        """Speak `text`, stopping early if the user starts talking. Returns True
        if interrupted by speech. Falls back to blocking speech when barge-in is
        disabled or unsupported by the TTS backend."""
        if not (cfg.BARGE_IN and self.tts.supports_async):
            self.tts.speak(text)
            return False
        self.audio.flush()
        start = time.monotonic()
        self.tts.begin(text)

        voiced_ms = 0
        calib_ms = 0
        echo_samples = []
        threshold = float(cfg.BARGE_IN_ENERGY)  # until calibration finishes

        while self.tts.is_busy():
            if self.interrupt.is_set():   # a hotkey command arrived
                self.tts.stop()
                return False
            res = self.audio.poll_speech(timeout=0.1)
            if res is None:
                continue
            is_speech, rms = res

            # Calibrate the echo floor from the first part of playback (the user
            # almost never barges in this quickly), then lock the threshold.
            if calib_ms < cfg.BARGE_IN_CALIB_MS:
                echo_samples.append(rms)
                calib_ms += cfg.FRAME_MS
                if calib_ms >= cfg.BARGE_IN_CALIB_MS and echo_samples:
                    echo_samples.sort()
                    baseline = echo_samples[len(echo_samples) // 2]  # median
                    threshold = max(cfg.BARGE_IN_ENERGY,
                                    baseline * cfg.BARGE_IN_ENERGY_RATIO)
                    self.log.debug("barge-in echo baseline=%.0f threshold=%.0f",
                                   baseline, threshold)
                continue  # don't allow interruption during calibration

            # Only loud, voiced audio counts — this rejects the agent's own echo.
            if is_speech and rms > threshold:
                voiced_ms += cfg.FRAME_MS
                if voiced_ms >= cfg.BARGE_IN_MS:
                    self.tts.stop()
                    self._save_interrupted(text, time.monotonic() - start)
                    self.log.info("(interrupted — listening)")
                    return True
            else:
                voiced_ms = 0
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
        self.tts.speak("Note taking started.")

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

        self.tts.speak("Note taking stopped.")
        self.log.info("=== notetaking stopped (%s) — summarising ===", note_id)
        transcript = self.store.read_transcript(note_id)
        if transcript.strip():
            title, spoken, full = self.llm.summarize(transcript)
            self.store.save_summary(note_id, title, full)
            self.log.info("saved '%s'", title)
            self.tts.speak(f"Notes saved. {spoken}")
        else:
            self.tts.speak("No speech was recorded, so nothing was saved.")
        self.audio.flush()

    # --- main loop -----------------------------------------------------------
    def run(self):
        self.audio.start()
        self.start_controls()
        self.log.info(
            "Ready. Headset: play/pause=notetaking  next=mute  previous=quit"
        )
        self.tts.speak("Voice agent ready. Conversation mode.")
        try:
            while self.running:
                self.run_conversation_turn()
        except KeyboardInterrupt:
            pass
        finally:
            if getattr(self, "_media", None):
                self._media.stop()
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
