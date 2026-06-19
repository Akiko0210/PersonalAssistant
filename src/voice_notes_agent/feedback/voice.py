"""Voice feedback manager: earcons + spoken confirmations + working cue (§5.9, §10).

This is the entire UX surface when the screen is untouched (§C5). It owns:
  * the earcon vocabulary playback (FR-V1)
  * spoken confirmations where they add information (FR-V2) — e.g. "Listening",
    "Recording notes", "Stopped. Summarizing."
  * the rule that muting is earcon-only, never a spoken "muted" (FR-V3)
  * a subtle periodic "working" tick during long operations (FR-V4)
  * audible error announcements: error earcon + brief spoken reason (FR-V5)

Playback (`play`) and speech (`speak`) are injected so the app can route them to the
output device / TTS provider, and so tests can capture calls without audio hardware.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

from .earcons import SAMPLE_RATE, Earcon, render

log = logging.getLogger(__name__)

PlayFn = Callable[[np.ndarray, int], None]   # (audio, sample_rate)
SpeakFn = Callable[[str], None]              # text -> TTS playback


class VoiceFeedback:
    def __init__(
        self,
        cfg,
        *,
        play: PlayFn,
        speak: SpeakFn,
    ) -> None:
        self._cfg = cfg.feedback
        self._play = play
        self._speak = speak
        self._working: threading.Thread | None = None
        self._working_stop = threading.Event()

    # -- primitives -----------------------------------------------------------
    def earcon(self, which: Earcon) -> None:
        if self._cfg.earcons:
            self._play(render(which, volume=self._cfg.volume, sr=SAMPLE_RATE), SAMPLE_RATE)

    def say(self, text: str) -> None:
        if self._cfg.spoken_confirmations and text:
            self._speak(text)

    # -- §10 event vocabulary -------------------------------------------------
    def listening(self) -> None:
        self.earcon(Earcon.LISTENING)
        self.say("Listening")

    def muted(self) -> None:
        # Earcon only — never speak "muted" while going silent (FR-V3).
        self.earcon(Earcon.MUTED)

    def recording_notes(self) -> None:
        self.earcon(Earcon.START_NOTES)
        self.say("Recording notes")

    def stopped_summarizing(self) -> None:
        self.earcon(Earcon.STOP_NOTES)
        self.say("Stopped. Summarizing.")

    def summary_ready(self, spoken_summary: str) -> None:
        self.earcon(Earcon.SUMMARY_READY)
        self.say(spoken_summary)

    def status(self, text: str) -> None:
        self.say(text)

    def error(self, reason: str) -> None:
        self.earcon(Earcon.ERROR)
        self.say(reason)

    # -- working cue (FR-V4) --------------------------------------------------
    def start_working_cue(self) -> None:
        """Begin a subtle periodic tick so silence isn't mistaken for a hang."""
        if not self._cfg.earcons or self._working is not None:
            return
        self._working_stop.clear()
        interval = self._cfg.working_cue_interval_sec

        def _loop() -> None:
            while not self._working_stop.wait(interval):
                self.earcon(Earcon.WORKING)

        self._working = threading.Thread(target=_loop, name="working-cue", daemon=True)
        self._working.start()

    def stop_working_cue(self) -> None:
        if self._working is not None:
            self._working_stop.set()
            self._working.join(timeout=1.0)
            self._working = None


# -- Deepgram REST TTS for feedback speech outside the conversation loop ----------
def deepgram_speak_factory(cfg, play: PlayFn) -> SpeakFn:
    """Build a ``speak`` function that synthesizes via Deepgram and plays locally.

    Used for confirmations/summaries spoken while NOT in the Pipecat pipeline (e.g. the
    auto read-back right after capture stops, before LISTENING resumes). Network errors
    degrade gracefully to a no-op so feedback never blocks the state machine.
    """
    import os

    voice = cfg.providers.tts.voice

    def speak(text: str) -> None:  # pragma: no cover - network + audio
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            log.warning("DEEPGRAM_API_KEY unset; cannot speak: %s", text)
            return
        try:
            import io

            import requests
            import soundfile as sf

            resp = requests.post(
                f"https://api.deepgram.com/v1/speak?model={voice}&encoding=linear16"
                f"&sample_rate={SAMPLE_RATE}",
                headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
                json={"text": text},
                timeout=15,
            )
            resp.raise_for_status()
            audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
            play(np.asarray(audio, dtype=np.float32), sr)
        except Exception:
            log.exception("TTS failed for: %s", text)

    return speak
