"""Local text-to-speech for Windows.

Primary backend talks to SAPI directly via pywin32 (`SAPI.SpVoice`). It supports
asynchronous speaking + purge, which is what enables barge-in (interrupting the
agent mid-sentence). pyttsx3 is kept as a fallback for non-Windows / missing
pywin32 setups, but it speaks synchronously only (no barge-in there).
"""

import logging

import config as cfg

log = logging.getLogger("tts")

# SAPI speak flags
_SVSF_ASYNC = 1
_SVSF_PURGE = 2


def _wpm_to_sapi_rate(wpm: int) -> int:
    # SAPI rate is an int in [-10, 10]; 0 is roughly 200 wpm.
    return max(-10, min(10, round((wpm - 200) / 20)))


class _SapiSpeaker:
    """Direct Windows SAPI backend with async speak + interrupt."""

    supports_async = True

    def __init__(self):
        import win32com.client  # part of pywin32
        self._voice = win32com.client.Dispatch("SAPI.SpVoice")
        self._voice.Rate = _wpm_to_sapi_rate(cfg.TTS_RATE)
        if cfg.TTS_VOICE:
            self.set_voice(cfg.TTS_VOICE)

    def set_voice(self, substring, rate_wpm=None):
        """Switch to the first installed voice whose description contains
        `substring` (case-insensitive), at `rate_wpm` (None = config default).
        Fail-soft: an unknown/None voice keeps the current one — on a machine
        with only Zira and David installed, a third persona simply shares a
        voice and the spoken announcement carries the switch signal. Call only
        between utterances, from the thread that owns the COM object."""
        self._voice.Rate = _wpm_to_sapi_rate(rate_wpm or cfg.TTS_RATE)
        if not substring:
            return
        for token in self._voice.GetVoices():
            if substring.lower() in token.GetDescription().lower():
                self._voice.Voice = token
                return
        log.info("no installed voice matches %r; keeping current voice", substring)

    def speak(self, text: str):
        self._voice.Speak(text)  # synchronous, blocks until done

    def begin(self, text: str):
        self._voice.Speak(text, _SVSF_ASYNC)  # returns immediately

    def is_busy(self) -> bool:
        # WaitUntilDone(0) returns True if speech has already finished.
        return not self._voice.WaitUntilDone(0)

    def stop(self):
        # Purge the current + pending speech, ending playback immediately.
        self._voice.Speak("", _SVSF_ASYNC | _SVSF_PURGE)


class _Pyttsx3Speaker:
    """Fallback backend. Synchronous only — no barge-in. Recreates the engine
    per call to dodge the speaks-only-once bug as best as possible."""

    supports_async = False

    def speak(self, text: str):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", cfg.TTS_RATE)
        engine.say(text)
        engine.runAndWait()
        engine.stop()


class Speaker:
    def __init__(self):
        try:
            self._backend = _SapiSpeaker()
            log.info("TTS backend: Windows SAPI")
        except Exception as e:  # noqa: BLE001 - fall back on any SAPI/pywin32 issue
            log.warning("SAPI unavailable (%s); falling back to pyttsx3", e)
            self._backend = _Pyttsx3Speaker()

    @property
    def supports_async(self) -> bool:
        return self._backend.supports_async

    def set_voice(self, substring, rate_wpm=None):
        """Per-persona voice switching (see agents.py). No-op on the pyttsx3
        fallback — there the personas share the one system voice."""
        backend_set = getattr(self._backend, "set_voice", None)
        if backend_set is None:
            return
        try:
            backend_set(substring, rate_wpm)
        except Exception:  # noqa: BLE001 - a voice change must never crash speech
            log.exception("set_voice(%r) failed; keeping current voice", substring)

    def speak(self, text: str):
        """Blocking speak (used when interruption isn't needed)."""
        text = (text or "").strip()
        if not text:
            return
        log.info("speaking: %s", text)
        try:
            self._backend.speak(text)
        except Exception:  # noqa: BLE001 - never let TTS crash the agent loop
            log.exception("TTS failed for: %s", text)

    # --- async / interruptible API (SAPI backend only) -----------------------
    def begin(self, text: str):
        log.info("speaking: %s", text)
        self._backend.begin(text)

    def is_busy(self) -> bool:
        return self._backend.is_busy()

    def stop(self):
        self._backend.stop()
