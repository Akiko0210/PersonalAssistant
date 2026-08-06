"""Local text-to-speech for Windows.

Primary backend talks to SAPI directly via pywin32 (`SAPI.SpVoice`). It supports
asynchronous speaking + purge, which is what enables barge-in (interrupting the
agent mid-sentence), and pause + resume, which is what lets a mute click leave
a reply intact instead of cutting it off. pyttsx3 is kept as a fallback for
non-Windows / missing pywin32 setups, but it speaks synchronously only (no
barge-in, no pause there).

`Announcer` is a deliberately separate second voice for short notices that must
be heard *while* the main voice is talking — see its docstring.
"""

import logging
import threading

import config as cfg

log = logging.getLogger("tts")

# SAPI speak flags
_SVSF_ASYNC = 1
_SVSF_PURGE = 2


def _wpm_to_sapi_rate(wpm: int) -> int:
    # SAPI rate is an int in [-10, 10]; 0 is roughly 200 wpm.
    return max(-10, min(10, round((wpm - 200) / 20)))


def _describe(voice) -> str:
    """Description of an SpVoice's current voice token ('' if unavailable)."""
    try:
        return voice.Voice.GetDescription()
    except Exception:  # noqa: BLE001 - purely informational
        return ""


class _SapiSpeaker:
    """Direct Windows SAPI backend with async speak + interrupt."""

    supports_async = True

    def __init__(self):
        import win32com.client  # part of pywin32
        self._voice = win32com.client.Dispatch("SAPI.SpVoice")
        self._voice.Rate = _wpm_to_sapi_rate(cfg.TTS_RATE)
        self._paused = False
        self._voice_desc = ""
        if cfg.TTS_VOICE:
            self.set_voice(cfg.TTS_VOICE)
        else:
            self._voice_desc = _describe(self._voice)

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
                self._voice_desc = token.GetDescription()
                return
        log.info("no installed voice matches %r; keeping current voice", substring)

    def current_voice(self) -> str:
        """Description of the voice now speaking — the Announcer uses it to
        pick a *different* one, so overlapping speech stays tellable apart."""
        return self._voice_desc

    def speak(self, text: str):
        self._voice.Speak(text)  # synchronous, blocks until done

    def begin(self, text: str):
        self.resume()  # never start new speech into a paused voice
        self._voice.Speak(text, _SVSF_ASYNC)  # returns immediately

    def is_busy(self) -> bool:
        # WaitUntilDone(0) returns True if speech has already finished. A
        # paused utterance isn't finished, so this stays True across a pause.
        return not self._voice.WaitUntilDone(0)

    def pause(self):
        """Suspend playback, keeping the utterance intact so it can resume.
        Unlike stop(), nothing is discarded — this is what lets a reply
        survive a button press that turns out to be a mute."""
        if not self._paused:
            self._voice.Pause()
            self._paused = True

    def resume(self):
        if self._paused:
            self._voice.Resume()
            self._paused = False

    def stop(self):
        # Purge the current + pending speech, ending playback immediately.
        # Resume first: a purge issued to a paused voice isn't acted on until
        # the voice runs again, which would leave the speech hanging.
        self.resume()
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

    def pause(self) -> bool:
        """Suspend playback so it can be resumed later. Returns False if the
        backend can't pause (pyttsx3, or a SAPI voice that refused) — the
        caller then has only stop() to fall back on."""
        fn = getattr(self._backend, "pause", None)
        if fn is None:
            return False
        try:
            fn()
            return True
        except Exception:  # noqa: BLE001 - a failed pause must not kill the reply
            log.exception("TTS pause failed")
            return False

    def resume(self):
        fn = getattr(self._backend, "resume", None)
        if fn is None:
            return
        try:
            fn()
        except Exception:  # noqa: BLE001 - fall through to stop() at worst
            log.exception("TTS resume failed")

    def stop(self):
        self._backend.stop()

    def current_voice(self) -> str:
        fn = getattr(self._backend, "current_voice", None)
        return fn() if fn else ""


class Announcer:
    """A second, independent voice for short notices spoken *over* the main one.

    One SpVoice serialises everything sent to it: a notice handed to the
    speaking voice would queue behind the reply (heard minutes later) or purge
    it (the reply lost). Neither is what "Muted." should do — the point of the
    acknowledgement is to land the instant the button does, while the reply
    carries on. So the notice gets its own voice object, in its own thread and
    COM apartment (the main voice belongs to the thread that created it, and
    the mute click arrives on a button/timer thread).

    Kept quieter than the reply, and on a different installed voice where the
    machine has one, so two simultaneous voices remain tellable apart.
    """

    def __init__(self):
        try:
            import pythoncom  # noqa: F401 - probing availability only
            import win32com.client  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False
            log.info("no pywin32: spoken notices will wait for the main voice")

    def announce(self, text: str, avoid_voice: str = "") -> bool:
        """Say `text` over whatever is already playing. Returns False when a
        second voice isn't available, so the caller can fall back to saying it
        on the main voice once that frees up. Never raises and never blocks —
        the notice is spoken on a throwaway daemon thread."""
        if not text or not self.available:
            return False
        threading.Thread(target=self._speak, args=(text, avoid_voice),
                         daemon=True, name="announcer").start()
        return True

    def _speak(self, text, avoid_voice):
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()  # this thread needs its own COM apartment
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            voice.Rate = _wpm_to_sapi_rate(cfg.ANNOUNCE_RATE)
            voice.Volume = cfg.ANNOUNCE_VOLUME
            self._pick_contrasting_voice(voice, avoid_voice)
            log.info("announcing (second voice): %s", text)
            voice.Speak(text)  # synchronous *on this thread* only
        except Exception:  # noqa: BLE001 - a notice must never crash the agent
            log.exception("second-voice announcement failed: %s", text)
        finally:
            pythoncom.CoUninitialize()

    @staticmethod
    def _pick_contrasting_voice(voice, avoid_voice):
        """Switch to any installed voice other than `avoid_voice`. A machine
        with a single voice simply keeps it — the volume and rate difference
        still mark the notice apart."""
        if not avoid_voice:
            return
        for token in voice.GetVoices():
            if token.GetDescription() != avoid_voice:
                voice.Voice = token
                return
