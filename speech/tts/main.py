"""Local text-to-speech — the integrator over the per-OS variant files.

One native backend per OS (Linux: two, tried in order), each supporting
asynchronous speaking + purge, which is what enables barge-in (interrupting
the agent mid-sentence), and pause + resume, which is what lets a mute click
leave a reply intact instead of cutting it off:
  - `windows.py`: SAPI directly via pywin32;
  - `macos.py`: NSSpeechSynthesizer via pyobjc;
  - `linux.py`: Piper (neural, via sounddevice), then speech-dispatcher.
`fallback.py` (pyttsx3) is the universal fallback, but it speaks synchronously
only (no barge-in, no pause there — a mute click then cuts the reply off).
All variant modules keep their OS imports lazy, so importing them here is free
on every OS — which is what keeps `_select_backend` a pure, testable mapping.

`Announcer` is a deliberately separate second voice for short notices that must
be heard *while* the main voice is talking — see its docstring.
"""

import logging
import re
import sys
import threading
import time

import config as cfg  # noqa: F401 - variants read it; kept for parity/tunables
from speech.tts import fallback, linux, macos, windows

log = logging.getLogger("tts")


def estimate_chars(text, seconds, wpm):
    """Where an utterance said for `seconds` at `wpm` words a minute has got
    to: the offset of the first word not yet reached (len(text) once every
    word has been). Whole words only — a word in progress was not heard."""
    starts = [m.start() for m in re.finditer(r"\S+", text)]
    reached = int(max(0.0, seconds) * wpm / 60)
    return starts[reached] if reached < len(starts) else len(text)


def _select_backend(platform=None):
    """The async backend classes for this OS, best first. Pure mapping (no
    construction), so the platform routing is testable anywhere."""
    platform = platform or sys.platform
    if platform == "win32":
        return (windows.SapiSpeaker,)
    if platform == "darwin":
        return (macos.NSSpeaker,)
    if platform.startswith("linux"):
        return (linux.PiperSpeaker, linux.SpeechdSpeaker)
    raise RuntimeError(f"no native TTS backend for {platform}")


class Speaker:
    def __init__(self):
        self._backend = None
        try:
            for cls in _select_backend():
                try:
                    self._backend = cls()
                    break
                except Exception as e:  # noqa: BLE001 - try the next variant
                    log.warning("%s unavailable (%s)", cls.__name__, e)
        except RuntimeError as e:
            log.warning("%s", e)
        if self._backend is None:
            log.warning("native TTS unavailable; falling back to pyttsx3")
            self._backend = fallback.Pyttsx3Speaker()
        log.info("TTS backend: %s", type(self._backend).__name__)
        # The utterance in flight, for spoken_chars(): its text, the rate it
        # is said at, when it began, and how long it has sat paused.
        self._text, self._wpm = "", cfg.TTS_RATE
        self._began, self._paused_at, self._paused_s = 0.0, None, 0.0

    @property
    def supports_async(self) -> bool:
        return self._backend.supports_async

    def set_voice(self, substring, rate_wpm=None):
        """Per-persona voice switching (see agents.py). No-op on the pyttsx3
        fallback — there the personas share the one system voice."""
        self._wpm = rate_wpm or cfg.TTS_RATE
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

    # --- async / interruptible API (native backends only) --------------------
    def begin(self, text: str):
        log.info("speaking: %s", text)
        self._text, self._began = text, time.monotonic()
        self._paused_at, self._paused_s = None, 0.0
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
            if self._paused_at is None:
                self._paused_at = time.monotonic()
            return True
        except Exception:  # noqa: BLE001 - a failed pause must not kill the reply
            log.exception("TTS pause failed")
            return False

    def resume(self):
        if self._paused_at is not None:
            self._paused_s += time.monotonic() - self._paused_at
            self._paused_at = None
        fn = getattr(self._backend, "resume", None)
        if fn is None:
            return
        try:
            fn()
        except Exception:  # noqa: BLE001 - fall through to stop() at worst
            log.exception("TTS resume failed")

    def stop(self):
        self._backend.stop()

    def spoken_chars(self) -> int:
        """How far the current utterance has got: the character offset where
        the word being said right now begins, so text[:offset] is what the
        user has heard. Read it BEFORE stop() — a purge resets the voice.
        A backend that can report its position (SAPI) is exact; the rest are
        estimated from speaking time at the voice's rate, pauses excluded —
        a mute click can hold a reply paused for seconds."""
        exact = getattr(self._backend, "spoken_chars", None)
        if exact is not None:
            try:
                return max(0, min(len(self._text), int(exact())))
            except Exception:  # noqa: BLE001 - an estimate beats no answer
                log.exception("TTS position read failed; estimating instead")
        now = self._paused_at or time.monotonic()
        return estimate_chars(self._text, now - self._began - self._paused_s,
                              self._wpm)

    def current_voice(self) -> str:
        fn = getattr(self._backend, "current_voice", None)
        return fn() if fn else ""


class Announcer:
    """A second, independent voice for short notices spoken *over* the main one.

    One voice object serialises everything sent to it: a notice handed to the
    speaking voice would queue behind the reply (heard minutes later) or purge
    it (the reply lost). Neither is what "Muted." should do — the point of the
    acknowledgement is to land the instant the button does, while the reply
    carries on. So the notice gets its own voice object, in its own thread
    (the main voice belongs to the thread that created it, and the mute click
    arrives on a button/timer thread) — see announce() in the variant files.

    Kept quieter than the reply, and on a different installed voice where the
    machine has one, so two simultaneous voices remain tellable apart.
    """

    def __init__(self):
        if sys.platform == "win32":
            self._impl = windows
        elif sys.platform == "darwin":
            self._impl = macos
        else:
            self._impl = linux
        self.available = self._impl.announcer_available()

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
        try:
            self._impl.announce(text, avoid_voice)
        except Exception:  # noqa: BLE001 - a notice must never crash the agent
            log.exception("second-voice announcement failed: %s", text)


def list_voices():
    """Installed voice names for the dashboard's voice dropdown, from whichever
    variant can enumerate cheaply (winreg / `say -v ?` / a directory scan).
    [] on any failure — the UI then falls back to a free-text input."""
    if sys.platform == "win32":
        return windows.list_voices()
    if sys.platform == "darwin":
        return macos.list_voices()
    return linux.list_voices()
