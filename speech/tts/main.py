"""Local text-to-speech — the integrator over the per-OS variant files.

One primary backend per OS, each supporting asynchronous speaking + purge,
which is what enables barge-in (interrupting the agent mid-sentence), and
pause + resume, which is what lets a mute click leave a reply intact instead
of cutting it off:
  - `windows.py`: SAPI directly via pywin32;
  - `macos.py`: NSSpeechSynthesizer via pyobjc;
  - `linux.py`: speech-dispatcher's SSIP client.
`fallback.py` (pyttsx3) is the universal fallback, but it speaks synchronously
only (no barge-in, no pause there — a mute click then cuts the reply off).
All variant modules keep their OS imports lazy, so importing them here is free
on every OS — which is what keeps `_select_backend` a pure, testable mapping.

`Announcer` is a deliberately separate second voice for short notices that must
be heard *while* the main voice is talking — see its docstring.
"""

import logging
import sys
import threading

import config as cfg  # noqa: F401 - variants read it; kept for parity/tunables
from speech.tts import fallback, linux, macos, windows

log = logging.getLogger("tts")


def _select_backend(platform=None):
    """The async backend class for this OS. Pure mapping (no construction),
    so the platform routing is testable anywhere."""
    platform = platform or sys.platform
    if platform == "win32":
        return windows.SapiSpeaker
    if platform == "darwin":
        return macos.NSSpeaker
    if platform.startswith("linux"):
        return linux.SpeechdSpeaker
    raise RuntimeError(f"no native TTS backend for {platform}")


class Speaker:
    def __init__(self):
        try:
            cls = _select_backend()
            self._backend = cls()
            log.info("TTS backend: %s", cls.__name__)
        except Exception as e:  # noqa: BLE001 - fall back on any backend issue
            log.warning("native TTS unavailable (%s); falling back to pyttsx3", e)
            self._backend = fallback.Pyttsx3Speaker()

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

    # --- async / interruptible API (native backends only) --------------------
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
            # Linux: speechd serialises through one connection like SAPI does
            # through one voice, so there is no cheap second channel yet.
            self._impl = None
            log.info("no second voice on this OS: "
                     "spoken notices will wait for the main voice")
        self.available = bool(self._impl) and self._impl.announcer_available()

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
    variant can enumerate cheaply (winreg / `say -v ?`). [] elsewhere and on
    any failure — the UI then falls back to a free-text input."""
    if sys.platform == "win32":
        return windows.list_voices()
    if sys.platform == "darwin":
        return macos.list_voices()
    return []
