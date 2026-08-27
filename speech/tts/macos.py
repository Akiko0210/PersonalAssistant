"""macOS TTS variant: NSSpeechSynthesizer via pyobjc.

Deprecated by Apple but the only macOS API with pause + continue, which the
mute-click gesture depends on; AVSpeechSynthesizer is the successor if it ever
breaks. The only place AppKit is imported for speech — and ONLY inside
functions/constructors: the standalone dashboard's instant start depends on
importing this module (via main.py) staying free, so never hoist an AppKit
import to module level.
"""

import logging
import time

import config as cfg

log = logging.getLogger("tts")


class NSSpeaker:
    """macOS backend on NSSpeechSynthesizer with async speak + interrupt.

    startSpeakingString_ is asynchronous from any thread, and completion is
    observed by polling isSpeaking() — deliberately no delegate callbacks,
    which would need a runloop this plain-Python process doesn't pump."""

    supports_async = True

    def __init__(self):
        from AppKit import NSSpeechSynthesizer
        self._cls = NSSpeechSynthesizer
        self._synth = NSSpeechSynthesizer.alloc().initWithVoice_(None)
        if self._synth is None:
            raise RuntimeError("NSSpeechSynthesizer init failed")
        self._synth.setRate_(float(cfg.TTS_RATE))  # takes wpm directly
        self._paused = False
        self._voice_desc = ""
        if cfg.TTS_VOICE:
            self.set_voice(cfg.TTS_VOICE)
        else:
            self._voice_desc = str(self._synth.voice() or "")

    def set_voice(self, substring, rate_wpm=None):
        """Same contract as SapiSpeaker.set_voice: first identifier containing
        `substring` (e.g. 'Samantha' matches
        'com.apple.voice.compact.en-US.Samantha'), fail-soft on no match. Rate
        is set AFTER the voice — setVoice_ resets it to the voice default."""
        if substring:
            for ident in self._cls.availableVoices():
                if substring.lower() in str(ident).lower():
                    self._synth.setVoice_(ident)
                    self._voice_desc = str(ident)
                    break
            else:
                log.info("no installed voice matches %r; keeping current voice",
                         substring)
        self._synth.setRate_(float(rate_wpm or cfg.TTS_RATE))

    def current_voice(self) -> str:
        return self._voice_desc

    def speak(self, text: str):
        self.begin(text)
        while self.is_busy():
            time.sleep(0.05)

    def begin(self, text: str):
        self.resume()  # never start new speech into a paused voice
        self._synth.startSpeakingString_(text)  # returns immediately

    def is_busy(self) -> bool:
        # OR-ing _paused keeps the "busy while paused" contract even if
        # isSpeaking() reports False across a pause.
        return self._paused or bool(self._synth.isSpeaking())

    def pause(self):
        """Suspend playback, keeping the utterance intact so it can resume.
        Unlike stop(), nothing is discarded — this is what lets a reply
        survive a button press that turns out to be a mute."""
        if not self._paused:
            self._synth.pauseSpeakingAtBoundary_(0)  # 0 = immediate boundary
            self._paused = True

    def resume(self):
        if self._paused:
            self._synth.continueSpeaking()
            self._paused = False

    def stop(self):
        # Resume first, mirroring SAPI: stopping a paused synthesizer is the
        # under-documented corner; running-then-stop is the reliable path.
        self.resume()
        self._synth.stopSpeaking()


# --- Second voice (Announcer's macOS half) -----------------------------------
def announcer_available() -> bool:
    try:
        from AppKit import NSSpeechSynthesizer  # noqa: F401 - probe
        return True
    except ImportError:
        log.info("no pyobjc: spoken notices will wait for the main voice")
        return False


def announce(text, avoid_voice):
    """Speak a notice on a fresh synthesizer: NSSpeechSynthesizer instances
    speak concurrently, which is the whole point of the second voice. Voice is
    set BEFORE volume/rate — setVoice_ resets both to voice defaults."""
    from AppKit import NSSpeechSynthesizer
    synth = NSSpeechSynthesizer.alloc().initWithVoice_(None)
    if avoid_voice and str(synth.voice() or "") == avoid_voice:
        # Clash with the main voice: switch to a different voice in the
        # SAME locale — unlike Windows' two-or-three installed voices,
        # macOS lists ~100 across every language, so "any other voice"
        # would land on another language. Identifiers look like
        # com.apple.voice.compact.en-US.Samantha.
        locale = avoid_voice.rsplit(".", 2)[-2] if "." in avoid_voice else ""
        for ident in NSSpeechSynthesizer.availableVoices():
            s = str(ident)
            if s != avoid_voice and (not locale or f".{locale}." in s):
                synth.setVoice_(ident)
                break
    synth.setVolume_(cfg.ANNOUNCE_VOLUME / 100.0)  # NS volume is 0.0-1.0
    synth.setRate_(float(cfg.ANNOUNCE_RATE))
    log.info("announcing (second voice): %s", text)
    synth.startSpeakingString_(text)
    while synth.isSpeaking():  # poll; no runloop for delegate callbacks
        time.sleep(0.05)


# --- Voice enumeration (dashboard dropdown) ----------------------------------
def _parse_say_voices(text):
    """Voice names out of `say -v ?` output. Each line is
    `Name (maybe with spaces)   locale   # sample sentence`; the name is
    everything before the locale token. Pure, for testing on any OS."""
    voices = []
    for line in text.splitlines():
        left = line.split("#", 1)[0].rstrip()
        parts = left.split()
        if len(parts) >= 2:
            voices.append(" ".join(parts[:-1]))
    return voices


def list_voices():
    """Installed voice names via `say -v ?` — zero imports, ~50 ms, and the
    names substring-match NSSpeechSynthesizer identifiers, so a dashboard
    dropdown value feeds set_voice() unchanged. (AppKit here would break the
    standalone dashboard's instant start.) [] on any failure."""
    try:
        import subprocess
        out = subprocess.run(["say", "-v", "?"], capture_output=True,
                             text=True, timeout=5)
        return _parse_say_voices(out.stdout)
    except Exception:  # noqa: BLE001 - a voice list must never break a page
        return []
