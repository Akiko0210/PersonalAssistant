"""Linux TTS variant: speech-dispatcher's SSIP client.

`import speechd` comes from the `python3-speechd` system package, not PyPI —
and stays inside the constructor so importing this module is free everywhere.
No second-voice half here: speechd serialises through one connection like SAPI
does through one voice, so there is no cheap second channel yet (main.py's
Announcer reports unavailable on Linux). No list_voices either — the dashboard
falls back to a free-text voice field.
"""

import logging
import time

import config as cfg

log = logging.getLogger("tts")


def _wpm_to_speechd_rate(wpm: int) -> int:
    # speechd rate is an int in [-100, 100]; 0 is the engine default (~180 wpm
    # for espeak). Same anchor-and-scale shape as the SAPI mapping.
    return max(-100, min(100, round((wpm - 180) / 2)))


class SpeechdSpeaker:
    """Linux backend on speech-dispatcher (SSIP). Async with pause/resume, so
    the gesture design carries over. UNTESTED on real hardware so far — it
    fails soft to pyttsx3 like every other backend."""

    supports_async = True

    def __init__(self):
        import speechd  # python3-speechd (system package)
        self._speechd = speechd
        self._client = speechd.SSIPClient("voice-agent")
        self._client.set_rate(_wpm_to_speechd_rate(cfg.TTS_RATE))
        self._busy = False
        self._paused = False
        if cfg.TTS_VOICE:
            self.set_voice(cfg.TTS_VOICE)

    def set_voice(self, substring, rate_wpm=None):
        self._client.set_rate(_wpm_to_speechd_rate(rate_wpm or cfg.TTS_RATE))
        if not substring:
            return
        try:
            for name, _lang, _variant in self._client.list_synthesis_voices():
                if substring.lower() in name.lower():
                    self._client.set_synthesis_voice(name)
                    return
        except Exception:  # noqa: BLE001 - voice listing varies by backend
            pass
        log.info("no installed voice matches %r; keeping current voice", substring)

    def speak(self, text: str):
        self.begin(text)
        while self.is_busy():
            time.sleep(0.05)

    def begin(self, text: str):
        self.resume()
        self._busy = True
        # END fires on completion, CANCEL on stop(); pause suppresses both, so
        # is_busy stays True across a pause — same contract as the other
        # backends.
        self._client.speak(
            text, callback=self._done,
            event_types=(self._speechd.CallbackType.END,
                         self._speechd.CallbackType.CANCEL))

    def _done(self, *_args):
        self._busy = False

    def is_busy(self) -> bool:
        return self._busy

    def pause(self):
        if not self._paused:
            self._client.pause()
            self._paused = True

    def resume(self):
        if self._paused:
            self._client.resume()
            self._paused = False

    def stop(self):
        self.resume()
        self._client.stop()
        self._busy = False
