"""Universal fallback variant: pyttsx3 — works everywhere, but synchronously
only, so no barge-in and no pause (a mute click then cuts the reply off)."""

import config as cfg


class Pyttsx3Speaker:
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
