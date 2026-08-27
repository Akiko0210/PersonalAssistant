"""Windows TTS variant: SAPI directly via pywin32 (`SAPI.SpVoice`).

The only place win32com/pythoncom/winreg are touched for speech — and all of
them inside functions/constructors, so importing this module is free on any OS
(main.py imports every variant to keep its routing a pure mapping).
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


def _describe(voice) -> str:
    """Description of an SpVoice's current voice token ('' if unavailable)."""
    try:
        return voice.Voice.GetDescription()
    except Exception:  # noqa: BLE001 - purely informational
        return ""


class SapiSpeaker:
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


# --- Second voice (Announcer's Windows half) ---------------------------------
def announcer_available() -> bool:
    try:
        import pythoncom  # noqa: F401 - probing availability only
        import win32com.client  # noqa: F401
        return True
    except ImportError:
        log.info("no pywin32: spoken notices will wait for the main voice")
        return False


def announce(text, avoid_voice):
    """Speak a notice on a fresh SpVoice. Runs on the Announcer's throwaway
    daemon thread, which needs its own COM apartment (the main voice belongs
    to the thread that created it)."""
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    try:
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Rate = _wpm_to_sapi_rate(cfg.ANNOUNCE_RATE)
        voice.Volume = cfg.ANNOUNCE_VOLUME
        _pick_contrasting_voice(voice, avoid_voice)
        log.info("announcing (second voice): %s", text)
        voice.Speak(text)  # synchronous *on this thread* only
    finally:
        pythoncom.CoUninitialize()


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


# --- Voice enumeration (dashboard dropdown) ----------------------------------
def list_voices():
    """Installed SAPI voice names, read from the registry (stdlib winreg)
    rather than COM: the same tokens SAPI's GetVoices() returns, but with no
    pywin32 dependency and no per-thread CoInitialize headaches under the
    dashboard's ThreadingHTTPServer. [] on any failure."""
    voices = []
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Speech\Voices\Tokens") as k:
            for i in range(winreg.QueryInfoKey(k)[0]):
                name = winreg.EnumKey(k, i)
                try:
                    voices.append(winreg.QueryValue(k, name))
                except OSError:
                    pass
    except (OSError, ImportError):
        pass
    return voices
