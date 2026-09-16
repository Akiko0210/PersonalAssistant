"""Linux TTS variants: Piper first, speech-dispatcher as the fallback.

Both keep their library imports inside constructors/functions, so importing
this module is free everywhere (main.py imports every variant to keep its
routing a pure mapping).

`PiperSpeaker` is the voice worth listening to: Piper is a local neural
synthesizer (`piper-tts` on PyPI), and its voices sound like a person where
espeak-ng — what speech-dispatcher and pyttsx3 both default to on Linux —
sounds like a 1980s modem. Piper produces PCM, not playback, so this file
plays it through sounddevice (already the microphone's library) on a callback
stream; that is also what makes pause/resume/stop trivially exact — pausing
just fills the callback with silence, keeping the rest of the utterance
intact, which is what the mute gesture depends on. Voices are files: the
configured default is fetched once into PIPER_VOICE_DIR (~60 MB) the way
faster-whisper fetches its model, and any other downloaded voice is picked
by name substring, the same contract as the SAPI/macOS variants.

`SpeechdSpeaker` (speech-dispatcher's SSIP client, `python3-speechd` system
package) stays as the fallback for a machine without piper; there is no
second-voice half for it, since speechd serialises through one connection.
"""

import logging
import threading
import time
from pathlib import Path

import config as cfg

log = logging.getLogger("tts")


# --- Piper --------------------------------------------------------------------
def _wpm_to_length_scale(wpm: int) -> float:
    # Piper stretches time by length_scale (1.0 = the voice's native pace,
    # roughly the config default of 175 wpm for the medium voices; smaller is
    # faster). Same anchor-and-scale shape as the SAPI mapping.
    return round(175 / max(int(wpm), 60), 2)


def _installed_voices(voice_dir=None):
    """Voice names (file stems like en_US-lessac-medium) present on disk."""
    voice_dir = Path(voice_dir or cfg.PIPER_VOICE_DIR)
    return sorted(p.stem for p in voice_dir.glob("*.onnx")) if voice_dir.is_dir() else []


def _find_voice(substring, voice_dir=None):
    """Path of the first downloaded voice whose name contains `substring`
    (case-insensitive); None when nothing matches, so callers fail soft."""
    if not substring:
        return None
    for name in _installed_voices(voice_dir):
        if substring.lower() in name.lower():
            return Path(voice_dir or cfg.PIPER_VOICE_DIR) / f"{name}.onnx"
    return None


def _default_voice():
    """Path of the configured default voice, downloading it on first use."""
    path = Path(cfg.PIPER_VOICE_DIR) / f"{cfg.PIPER_VOICE}.onnx"
    if not path.is_file():
        from piper.download_voices import download_voice
        log.info("downloading Piper voice %s to %s (once, ~60 MB)",
                 cfg.PIPER_VOICE, cfg.PIPER_VOICE_DIR)
        path.parent.mkdir(parents=True, exist_ok=True)
        download_voice(cfg.PIPER_VOICE, path.parent)
    return path


class PiperSpeaker:
    """Linux backend on Piper + sounddevice with async speak, pause/resume and
    purge. Synthesis streams sentence by sentence on a feeder thread into a
    block list the output callback drains, so speech starts after the first
    sentence rather than the whole reply."""

    supports_async = True

    def __init__(self, volume=1.0):
        from piper import PiperVoice, SynthesisConfig  # piper-tts (pip)
        import sounddevice as sd  # fail here, not mid-reply
        self._PiperVoice, self._SynthesisConfig, self._sd = PiperVoice, SynthesisConfig, sd
        self._volume = volume
        self._voice = None
        self._voice_desc = ""
        self._length_scale = _wpm_to_length_scale(cfg.TTS_RATE)
        self._lock = threading.Lock()   # guards everything the callback reads
        self._pcm = []                  # int16 blocks still to play, head first
        self._feeding = False           # synthesis still producing blocks
        self._paused = False
        self._stream = None
        self._gen = 0                   # a stale feeder recognises itself by it
        self.set_voice(cfg.TTS_VOICE)

    def set_voice(self, substring, rate_wpm=None):
        """Same contract as SapiSpeaker.set_voice: first downloaded voice whose
        name contains `substring`, fail-soft on no match (the persona then
        keeps the current voice; only the configured default is ever
        downloaded). Call between utterances."""
        self._length_scale = _wpm_to_length_scale(rate_wpm or cfg.TTS_RATE)
        path = _find_voice(substring)
        if substring and path is None:
            log.info("no downloaded voice matches %r; keeping current voice", substring)
        if path is None and self._voice is None:
            path = _default_voice()
        if path is not None and path.stem != self._voice_desc:
            self._voice = self._PiperVoice.load(path)
            self._voice_desc = path.stem

    def current_voice(self) -> str:
        return self._voice_desc

    def speak(self, text: str):
        self.begin(text)
        while self.is_busy():
            time.sleep(0.05)

    def begin(self, text: str):
        self.stop()
        with self._lock:
            self._feeding = True      # busy from this instant, before any PCM exists
            self._gen += 1
            gen = self._gen
            self._stream = self._sd.OutputStream(
                samplerate=self._voice.config.sample_rate, channels=1,
                dtype="int16", callback=self._fill)
            self._stream.start()
        threading.Thread(target=self._feed, args=(text, gen),
                         daemon=True, name="tts-synth").start()
        # Return only once the first sentence is ready to play: the barge-in
        # detector calibrates its echo baseline from the first moments after
        # begin(), and measuring synthesis latency instead of the voice would
        # lock its threshold on silence — the reply's own echo could then
        # interrupt it.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._pcm or not self._feeding:
                    return
            time.sleep(0.005)

    def _feed(self, text, gen):
        try:
            config = self._SynthesisConfig(length_scale=self._length_scale,
                                           volume=self._volume)
            for chunk in self._voice.synthesize(text, config):
                with self._lock:
                    if gen != self._gen:
                        return            # stop()/begin() moved on without us
                    self._pcm.append(chunk.audio_int16_array)
        except Exception:  # noqa: BLE001 - a bad sentence must not hang is_busy
            log.exception("Piper synthesis failed")
        finally:
            with self._lock:
                if gen == self._gen:
                    self._feeding = False

    def _fill(self, outdata, frames, _time, _status):
        """sounddevice callback: hand over the next `frames` samples, silence
        while paused or while synthesis is still ahead of us, and end the
        stream once the last block has been played."""
        outdata.fill(0)
        with self._lock:
            if self._paused:
                return
            i = 0
            while i < frames and self._pcm:
                head = self._pcm[0]
                n = min(frames - i, len(head))
                outdata[i:i + n, 0] = head[:n]
                i += n
                if n == len(head):
                    self._pcm.pop(0)
                else:
                    self._pcm[0] = head[n:]
            if not self._pcm and not self._feeding:
                raise self._sd.CallbackStop

    def is_busy(self) -> bool:
        # Paused counts as busy: the utterance isn't done — same contract as
        # the other backends.
        with self._lock:
            return self._feeding or bool(self._pcm) or self._paused

    def pause(self):
        """Suspend playback, keeping the rest of the utterance intact — this is
        what lets a reply survive a button press that turns out to be a mute."""
        with self._lock:
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False

    def stop(self):
        with self._lock:
            self._gen += 1
            self._pcm.clear()
            self._feeding = False
            self._paused = False
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.abort(ignore_errors=True)
            stream.close(ignore_errors=True)


# --- Second voice (Announcer's Linux half) ------------------------------------
_announcer = None


def announcer_available() -> bool:
    """True once Piper is importable AND a voice is on disk — the main Speaker
    is built first, so if its download failed there is no voice here either
    and the notice must go to whatever voice the reply ended up on."""
    try:
        import piper  # noqa: F401 - probing availability only
        import sounddevice  # noqa: F401
        if _installed_voices():
            return True
    except ImportError:
        pass
    log.info("no Piper voice: spoken notices will wait for the main voice")
    return False


def announce(text, avoid_voice):
    """Speak a notice on a second PiperSpeaker: sounddevice streams mix, so it
    plays over the reply. Kept for the life of the process (loading a voice
    model per notice would cost the instant reaction the notice exists for);
    the Announcer runs it on its own daemon thread."""
    global _announcer
    if _announcer is None:
        _announcer = PiperSpeaker(volume=cfg.ANNOUNCE_VOLUME / 100.0)
    # Any other downloaded voice than the reply's; a machine with one voice
    # keeps it — the volume and rate difference still mark the notice apart.
    other = next((v for v in _installed_voices() if v != avoid_voice), None)
    _announcer.set_voice(other, cfg.ANNOUNCE_RATE)
    log.info("announcing (second voice): %s", text)
    _announcer.speak(text)


# --- Voice enumeration (dashboard dropdown) ----------------------------------
def list_voices():
    """Downloaded Piper voice names — the stems feed set_voice() unchanged.
    A directory scan, so it costs the standalone dashboard nothing."""
    try:
        return _installed_voices()
    except Exception:  # noqa: BLE001 - a voice list must never break a page
        return []


# --- speech-dispatcher (fallback when piper isn't installed) ------------------
def _wpm_to_speechd_rate(wpm: int) -> int:
    # speechd rate is an int in [-100, 100]; 0 is the engine default (~180 wpm
    # for espeak). Same anchor-and-scale shape as the SAPI mapping.
    return max(-100, min(100, round((wpm - 180) / 2)))


class SpeechdSpeaker:
    """Linux fallback on speech-dispatcher (SSIP). Async with pause/resume, so
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
