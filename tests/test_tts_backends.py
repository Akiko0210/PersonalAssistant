"""Tests for the per-OS TTS variant seam (speech/tts/).

Covers the routing (which variant classes each platform gets, in order, and
the honest fallback to pyttsx3 when none constructs), the NSSpeaker contract
against a stub synthesizer — no pyobjc needed, same __new__-and-hand-install
trick as agent_fixtures — the PiperSpeaker contract against a stub voice and
output stream (the callback is pumped by hand), and the per-OS voice
enumeration the dashboard dropdown delegates to. The busy-while-paused
behaviour is the load-bearing one: _hold_for_gesture's resume path depends on
it (see FakeTTS in test_mute_speech.py, which models the same contract)."""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import config as cfg
from speech.tts import fallback, linux, macos, main, windows


class FakeSynth:
    """Records the NSSpeechSynthesizer calls NSSpeaker makes."""

    def __init__(self):
        self.calls = []
        self.speaking = False
        self._voice = "com.apple.voice.compact.en-US.Samantha"

    def setRate_(self, rate):
        self.calls.append(("rate", rate))

    def setVoice_(self, ident):
        self.calls.append(("voice", str(ident)))
        self._voice = str(ident)

    def voice(self):
        return self._voice

    def startSpeakingString_(self, text):
        self.calls.append(("speak", text))
        self.speaking = True

    def isSpeaking(self):
        return self.speaking

    def pauseSpeakingAtBoundary_(self, boundary):
        self.calls.append(("pause", boundary))

    def continueSpeaking(self):
        self.calls.append(("continue",))

    def stopSpeaking(self):
        self.calls.append(("stop",))
        self.speaking = False


def make_ns_speaker(voices=()):
    """An NSSpeaker without touching AppKit."""
    sp = macos.NSSpeaker.__new__(macos.NSSpeaker)
    sp._synth = FakeSynth()
    sp._paused = False
    sp._voice_desc = ""
    sp._cls = mock.Mock(availableVoices=lambda: list(voices))
    return sp


class SelectBackendTests(unittest.TestCase):
    def test_platform_routing(self):
        self.assertEqual(main._select_backend("win32"), (windows.SapiSpeaker,))
        self.assertEqual(main._select_backend("darwin"), (macos.NSSpeaker,))
        # Linux: Piper (the voice worth hearing) before speech-dispatcher
        self.assertEqual(main._select_backend("linux"),
                         (linux.PiperSpeaker, linux.SpeechdSpeaker))

    def test_speaker_tries_the_next_variant_when_one_wont_construct(self):
        class Broken:
            def __init__(self):
                raise ImportError("no piper")

        class Good:
            supports_async = True

        with mock.patch.object(main, "_select_backend",
                               return_value=(Broken, Good)):
            speaker = main.Speaker()
        self.assertIsInstance(speaker._backend, Good)

    def test_unknown_platform_raises(self):
        with self.assertRaises(RuntimeError):
            main._select_backend("solaris")

    def test_speaker_falls_back_to_pyttsx3(self):
        # The honest-failure path: native backend won't construct (missing
        # pyobjc/pywin32/speechd) -> the sync fallback, never a crash.
        with mock.patch.object(main, "_select_backend",
                               side_effect=RuntimeError("no backend")):
            speaker = main.Speaker()
        self.assertIsInstance(speaker._backend, fallback.Pyttsx3Speaker)
        self.assertFalse(speaker.supports_async)
        self.assertFalse(speaker.pause())  # callers get the stop()-only signal


class AsyncContractTests(unittest.TestCase):
    def test_every_async_variant_exposes_the_gesture_api(self):
        # The verbs say()/_hold_for_gesture drive, per FakeTTS's contract.
        for cls in (windows.SapiSpeaker, macos.NSSpeaker,
                    linux.PiperSpeaker, linux.SpeechdSpeaker):
            self.assertTrue(cls.supports_async, cls.__name__)
            for verb in ("begin", "is_busy", "pause", "resume", "stop",
                         "speak", "set_voice"):
                self.assertTrue(callable(getattr(cls, verb)),
                                f"{cls.__name__}.{verb}")


class NSSpeakerTests(unittest.TestCase):
    def test_busy_stays_true_across_a_pause(self):
        sp = make_ns_speaker()
        sp.begin("hello")
        sp.pause()
        # Whatever isSpeaking() claims while paused, the utterance isn't done.
        sp._synth.speaking = False
        self.assertTrue(sp.is_busy())
        sp.resume()
        sp._synth.speaking = True
        self.assertTrue(sp.is_busy())

    def test_pause_is_sent_once(self):
        sp = make_ns_speaker()
        sp.begin("hello")
        sp.pause()
        sp.pause()
        self.assertEqual([c for c in sp._synth.calls if c[0] == "pause"],
                         [("pause", 0)])

    def test_stop_resumes_first_so_the_purge_lands(self):
        sp = make_ns_speaker()
        sp.begin("hello")
        sp.pause()
        sp.stop()
        calls = [c[0] for c in sp._synth.calls]
        self.assertLess(calls.index("continue"), calls.index("stop"))
        self.assertFalse(sp._paused)
        self.assertFalse(sp.is_busy())

    def test_begin_never_starts_into_a_paused_voice(self):
        sp = make_ns_speaker()
        sp.pause()
        sp.begin("hello")
        calls = [c[0] for c in sp._synth.calls]
        self.assertLess(calls.index("continue"), calls.index("speak"))

    def test_set_voice_matches_substring_then_sets_rate(self):
        sp = make_ns_speaker(voices=["com.apple.voice.compact.en-US.Samantha",
                                     "com.apple.voice.compact.en-GB.Daniel"])
        sp.set_voice("daniel", rate_wpm=190)
        self.assertEqual(sp.current_voice(),
                         "com.apple.voice.compact.en-GB.Daniel")
        # rate AFTER voice: setVoice_ resets rate to the voice default
        voice_i = sp._synth.calls.index(
            ("voice", "com.apple.voice.compact.en-GB.Daniel"))
        self.assertIn(("rate", 190.0), sp._synth.calls[voice_i:])

    def test_unknown_voice_keeps_the_current_one(self):
        sp = make_ns_speaker(voices=["com.apple.voice.compact.en-US.Samantha"])
        sp._voice_desc = "com.apple.voice.compact.en-US.Samantha"
        sp.set_voice("zira")
        self.assertEqual(sp.current_voice(),
                         "com.apple.voice.compact.en-US.Samantha")
        self.assertNotIn("voice", [c[0] for c in sp._synth.calls])


# --- Piper ---------------------------------------------------------------------
class FakePiperVoice:
    """Stands in for piper.PiperVoice: two 50-sample 'sentences' per text."""
    config = SimpleNamespace(sample_rate=100)

    def __init__(self, path):
        self.path = Path(path)

    def synthesize(self, text, _config):
        for _sentence in text.split(". "):
            yield SimpleNamespace(audio_int16_array=np.full(50, 7, np.int16))


class FakeStream:
    """Stands in for sounddevice.OutputStream; the test pumps the callback."""
    instances = []

    def __init__(self, samplerate, channels, dtype, callback):
        self.callback = callback
        self.started = self.aborted = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def abort(self, ignore_errors=False):
        self.aborted = True

    def close(self, ignore_errors=False):
        pass

    def pump(self, frames):
        """One callback: (samples handed over, stream ended?)."""
        out = np.zeros((frames, 1), np.int16)
        try:
            self.callback(out, frames, None, None)
            return out[:, 0], False
        except FakeSD.CallbackStop:
            return out[:, 0], True


class FakeSD:
    OutputStream = FakeStream

    class CallbackStop(Exception):
        pass


class PiperSpeakerTests(unittest.TestCase):
    def setUp(self):
        FakeStream.instances = []
        self.voice_dir = Path(tempfile.mkdtemp())
        for name in ("en_US-lessac-medium", "en_US-ryan-medium"):
            (self.voice_dir / f"{name}.onnx").touch()
        fake_piper = SimpleNamespace(
            PiperVoice=SimpleNamespace(load=FakePiperVoice),
            SynthesisConfig=lambda **kw: kw)
        patches = [
            mock.patch.dict(sys.modules, {"piper": fake_piper, "sounddevice": FakeSD}),
            mock.patch.object(cfg, "PIPER_VOICE_DIR", self.voice_dir),
            mock.patch.object(cfg, "TTS_VOICE", None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.sp = linux.PiperSpeaker()

    def begin(self, text="One. Two"):
        self.sp.begin(text)
        deadline = time.monotonic() + 2
        while self.sp._feeding and time.monotonic() < deadline:
            time.sleep(0.01)  # let the synthesis thread hand over every block
        return FakeStream.instances[-1]

    def test_default_voice_when_none_configured(self):
        self.assertEqual(self.sp.current_voice(), cfg.PIPER_VOICE)

    def test_begin_returns_with_the_first_sentence_ready(self):
        # so the barge-in echo calibration measures the voice, not the wait
        self.sp.begin("One. Two")
        self.assertTrue(self.sp._pcm)

    def test_announcer_needs_a_downloaded_voice(self):
        self.assertTrue(linux.announcer_available())
        with mock.patch.object(cfg, "PIPER_VOICE_DIR", "/nonexistent/piper"):
            self.assertFalse(linux.announcer_available())

    def test_busy_from_begin_until_the_last_block_is_played(self):
        stream = self.begin()
        self.assertTrue(stream.started)
        self.assertTrue(self.sp.is_busy())
        played = 0
        for _ in range(20):
            out, ended = stream.pump(30)
            played += int(np.count_nonzero(out))
            if ended:
                break
        self.assertTrue(ended)
        self.assertEqual(played, 100)  # both sentences, nothing dropped
        self.assertFalse(self.sp.is_busy())

    def test_pause_plays_silence_and_keeps_the_rest(self):
        stream = self.begin()
        stream.pump(30)
        self.sp.pause()
        out, ended = stream.pump(30)
        self.assertEqual(int(np.count_nonzero(out)), 0)
        self.assertFalse(ended)
        self.assertTrue(self.sp.is_busy())   # paused is not done
        self.sp.resume()
        out, _ = stream.pump(30)
        self.assertEqual(int(np.count_nonzero(out)), 30)  # carries on where it left

    def test_stop_aborts_the_stream_and_purges(self):
        stream = self.begin()
        self.sp.pause()
        self.sp.stop()
        self.assertTrue(stream.aborted)
        self.assertFalse(self.sp.is_busy())
        self.assertFalse(self.sp._paused)  # a new utterance never starts paused

    def test_set_voice_matches_substring_and_fails_soft(self):
        self.sp.set_voice("ryan", rate_wpm=350)
        self.assertEqual(self.sp.current_voice(), "en_US-ryan-medium")
        self.assertEqual(self.sp._length_scale, 0.5)  # twice the pace
        self.sp.set_voice("zira")
        self.assertEqual(self.sp.current_voice(), "en_US-ryan-medium")

    def test_linux_voice_list_is_the_downloaded_voices(self):
        with mock.patch.object(main.sys, "platform", "linux"):
            self.assertEqual(main.list_voices(),
                             ["en_US-lessac-medium", "en_US-ryan-medium"])


class VoiceListTests(unittest.TestCase):
    SAY_OUTPUT = (
        "Albert              en_US    # Hello! My name is Albert.\n"
        "Bad News            en_US    # The light you see at the end...\n"
        "Kyoko               ja_JP    # こんにちは、私の名前はKyokoです。\n"
        "\n"
    )

    def test_say_output_parses_names_including_spaced_ones(self):
        self.assertEqual(macos._parse_say_voices(self.SAY_OUTPUT),
                         ["Albert", "Bad News", "Kyoko"])

    def test_linux_without_a_voice_dir_answers_empty(self):
        # honest [] so the UI shows free text
        with mock.patch.object(main.sys, "platform", "linux"), \
                mock.patch.object(cfg, "PIPER_VOICE_DIR", "/nonexistent/piper"):
            self.assertEqual(main.list_voices(), [])

    def test_windows_without_winreg_answers_empty(self):
        # a missing winreg raises ModuleNotFoundError, which once escaped the
        # old OSError-only catch and turned the dashboard route into a 500
        with mock.patch.object(main.sys, "platform", "win32"):
            self.assertEqual(main.list_voices(), [])


if __name__ == "__main__":
    unittest.main()
