"""Tests for the per-OS TTS variant seam (speech/tts/).

Covers the routing (which variant class each platform gets, and the honest
fallback to pyttsx3 when the native one won't construct), the NSSpeaker
contract against a stub synthesizer — no pyobjc needed, same __new__-and-
hand-install trick as agent_fixtures — and the per-OS voice enumeration the
dashboard dropdown delegates to. The busy-while-paused behaviour is the
load-bearing one: _hold_for_gesture's resume path depends on it (see
FakeTTS in test_mute_speech.py, which models the same contract)."""

import unittest
from unittest import mock

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
        self.assertIs(main._select_backend("win32"), windows.SapiSpeaker)
        self.assertIs(main._select_backend("darwin"), macos.NSSpeaker)
        self.assertIs(main._select_backend("linux"), linux.SpeechdSpeaker)

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
        for cls in (windows.SapiSpeaker, macos.NSSpeaker, linux.SpeechdSpeaker):
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

    def test_unsupported_platform_answers_empty(self):
        # linux (and anything unknown): honest [] so the UI shows free text
        with mock.patch.object(main.sys, "platform", "linux"):
            self.assertEqual(main.list_voices(), [])

    def test_windows_without_winreg_answers_empty(self):
        # a missing winreg raises ModuleNotFoundError, which once escaped the
        # old OSError-only catch and turned the dashboard route into a 500
        with mock.patch.object(main.sys, "platform", "win32"):
            self.assertEqual(main.list_voices(), [])


if __name__ == "__main__":
    unittest.main()
