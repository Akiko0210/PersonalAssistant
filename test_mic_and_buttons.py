"""Experiment: do AirPods stem buttons still fire WHILE the mic is open?

This runs two things at once:
  1. The same SMTC media-button listener as test_media_buttons.py / the agent.
  2. The real microphone capture + Whisper transcription from the agent.

If button presses STILL print while the mic is open and transcribing, then the
"mic forces HFP and kills the buttons" theory is wrong for your setup. If they
print fine standalone (test_media_buttons.py) but go silent here, the theory holds.

    python test_mic_and_buttons.py

Then: talk (watch the transcript), and press your AirPods stem 1x / 2x / 3x at
various times — including WHILE you're talking and right after. Watch whether the
">>> BUTTON" lines appear. Ctrl+C to quit.
"""

import logging
import sys
import threading
import time

import config as cfg
from audio import AudioEngine
from stt import Transcriber
from media_control import MediaButtonListener

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-7s %(levelname)-5s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("test")

# Count button events so we can prove they did / didn't arrive while the mic ran.
counts = {"play_pause": 0, "next": 0, "previous": 0}


def make_cb(name):
    def _cb():
        counts[name] += 1
        print(f"\n>>> BUTTON [{name}]  (total {name}={counts[name]})  "
              f"mic is OPEN right now\n")
    return _cb


def main():
    log.info("loading whisper model (first run downloads weights) ...")
    stt = Transcriber()

    audio = AudioEngine()
    audio.start()
    log.info("microphone stream started  <-- AirPods should now be in HFP mode")

    media = MediaButtonListener(
        on_play_pause=make_cb("play_pause"),
        on_next=make_cb("next"),
        on_previous=make_cb("previous"),
        session_title="Mic+Buttons Test",
    )
    media.start()

    print("\n" + "=" * 64)
    print("TALK to see transcripts. PRESS the stem 1x/2x/3x anytime.")
    print("Watch whether >>> BUTTON lines appear while the mic is open.")
    print("Ctrl+C to stop and see the tally.")
    print("=" * 64 + "\n")

    interrupt = threading.Event()  # unused, but collect_utterance wants the arg
    try:
        while True:
            utt = audio.collect_utterance(interrupt=interrupt,
                                          endpoint_ms=cfg.CONVO_ENDPOINT_MS)
            if utt is None or utt.size == 0:
                continue
            text = stt.transcribe(utt)
            if text:
                print(f"    transcript: {text!r}")
    except KeyboardInterrupt:
        pass
    finally:
        media.stop()
        audio.stop()
        print("\n" + "=" * 64)
        print("RESULT — button events received while the mic was open:")
        for k, v in counts.items():
            print(f"    {k:12s}: {v}")
        total = sum(counts.values())
        if total == 0:
            print("  => ZERO buttons fired with the mic open. Theory holds:")
            print("     the mic (HFP) suppresses the stem media buttons.")
        else:
            print(f"  => {total} button events fired WITH the mic open. Theory is")
            print("     WRONG for your setup — we can keep the button approach.")
        print("=" * 64)


if __name__ == "__main__":
    main()
