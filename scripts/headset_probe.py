"""Linux headset probe: does a button press reach this PC, and on which channel?

Run it from the project root with the agent STOPPED (the two would fight over
the MPRIS name), then click the headset a few times:

    .venv/bin/python scripts/headset_probe.py           # buttons, 60 s
    .venv/bin/python scripts/headset_probe.py --tts     # which voice, and say a line

Every press is reported on up to three channels, each a separate stage of the
journey — so the first one that stays silent names the broken link:

  1. kernel   the raw input device (/dev/input/event*). BlueZ turns an AVRCP
              press into a key here. Silent = the headset isn't sending, or
              is paired as audio only. Needs read access to /dev/input:
              run with sudo once, or add yourself to the `input` group.
  2. MPRIS    the agent's own D-Bus player (media_control/linux.py). The
              desktop's media-key daemon forwards kernel presses here. Kernel
              yes / MPRIS no = the desktop isn't forwarding (no daemon, or it
              picked another player — close music apps, or run `playerctld`).
  3. keyboard the pynput hook — X11 only, wired headsets mostly.

--tts prints which speech backend the agent gets and why, then speaks.
"""

import argparse
import logging
import os
import selectors
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Linux input_event key codes (linux/input-event-codes.h) a headset can send.
_KEYS = {164: "PLAYPAUSE", 200: "PLAYCD", 201: "PAUSECD", 207: "PLAY",
         119: "PAUSE", 163: "NEXTSONG", 165: "PREVIOUSSONG", 166: "STOPCD"}
_EV_KEY = 1
_EVENT = struct.Struct("llHHi")  # tv_sec, tv_usec, type, code, value


def say(channel, what):
    print(f"{time.strftime('%H:%M:%S')}  [{channel:8}] {what}", flush=True)


# --- 1. kernel input devices ---------------------------------------------------
def input_devices():
    """{event node: device name} from /proc/bus/input/devices."""
    devices, name = {}, ""
    for line in Path("/proc/bus/input/devices").read_text().splitlines():
        if line.startswith("N: Name="):
            name = line.split("=", 1)[1].strip('"')
        elif line.startswith("H: Handlers="):
            for h in line.split("=", 1)[1].split():
                if h.startswith("event"):
                    devices[h] = name
    return devices


def open_kernel_devices(sel):
    devices = input_devices()
    if not devices:
        say("kernel", "no input devices listed in /proc/bus/input/devices")
    opened = 0
    for node, name in devices.items():
        try:
            f = open(f"/dev/input/{node}", "rb", buffering=0)
        except PermissionError:
            continue
        except OSError as e:
            say("kernel", f"{node} ({name}): {e}")
            continue
        os.set_blocking(f.fileno(), False)
        sel.register(f, selectors.EVENT_READ, (node, name))
        opened += 1
    say("kernel", f"watching {opened} of {len(devices)} input devices")
    if opened < len(devices):
        say("kernel", "some devices unreadable: run with sudo, or add "
                      "yourself to the `input` group and log in again")
    for node, name in devices.items():
        if any(w in name.lower() for w in ("avrcp", "bluetooth", "headset", "consumer")):
            say("kernel", f"looks like a headset: {node} = {name}")


def pump_kernel(sel):
    for key, _ in sel.select(timeout=0.2):
        node, name = key.data
        try:
            data = key.fileobj.read(_EVENT.size * 64)
        except OSError:
            continue
        for off in range(0, len(data or b"") - _EVENT.size + 1, _EVENT.size):
            _s, _us, etype, code, value = _EVENT.unpack_from(data, off)
            if etype == _EV_KEY and value == 1 and code in _KEYS:
                say("kernel", f"{_KEYS[code]} from {node} ({name})")


# --- 2. MPRIS player -----------------------------------------------------------
def start_mpris():
    try:
        from media_control.linux import MediaButtonListener
    except ImportError as e:
        say("MPRIS", f"not available: {e} — run: .venv/bin/pip install -r requirements.txt")
        return None
    try:
        listener = MediaButtonListener(
            on_play_pause=lambda: say("MPRIS", "click (Play/Pause/PlayPause)"),
            on_next=lambda: say("MPRIS", "Next (= double press)"),
            on_previous=lambda: say("MPRIS", "Previous (= triple press)"))
        listener.start()
        say("MPRIS", "player registered on the session bus")
        return listener
    except Exception as e:  # noqa: BLE001 - report, don't die
        say("MPRIS", f"could not register: {e}")
        return None


# --- 3. pynput keyboard hook ---------------------------------------------------
def start_hook():
    try:
        from pynput import keyboard

        def on_press(key):
            if key in (keyboard.Key.media_play_pause, keyboard.Key.media_next,
                       keyboard.Key.media_previous):
                say("keyboard", str(key))

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        say("keyboard", "hook active (X11 only; deaf under Wayland)")
        return listener
    except Exception as e:  # noqa: BLE001
        say("keyboard", f"hook unavailable: {e}")
        return None


def probe_buttons(seconds):
    say("session", f"type={os.environ.get('XDG_SESSION_TYPE', '?')} "
                   f"desktop={os.environ.get('XDG_CURRENT_DESKTOP', '?')} "
                   f"bus={'yes' if os.environ.get('DBUS_SESSION_BUS_ADDRESS') else 'NO'}")
    sel = selectors.DefaultSelector()
    open_kernel_devices(sel)
    mpris = start_mpris()
    hook = start_hook()
    print(f"\nClick the headset button now (single, double, triple). "
          f"Watching for {seconds} s, Ctrl-C to stop.\n", flush=True)
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            pump_kernel(sel)
    except KeyboardInterrupt:
        pass
    for ch in (mpris, hook):
        if ch is not None:
            ch.stop()


# --- TTS -----------------------------------------------------------------------
def probe_tts():
    import config as cfg
    for mod in ("piper", "sounddevice", "dbus_fast", "speechd", "pyttsx3", "pynput"):
        try:
            __import__(mod)
            say("deps", f"{mod}: ok")
        except ImportError as e:
            say("deps", f"{mod}: MISSING ({e})")
    say("voices", f"{cfg.PIPER_VOICE_DIR}: "
                  f"{sorted(p.name for p in Path(cfg.PIPER_VOICE_DIR).glob('*.onnx')) or 'nothing downloaded yet'}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-8s %(levelname)-7s %(message)s")
    from speech.tts.main import Speaker
    speaker = Speaker()
    say("tts", f"backend = {type(speaker._backend).__name__}, voice = {speaker.current_voice() or '(default)'}")
    speaker.speak("This is the voice the agent will use.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tts", action="store_true", help="probe speech instead of buttons")
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()
    probe_tts() if args.tts else probe_buttons(args.seconds)
