"""Diagnose media-key interference from audio playback.

Media/transport keys (play-pause, vol up/down) can be swallowed by Windows'
media-session routing whenever audio is playing, so they never reach pynput's
keyboard hook. This test loops a sound (the agent's idle cue, via the same
winsound path) and logs every key event, so you can compare click detection with
the sound ON vs OFF.

The listener is a global hook — you do NOT need this window focused for the
media buttons to register.

Controls:
  media play/pause, vol up, vol down, mute : logged + counted (the thing we test)
  a : a normal key, logged as a CONTROL (should ALWAYS register, sound or not)
  p : toggle the looping test sound on/off
  esc : quit and print totals

Suggested run:
  1. With sound OFF, click play/pause ~5 times and press 'a' a couple times.
  2. Press 'p' to start the sound.
  3. With sound ON, click play/pause ~5 times and press 'a' a couple times.
  4. Press 'esc'. Compare: if media presses drop with sound ON but 'a' still
     registers, the audio session is eating the media keys.
"""

import time
from pathlib import Path

from pynput import keyboard

SOUND = Path(__file__).with_name("assets") / "summarizing.wav"

_playing = False
_media_count = 0
_ctrl_count = 0


def _ts():
    return time.strftime("%H:%M:%S")


def start_sound():
    global _playing
    if _playing:
        return
    if not SOUND.is_file():
        print(f"[!] no sound file at {SOUND} — can't test with audio")
        return
    try:
        import winsound
        winsound.PlaySound(
            str(SOUND),
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
        )
        _playing = True
        print(f"[{_ts()}] === SOUND ON  — now click the media buttons ===")
    except Exception as e:  # noqa: BLE001
        print("could not start sound:", e)


def stop_sound():
    global _playing
    if not _playing:
        return
    try:
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception as e:  # noqa: BLE001
        print("could not stop sound:", e)
    _playing = False
    print(f"[{_ts()}] === SOUND OFF — now click the media buttons ===")


MEDIA = {
    keyboard.Key.media_play_pause: "play_pause",
    keyboard.Key.media_volume_up: "vol_up",
    keyboard.Key.media_volume_down: "vol_down",
    keyboard.Key.media_volume_mute: "vol_mute",
}


def on_press(key):
    global _media_count, _ctrl_count
    if key in MEDIA:
        _media_count += 1
        print(f"[{_ts()}] #{_media_count:<3} MEDIA   {MEDIA[key]:11} "
              f"(sound={'ON ' if _playing else 'OFF'})")
        return
    if key == keyboard.KeyCode.from_char('a'):
        _ctrl_count += 1
        print(f"[{_ts()}] #{_ctrl_count:<3} CONTROL 'a'          "
              f"(sound={'ON ' if _playing else 'OFF'})")
        return
    if key == keyboard.KeyCode.from_char('p'):
        stop_sound() if _playing else start_sound()
        return
    if key == keyboard.Key.esc:
        print(f"\nTotals -> media keys: {_media_count}   control 'a': {_ctrl_count}")
        return False  # stops the listener


def on_release(key):
    if key in MEDIA:
        print(f"[{_ts()}]      release {MEDIA[key]:11} "
              f"(sound={'ON ' if _playing else 'OFF'})")


# --- raw low-level hook logging ------------------------------------------------
# pynput hands the Windows KBDLLHOOKSTRUCT to this filter for EVERY keyboard event
# the low-level hook sees, before it decides whether to raise on_press. Logging
# vkCode here is the ground truth: if a button press reaches the keyboard hook at
# all, it shows up as a vk=0x.. line even when on_press never fires. If a press is
# missing here too, Windows routed it around the keyboard hook (media-session /
# HID app-command) and only Raw Input can recover it.
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104

# Media-related virtual-key codes, for readability in the log.
VK_NAMES = {
    0xB3: "MEDIA_PLAY_PAUSE", 0xB2: "MEDIA_STOP",
    0xB0: "MEDIA_NEXT_TRACK", 0xB1: "MEDIA_PREV_TRACK",
    0xAD: "VOLUME_MUTE", 0xAE: "VOLUME_DOWN", 0xAF: "VOLUME_UP",
}


def win32_filter(msg, data):
    if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
        vk = data.vkCode
        name = VK_NAMES.get(vk, "")
        print(f"[{_ts()}]   [hook] vk=0x{vk:02X} {name:17} "
              f"(sound={'ON ' if _playing else 'OFF'})")
    return True  # never suppress — let normal processing continue


if __name__ == "__main__":
    print(__doc__)
    print("Starting with sound OFF. Press 'p' to toggle sound, 'esc' to quit.\n")
    with keyboard.Listener(on_press=on_press, on_release=on_release,
                           win32_event_filter=win32_filter) as listener:
        listener.join()
