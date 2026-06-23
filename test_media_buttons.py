"""Standalone test: capture Bluetooth headset media buttons on Windows.

Instead of *observing* the current media session (which can only see side
effects like a track change), this registers our OWN media session via SMTC
(System Media Transport Controls). Windows then routes the headset's media
button presses to us as distinct, explicit events:

    AirPods 1 press  -> Play or Pause
    AirPods 2 press  -> Next
    AirPods 3 press  -> Previous

Run it, then squeeze your AirPods stem 1x / 2x / 3x and watch what prints.

    pip install winrt-runtime "winrt-Windows.Media" "winrt-Windows.Media.Playback" "winrt-Windows.Foundation"
    python test_media_buttons.py

(For the optional silent keepalive also: pip install "winrt-Windows.Media.Core")

If NOTHING prints when you press the button, another app (Spotify, a browser
playing audio, etc.) probably owns the active media session. Pause/close that
app, or set KEEPALIVE_AUDIO = True below to make this script hold the session
by quietly looping a silent sound.
"""

import time

try:  # modern, prebuilt wheels (incl. Python 3.13)
    from winrt.windows.media.playback import MediaPlayer
    from winrt.windows.media import (
        SystemMediaTransportControlsButton as Button,
        MediaPlaybackStatus,
        MediaPlaybackType,
    )
except ImportError:  # legacy package, same API
    from winsdk.windows.media.playback import MediaPlayer
    from winsdk.windows.media import (
        SystemMediaTransportControlsButton as Button,
        MediaPlaybackStatus,
        MediaPlaybackType,
    )

KEEPALIVE_AUDIO = True  # set True if button presses aren't being captured


def main():
    player = MediaPlayer()

    # Disable the automatic command manager so the legacy button_pressed event
    # fires and we can handle the buttons ourselves.
    try:
        player.command_manager.is_enabled = False
    except Exception as e:  # noqa: BLE001
        print("warn: could not disable command_manager:", e)

    smtc = player.system_media_transport_controls
    smtc.is_enabled = True   # REQUIRED — without this, button_pressed never fires
    smtc.is_play_enabled = True
    smtc.is_pause_enabled = True
    smtc.is_next_enabled = True
    smtc.is_previous_enabled = True
    smtc.is_stop_enabled = True

    # Give the session visible metadata so Windows lists it as a real media
    # session and routes hardware buttons to it.
    updater = smtc.display_updater
    updater.type = MediaPlaybackType.MUSIC
    updater.music_properties.title = "Voice Agent"
    updater.update()

    # Claim the "playing" state so Windows treats us as the active session.
    smtc.playback_status = MediaPlaybackStatus.PLAYING

    labels = {
        Button.PLAY: "PLAY        (single press)",
        Button.PAUSE: "PAUSE       (single press)",
        Button.NEXT: "NEXT        (double press)",
        Button.PREVIOUS: "PREVIOUS    (triple press)",
        Button.STOP: "STOP",
    }

    def on_button(sender, args):
        print(">>> button:", labels.get(args.button, str(args.button)))

    smtc.add_button_pressed(on_button)

    if KEEPALIVE_AUDIO:
        _start_silent_keepalive(player)

    print("Listening for headset media buttons.")
    print("Squeeze your AirPods stem: 1x, then 2x, then 3x.")
    print("Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting.")


def _start_silent_keepalive(player):
    """Loop a short silent WAV so this process owns the active media session."""
    import struct
    import tempfile
    import wave
    try:
        from winrt.windows.media.core import MediaSource
        from winrt.windows.foundation import Uri
    except ImportError:
        from winsdk.windows.media.core import MediaSource
        from winsdk.windows.foundation import Uri

    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(struct.pack("<" + "h" * 8000, *([0] * 8000)))  # 1s silence

    player.source = MediaSource.create_from_uri(Uri("file:///" + path.replace("\\", "/")))
    player.is_looping_enabled = True
    player.play()
    print("(silent keepalive playing)")


if __name__ == "__main__":
    main()
