"""Headset / Bluetooth media-button control via Windows SMTC.

Registers our own System Media Transport Controls (SMTC) session so Windows
routes hardware media-button presses to this process as explicit events:

    Play / Pause   <- single press
    Next           <- double press   (how AirPods deliver a double-press)
    Previous       <- triple press   (how AirPods deliver a triple-press)

This is PC-free: it works from across the room over Bluetooth. A silent looping
audio keepalive makes us the active media session so the buttons reach us.
"""

import logging
import struct
import tempfile
import time
import wave

try:  # modern, prebuilt wheels (incl. Python 3.13)
    from winrt.windows.media.playback import MediaPlayer
    from winrt.windows.media import (
        SystemMediaTransportControlsButton as Button,
        MediaPlaybackStatus,
        MediaPlaybackType,
    )
    from winrt.windows.media.core import MediaSource
    from winrt.windows.foundation import Uri
except ImportError:  # legacy package, same API
    from winsdk.windows.media.playback import MediaPlayer
    from winsdk.windows.media import (
        SystemMediaTransportControlsButton as Button,
        MediaPlaybackStatus,
        MediaPlaybackType,
    )
    from winsdk.windows.media.core import MediaSource
    from winsdk.windows.foundation import Uri

log = logging.getLogger("media")


class MediaButtonListener:
    """Owns an SMTC session and dispatches button presses to callbacks.

    Callbacks fire on a WinRT thread-pool thread, so they must be thread-safe
    (e.g. push onto a queue rather than doing heavy work inline).
    """

    def __init__(self, on_play_pause=None, on_next=None, on_previous=None,
                 keepalive=True, session_title="Voice Agent", debounce_s=0.25):
        self._cb = {
            Button.PLAY: on_play_pause,
            Button.PAUSE: on_play_pause,
            Button.NEXT: on_next,
            Button.PREVIOUS: on_previous,
        }
        self._keepalive = keepalive
        self._title = session_title
        self._debounce_s = debounce_s
        self._last = {}
        self._player = None
        self._smtc = None

    def start(self):
        self._player = MediaPlayer()
        # Disable automatic command handling so the legacy button_pressed fires.
        try:
            self._player.command_manager.is_enabled = False
        except Exception as e:  # noqa: BLE001
            log.warning("could not disable command_manager: %s", e)

        smtc = self._player.system_media_transport_controls
        smtc.is_enabled = True   # REQUIRED — without this, no button events fire
        smtc.is_play_enabled = True
        smtc.is_pause_enabled = True
        smtc.is_next_enabled = True
        smtc.is_previous_enabled = True

        updater = smtc.display_updater
        updater.type = MediaPlaybackType.MUSIC
        updater.music_properties.title = self._title
        updater.update()

        smtc.playback_status = MediaPlaybackStatus.PLAYING
        smtc.add_button_pressed(self._on_button)
        self._smtc = smtc

        if self._keepalive:
            self._start_keepalive()
        log.info("media button listener active (session '%s')", self._title)

    def _on_button(self, sender, args):
        button = args.button
        cb = self._cb.get(button)
        if cb is None:
            return
        now = time.monotonic()
        if now - self._last.get(button, 0) < self._debounce_s:
            return
        self._last[button] = now
        log.info("media button: %s", button)
        try:
            cb()
        except Exception:  # noqa: BLE001
            log.exception("media button callback failed")

    def _start_keepalive(self):
        """Loop a short silent WAV so this process owns the active session."""
        path = tempfile.mktemp(suffix=".wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(struct.pack("<" + "h" * 8000, *([0] * 8000)))  # 1s silence
        self._player.source = MediaSource.create_from_uri(
            Uri("file:///" + path.replace("\\", "/"))
        )
        self._player.is_looping_enabled = True
        self._player.play()

    def stop(self):
        try:
            if self._player is not None:
                self._player.pause()
        except Exception:  # noqa: BLE001
            pass
