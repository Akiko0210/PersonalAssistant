"""The one entry point for headset media-button control.

Per-OS channel code lives beside this file — `windows.py` (SMTC session +
silent keepalive), `macos.py` (event-tap key decoding/suppression),
`linux.py` (status notes) — and this module wires whatever the running OS
provides into the agent's callbacks:

- Every OS: a pynput keyboard hook for media-KEY presses (how wired headsets
  and USB dongles deliver clicks). On macOS it is the only channel, so it
  also carries firmware-decoded Next/Previous there.
- Windows only: the SMTC session from `windows.py`, because Bluetooth-native
  AVRCP presses never appear as key events. A press arriving on both channels
  within MEDIA_CLICK_DEDUPE_S is counted once by the gesture decoder.

The agent talks to a single MediaButtons object: start()/stop() bracket the
lifetime, duck() forwards the Yealink keepalive workaround (a no-op on any OS
without a keepalive to pause). See docs/MEDIA_CONTROL.md for the full story.
"""

import logging
import sys

import config as cfg

log = logging.getLogger("media")


class MediaButtons:
    """Every button channel this OS provides, behind one start/stop/duck."""

    def __init__(self, on_click, on_note, on_quit, speaking=lambda: False):
        """`on_click` takes each raw single press (feeds the gesture decoder);
        `on_note`/`on_quit` take firmware-decoded double/triple presses
        (SMTC Next/Previous on Windows, media keys on macOS). `speaking` is
        polled only for the diagnostic log line — it is the way to tell, from
        the log, whether a press landed mid-reply."""
        self._on_click = on_click
        self._on_note = on_note
        self._on_quit = on_quit
        self._speaking = speaking
        self._listener = None
        self._smtc = None

    def start(self):
        self._start_keyboard_hook()
        if sys.platform == "win32":
            self._start_smtc()
        else:
            log.info("SMTC media session is Windows-only; "
                     "using the keyboard hook")

    def duck(self, seconds: float = 1.0):
        """Briefly pause the Windows keepalive so a state-tracking dongle sees
        its 'pause' honoured (see windows.py); nothing to do elsewhere."""
        if self._smtc is not None:
            self._smtc.duck(seconds)

    def stop(self):
        if self._smtc is not None:
            self._smtc.stop()
        if self._listener is not None:
            self._listener.stop()

    # --- channels ------------------------------------------------------------
    def _start_keyboard_hook(self):
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.media_play_pause:
                log.info("media key received (speaking=%s)", self._speaking())
                self._on_click()
            elif sys.platform == "darwin" and key == keyboard.Key.media_next:
                # No SMTC channel on macOS, so firmware-decoded double/triple
                # presses (AirPods-class) must be caught here, mapped the same
                # way windows.py maps SMTC Next/Previous.
                log.info("media key Next received")
                self._on_note()
            elif sys.platform == "darwin" and key == keyboard.Key.media_previous:
                log.info("media key Previous received")
                self._on_quit()

        # On macOS the tap must also SWALLOW these keys, or every press
        # reaches the system handler too and launches/controls Music.app.
        # Ignored by pynput's other backends (backend-prefixed kwarg).
        intercept = None
        if sys.platform == "darwin":
            from media_control.macos import darwin_intercept
            intercept = darwin_intercept

        self._listener = keyboard.Listener(on_press=on_press,
                                           darwin_intercept=intercept)
        self._listener.start()

    def _start_smtc(self):
        try:
            from media_control.windows import MediaButtonListener

            def on_play_pause():
                log.info("media button (SMTC) received (speaking=%s)",
                         self._speaking())
                self._on_click()

            self._smtc = MediaButtonListener(
                on_play_pause=on_play_pause,
                on_next=self._on_note,
                on_previous=self._on_quit,
                # Short debounce: real double-clicks arrive ~200 ms apart and
                # must get through; cross-channel dedupe lives in the gesture
                # decoder.
                debounce_s=0.08,
                keepalive=cfg.MEDIA_KEEPALIVE,
            )
            self._smtc.start()
        except Exception as e:  # noqa: BLE001 - any winrt/SMTC failure
            log.warning(
                "SMTC media session unavailable (%s); Bluetooth-native headset "
                "buttons won't be received (keyboard hook still active)", e
            )
