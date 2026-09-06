"""The one entry point for headset media-button control.

Per-OS channel code lives beside this file — `windows.py` (SMTC session +
silent keepalive), `macos.py` (event-tap key decoding/suppression),
`linux.py` (MPRIS player on D-Bus) — and this module wires whatever the
running OS provides into the agent's callbacks:

- Every OS: a pynput keyboard hook for media-KEY presses (how wired headsets
  and USB dongles deliver clicks). On macOS it is the only channel, so it
  also carries firmware-decoded Next/Previous there.
- Windows: the SMTC session from `windows.py`; Linux: the MPRIS player from
  `linux.py` — because Bluetooth-native AVRCP presses never appear as key
  events on either (and under Wayland nothing does). A press arriving on
  both channels within MEDIA_CLICK_DEDUPE_S is counted once by the gesture
  decoder; the media session alone carries firmware-decoded Next/Previous.

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
        (SMTC/MPRIS Next/Previous, media keys on macOS). `speaking` is
        polled only for the diagnostic log line — it is the way to tell, from
        the log, whether a press landed mid-reply."""
        self._on_click = on_click
        self._on_note = on_note
        self._on_quit = on_quit
        self._speaking = speaking
        self._listener = None
        self._smtc = None
        self._mpris = None

    def start(self):
        if sys.platform == "win32":
            self._start_keyboard_hook()
            self._smtc = self._start_session(self._make_smtc, "SMTC")
        elif sys.platform.startswith("linux"):
            # The MPRIS player first: it is the channel that works everywhere
            # (Wayland included), and the hook can't even be constructed
            # without an X display — that must not take the buttons down.
            self._mpris = self._start_session(self._make_mpris, "MPRIS")
            try:
                self._start_keyboard_hook()
            except Exception as e:  # noqa: BLE001 - no X display (Wayland)
                log.warning("keyboard hook unavailable (%s); MPRIS is the "
                            "only button channel", e)
        else:
            log.info("no media session on this OS; using the keyboard hook")
            self._start_keyboard_hook()

    def duck(self, seconds: float = 1.0):
        """Briefly pause the Windows keepalive so a state-tracking dongle sees
        its 'pause' honoured (see windows.py); nothing to do elsewhere."""
        if self._smtc is not None:
            self._smtc.duck(seconds)

    def stop(self):
        for channel in (self._smtc, self._mpris, self._listener):
            if channel is not None:
                channel.stop()

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

    def _start_session(self, make, label):
        """The OS media session (SMTC / MPRIS) — the channel Bluetooth-native
        headset presses arrive on. Fails soft: the keyboard hook stays."""
        def on_play_pause():
            log.info("media button (%s) received (speaking=%s)",
                     label, self._speaking())
            self._on_click()

        try:
            session = make(on_play_pause)
            session.start()
            return session
        except Exception as e:  # noqa: BLE001 - winrt/dbus_fast missing, bus down
            log.warning(
                "%s media session unavailable (%s); Bluetooth-native headset "
                "buttons won't be received (keyboard hook still active)",
                label, e
            )
            return None

    def _make_smtc(self, on_play_pause):
        from media_control.windows import MediaButtonListener
        return MediaButtonListener(
            on_play_pause=on_play_pause,
            on_next=self._on_note,
            on_previous=self._on_quit,
            # Short debounce: real double-clicks arrive ~200 ms apart and
            # must get through; cross-channel dedupe lives in the gesture
            # decoder.
            debounce_s=0.08,
            keepalive=cfg.MEDIA_KEEPALIVE,
        )

    def _make_mpris(self, on_play_pause):
        from media_control.linux import MediaButtonListener
        return MediaButtonListener(
            on_play_pause=on_play_pause,
            on_next=self._on_note,
            on_previous=self._on_quit,
        )
