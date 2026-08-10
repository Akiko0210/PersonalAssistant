"""Headset-button gesture decoding: raw clicks -> single/double/triple.

One physical press can surface more than once — as a duplicate event, or once
per listening channel (keyboard hook AND SMTC; see MEDIA_CONTROL.md) — so any
click inside the dedupe window counts once. Accepted clicks are then counted
within a resolution window; when it closes, the gesture (1, 2, or 3+ clicks)
is delivered to `on_gesture` on a timer thread.

Extracted from Agent._media_click so the timing/threading state machine is
self-contained and testable without hardware.
"""

import threading
import time

import config as cfg


class ClickGestureDecoder:
    """Thread-safe. Call `click()` from any listener channel on every raw press.

    on_press:   called immediately for every *accepted* (deduped) click — the
                agent uses this to go silent at once, before the gesture
                resolves (the dongle swallows follow-up clicks if playback
                runs on; see MEDIA_CONTROL.md).
    on_gesture: called with the final click count (1, 2, 3, ...) once the
                window closes. Runs on a daemon timer thread.
    """

    def __init__(self, on_gesture, on_press=None, *,
                 dedupe_s=None, window_s=0.45):
        self._on_gesture = on_gesture
        self._on_press = on_press
        self._dedupe_s = cfg.MEDIA_CLICK_DEDUPE_S if dedupe_s is None else dedupe_s
        self._window_s = window_s
        self._lock = threading.Lock()
        self._count = 0
        self._timer = None
        self._last_click = 0.0
        # Generation counter: cancel() can't stop a Timer whose callback has
        # already fired but not yet taken the lock, so each armed window gets a
        # generation and a stale callback recognises itself and does nothing.
        # Without this, a click landing in that gap was counted into the
        # expired gesture (one stray click read as a double), and _resolve then
        # orphaned the freshly armed timer by clearing self._timer.
        self._gen = 0

    def click(self):
        now = time.monotonic()
        with self._lock:
            # Cross-channel / duplicate-event dedupe. Real double-clicks arrive
            # further apart than the dedupe window.
            if now - self._last_click < self._dedupe_s:
                return
            self._last_click = now
            self._count += 1
            if self._timer:
                self._timer.cancel()
            self._gen += 1
            self._timer = threading.Timer(self._window_s, self._resolve,
                                          args=(self._gen,))
            self._timer.daemon = True
            self._timer.start()
        if self._on_press:
            self._on_press()

    def _resolve(self, gen):
        with self._lock:
            if gen != self._gen:
                return  # a newer click re-armed the window; that timer decides
            count = self._count
            self._count = 0
            self._timer = None
        if count:
            self._on_gesture(count)

    def stop(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            self._gen += 1  # invalidate any already-fired, not-yet-run callback
            self._count = 0
