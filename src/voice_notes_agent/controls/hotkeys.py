"""Hands-free controls: headset buttons (primary) + global hotkeys (backup) (§5.8, §11).

This is a screen-free, hands-free app, so the **earphone/headset button is the primary
control surface** (FR-K2). Earbud firmware already maps tap gestures to distinct media
keys, so we bind those directly — no tap-counting needed and it works regardless of
window focus (FR-K1):

    single tap (play/pause)  -> toggle MUTED <-> LISTENING   (the most-used action)
    double tap (next track)  -> start / stop note capture
    triple tap (prev track)  -> hard mute (true mute)

Long-press is intentionally unused: most headsets hijack a long hold for the OS voice
assistant, so it cannot be relied on (§R-5).

Global keyboard hotkeys remain as a **secondary/backup** surface for desk use. The mute
key still distinguishes short vs long press (short -> wake-word sleep, long -> true mute;
§11, Open Decision O-2).

The ``keyboard`` library is Windows-focused and imported lazily, so the package imports
on any OS. Callbacks are plain zero-arg functions wired up by the app.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

Action = Callable[[], None]


@dataclass
class HotkeyActions:
    mute_short: Action          # short press: wake-word sleep (or configured)
    mute_long: Action           # long press: true mute (or configured)
    notes_toggle: Action        # start if listening / stop if capturing
    push_to_talk_down: Action
    push_to_talk_up: Action


class HotkeyManager:
    """Binds headset media buttons (primary) and keyboard hotkeys (backup)."""

    # Media-key name candidates for the `keyboard` library (varies by build).
    _PLAY_PAUSE = ("play/pause media", "play/pause")
    _NEXT = ("next track", "media next")
    _PREV = ("previous track", "media previous")

    def __init__(self, cfg, actions: HotkeyActions) -> None:
        self._cfg = cfg
        self._actions = actions
        self._registered = False
        self._mute_press_t0: float | None = None

    def start(self) -> None:  # pragma: no cover - requires keyboard + OS focus
        try:
            import keyboard
        except Exception as exc:
            log.warning("global key hooks unavailable (%s); voice control still works", exc)
            return

        self._bind_headset(keyboard)   # primary, hands-free
        self._bind_keyboard(keyboard)  # secondary, backup

        self._registered = True
        log.info("controls registered (headset primary, keyboard backup)")

    # ── primary: headset media buttons ───────────────────────────────────────
    def _bind_headset(self, keyboard) -> None:  # pragma: no cover - needs OS
        hs = getattr(self._cfg, "headset", None)
        if hs is not None and not hs.enabled:
            return
        single = hs.single_tap if hs else "mute_toggle"
        double = hs.double_tap if hs else "notes_toggle"
        triple = hs.triple_tap if hs else "true_mute"

        self._bind_first(keyboard, self._PLAY_PAUSE, self._resolve(single), "single-tap")
        self._bind_first(keyboard, self._NEXT, self._resolve(double), "double-tap")
        self._bind_first(keyboard, self._PREV, self._resolve(triple), "triple-tap")

    def _bind_first(self, keyboard, names, action: Action | None, label: str) -> None:
        if action is None:
            return
        for key in names:
            try:
                keyboard.add_hotkey(key, action)
                log.info("headset %s -> %s bound", label, key)
                return
            except Exception:
                continue
        log.warning("could not bind headset %s (no media key available)", label)

    def _resolve(self, name: str) -> Action | None:
        return {
            "mute_toggle": self._actions.mute_short,
            "true_mute": self._actions.mute_long,
            "notes_toggle": self._actions.notes_toggle,
            "none": (lambda: None),
        }.get(name)

    # ── secondary: keyboard hotkeys ──────────────────────────────────────────
    def _bind_keyboard(self, keyboard) -> None:  # pragma: no cover - needs OS
        hk = self._cfg.hotkeys
        if getattr(hk, "enabled", True) is False:
            return
        long_ms = self._cfg.mute.long_press_ms

        # Mute key: distinguish short vs long press by tracking key down/up timing.
        def _mute_down(_e=None):
            if self._mute_press_t0 is None:
                self._mute_press_t0 = time.monotonic()

        def _mute_up(_e=None):
            t0 = self._mute_press_t0
            self._mute_press_t0 = None
            if t0 is None:
                return
            held_ms = (time.monotonic() - t0) * 1000.0
            if held_ms >= long_ms:
                self._actions.mute_long()
            else:
                self._actions.mute_short()

        keyboard.on_press_key(_last_key(hk.mute_toggle), _mute_down, suppress=False)
        keyboard.on_release_key(_last_key(hk.mute_toggle), _mute_up, suppress=False)
        keyboard.add_hotkey(hk.notes_toggle, self._actions.notes_toggle)

        # Push-to-talk: hold to talk.
        keyboard.on_press_key(_last_key(hk.push_to_talk), lambda _e: self._actions.push_to_talk_down())
        keyboard.on_release_key(_last_key(hk.push_to_talk), lambda _e: self._actions.push_to_talk_up())

    def stop(self) -> None:  # pragma: no cover - requires keyboard
        if not self._registered:
            return
        try:
            import keyboard

            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        self._registered = False


def _last_key(combo: str) -> str:
    """``keyboard.on_press_key`` wants a single key; take the last token of a combo."""
    return combo.split("+")[-1].strip()
