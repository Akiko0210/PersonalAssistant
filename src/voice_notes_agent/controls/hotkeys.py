"""Global hotkeys + headset media button (§5.8, §11).

Global hotkeys are the reliable control surface (§R-5): each primary action has a
distinct hotkey that works regardless of window focus (FR-K1). The headset media button
is bound to the single most-used action — mute toggle by default (FR-K2) — and multi-tap
gestures are explicitly *not* required for any core action (FR-K4).

Mute uses short-press vs long-press to pick between wake-word sleep and true mute
(§11, Open Decision O-2): a short press → wake-word sleep, a long press → true mute.

The ``keyboard`` library is Windows-focused and imported lazily, so the package imports
on any OS. Callbacks are plain zero-arg functions wired up by the app.
"""

from __future__ import annotations

import logging
import threading
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
    """Registers global hotkeys and translates press duration into short/long intents."""

    def __init__(self, cfg, actions: HotkeyActions) -> None:
        self._cfg = cfg
        self._actions = actions
        self._registered = False
        self._mute_press_t0: float | None = None

    def start(self) -> None:  # pragma: no cover - requires keyboard + OS focus
        try:
            import keyboard
        except Exception as exc:
            log.warning("global hotkeys unavailable (%s); voice/button control still works", exc)
            return

        hk = self._cfg.hotkeys
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

        # Headset media button -> configured action (default mute toggle, FR-K2).
        media_action = self._media_action()
        for key in ("play/pause media", "play/pause"):
            try:
                keyboard.add_hotkey(key, media_action)
                break
            except Exception:
                continue

        self._registered = True
        log.info("global hotkeys registered")

    def _media_action(self) -> Action:
        name = self._cfg.hotkeys.media_button_action
        if name == "mute_toggle":
            # Map the single button press to a short mute toggle.
            return self._actions.mute_short
        if name == "notes_toggle":
            return self._actions.notes_toggle
        return self._actions.mute_short

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
