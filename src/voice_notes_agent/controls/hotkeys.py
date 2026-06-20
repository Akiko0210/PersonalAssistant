"""Hands-free controls: headset media buttons + global hotkeys (§5.8, §11).

These are *best-effort* control surfaces. The reliable floor is the terminal-key loop in
``app.py`` (``m``/``n``/``q``); the bindings here add app-wide control when the OS exposes
the events.

**Headset media buttons.** Many Bluetooth headsets map tap gestures to distinct media
keys, so we bind those directly — no tap-counting, works regardless of window focus
(FR-K1):

    single tap (play/pause)  -> toggle MUTED <-> LISTENING   (the most-used action)
    double tap (next track)  -> start / stop note capture
    triple tap (prev track)  -> hard mute (true mute)

Caveat: many **wired** headsets (e.g. Apple EarPods) do **not** emit media keys on Windows
at all — their inline button signals over the mic line, which Windows doesn't decode — so
these bindings simply never fire for them (§R-5). Use the terminal keys / global hotkeys.
Long-press is also intentionally unused: most headsets hijack a long hold for the OS voice
assistant.

Global keyboard hotkeys are the dependable app-wide surface for desk use. The mute key
distinguishes short vs long press (short -> wake-word sleep, long -> true mute;
§11, Open Decision O-2).

The ``keyboard`` library is Windows-focused and imported lazily, so the package imports
on any OS. Callbacks are plain zero-arg functions wired up by the app.
"""

from __future__ import annotations

import logging
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

    def start(self) -> None:  # pragma: no cover - requires keyboard + OS focus
        try:
            import keyboard
        except Exception as exc:
            log.warning("global key hooks unavailable (%s); voice control still works", exc)
            return

        self._bind_headset(keyboard)   # primary, hands-free
        self._bind_keyboard(keyboard)  # secondary, backup

        self._registered = True
        log.info("global hotkeys + headset media-key bindings registered")

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

        # Bind the *whole* chord via add_hotkey. ``keyboard.on_press_key`` only accepts a
        # single key, so feeding it a chord like "ctrl+alt+m" would actually listen to the
        # bare last key ("m") — firing on every ordinary keystroke (and double-firing with
        # the terminal controls, which toggle on bare "m"). add_hotkey honors the full
        # combination, so it fires only on the real chord. This costs the keyboard
        # short-vs-long-press distinction (it needs down/up timing); long-press / true mute
        # remains available on the headset triple-tap.
        try:
            keyboard.add_hotkey(hk.mute_toggle, self._actions.mute_short)
            keyboard.add_hotkey(hk.notes_toggle, self._actions.notes_toggle)
            keyboard.add_hotkey(hk.push_to_talk, self._actions.push_to_talk_down)
            log.info(
                "keyboard hotkeys bound: %s (mute), %s (notes), %s (push-to-talk)",
                hk.mute_toggle,
                hk.notes_toggle,
                hk.push_to_talk,
            )
        except Exception:
            log.warning("could not bind one or more keyboard hotkeys", exc_info=True)

    def stop(self) -> None:  # pragma: no cover - requires keyboard
        if not self._registered:
            return
        try:
            import keyboard

            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        self._registered = False
