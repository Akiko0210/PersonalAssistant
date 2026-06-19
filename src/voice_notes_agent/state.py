"""Three-state machine (§4).

States and the only legal edges between them:

    MUTED  ⇄  LISTENING  ⇄  CAPTURING

* MUTED:     mic stream released (hard mute); cloud pipeline off. Default at launch (§C7).
* LISTENING: mic open; conversation agent running (STT+LLM+TTS, barge-in).
* CAPTURING: mic open but local-only; cloud loop suspended; VAD-gated recorder + Whisper.

The mute toggle drives MUTED ⇄ LISTENING; the note tools drive LISTENING ⇄ CAPTURING.

Guards (§4):
  * Entering MUTED from CAPTURING must first auto-stop and save the active session.
  * Entering CAPTURING cancels any in-flight TTS/LLM response.
  * Every transition emits an audio cue (handled by the app via the on_transition hook).

This module is pure coordination logic — it owns no audio, no models, no I/O. The app
registers callbacks that perform the side effects, so the machine stays unit-testable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class State(str, Enum):
    MUTED = "muted"
    LISTENING = "listening"
    CAPTURING = "capturing"


@dataclass(frozen=True)
class Transition:
    src: State
    dst: State
    trigger: str  # human-readable cause, e.g. "unmute", "take_notes", "stop_notes"


# Allowed edges. Anything not listed raises (defends against illogical jumps such as
# MUTED -> CAPTURING, which would skip the conversation agent entirely).
_ALLOWED: set[tuple[State, State]] = {
    (State.MUTED, State.LISTENING),
    (State.LISTENING, State.MUTED),
    (State.LISTENING, State.CAPTURING),
    (State.CAPTURING, State.LISTENING),
    (State.CAPTURING, State.MUTED),  # mute-while-capturing; guard auto-saves first
}


class InvalidTransition(RuntimeError):
    pass


# Side-effect hooks the app wires up. Each is called while the lock is held, so they
# must be quick / non-blocking (kick off work on another thread if needed).
OnEnter = Callable[[Transition], None]
OnExit = Callable[[Transition], None]
Guard = Callable[[Transition], None]


class StateMachine:
    """Thread-safe state machine for the agent's top-level mode."""

    def __init__(self, initial: State = State.MUTED) -> None:
        self._state = initial
        self._lock = threading.RLock()
        self._on_enter: dict[State, list[OnEnter]] = {s: [] for s in State}
        self._on_exit: dict[State, list[OnExit]] = {s: [] for s in State}
        self._guards: dict[tuple[State, State], list[Guard]] = {}
        self._on_transition: list[OnEnter] = []

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def on_enter(self, state: State, fn: OnEnter) -> None:
        self._on_enter[state].append(fn)

    def on_exit(self, state: State, fn: OnExit) -> None:
        self._on_exit[state].append(fn)

    def on_transition(self, fn: OnEnter) -> None:
        """Fires on every transition — used for the universal audio cue (§4)."""
        self._on_transition.append(fn)

    def add_guard(self, src: State, dst: State, fn: Guard) -> None:
        """Register a guard that runs *before* the edge is committed.

        Used for the §4 guards, e.g. auto-saving an active session before MUTED, and
        cancelling in-flight TTS/LLM before CAPTURING. A guard raising aborts the
        transition.
        """
        self._guards.setdefault((src, dst), []).append(fn)

    def transition(self, dst: State, trigger: str) -> Transition:
        with self._lock:
            src = self._state
            if src == dst:
                # Idempotent no-op; return a self-transition without firing hooks.
                return Transition(src, dst, trigger)
            if (src, dst) not in _ALLOWED:
                raise InvalidTransition(f"{src.value} -> {dst.value} is not allowed")

            tr = Transition(src, dst, trigger)
            for guard in self._guards.get((src, dst), []):
                guard(tr)  # may raise to abort

            for fn in self._on_exit[src]:
                fn(tr)
            self._state = dst
            for fn in self._on_enter[dst]:
                fn(tr)
            for fn in self._on_transition:
                fn(tr)
            return tr

    # Convenience edges named after the user-facing triggers ------------------
    def unmute(self) -> Transition:
        return self.transition(State.LISTENING, "unmute")

    def mute(self) -> Transition:
        """Mute from wherever we are. From CAPTURING the auto-save guard runs first."""
        return self.transition(State.MUTED, "mute")

    def start_capture(self) -> Transition:
        return self.transition(State.CAPTURING, "take_notes")

    def stop_capture(self) -> Transition:
        return self.transition(State.LISTENING, "stop_notes")
