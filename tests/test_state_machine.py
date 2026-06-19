"""Tests for the three-state machine and its guards (§4)."""

from __future__ import annotations

import pytest

from voice_notes_agent.state import InvalidTransition, State, StateMachine


def test_starts_muted():
    sm = StateMachine()
    assert sm.state is State.MUTED


def test_legal_cycle():
    sm = StateMachine()
    sm.unmute()
    assert sm.state is State.LISTENING
    sm.start_capture()
    assert sm.state is State.CAPTURING
    sm.stop_capture()
    assert sm.state is State.LISTENING
    sm.mute()
    assert sm.state is State.MUTED


def test_illegal_jump_muted_to_capturing():
    sm = StateMachine()
    with pytest.raises(InvalidTransition):
        sm.transition(State.CAPTURING, "bad")


def test_mute_from_capturing_runs_autosave_guard():
    sm = StateMachine()
    calls = []
    sm.add_guard(State.CAPTURING, State.MUTED, lambda tr: calls.append(tr.trigger))
    sm.unmute()
    sm.start_capture()
    sm.mute()
    assert sm.state is State.MUTED
    assert calls == ["mute"]


def test_enter_exit_hooks_fire_in_order():
    sm = StateMachine()
    events = []
    sm.on_exit(State.MUTED, lambda tr: events.append("exit_muted"))
    sm.on_enter(State.LISTENING, lambda tr: events.append("enter_listening"))
    sm.on_transition(lambda tr: events.append("transition"))
    sm.unmute()
    assert events == ["exit_muted", "enter_listening", "transition"]


def test_self_transition_is_noop():
    sm = StateMachine()
    fired = []
    sm.on_enter(State.MUTED, lambda tr: fired.append(1))
    sm.transition(State.MUTED, "again")
    assert fired == []


def test_guard_can_abort_transition():
    sm = StateMachine()

    def deny(tr):
        raise RuntimeError("nope")

    sm.add_guard(State.MUTED, State.LISTENING, deny)
    with pytest.raises(RuntimeError):
        sm.unmute()
    assert sm.state is State.MUTED  # transition rolled back
