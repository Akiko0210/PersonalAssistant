"""Tests for Pipecat conversation integration helpers."""

from __future__ import annotations

from voice_notes_agent.agent.conversation import _pipeline_runner_kwargs


class RunnerWithSignalHandling:
    def __init__(self, *, handle_sigint: bool = True, name: str | None = None):
        self.handle_sigint = handle_sigint
        self.name = name


class LegacyRunner:
    def __init__(self):
        pass


def test_pipeline_runner_disables_sigint_when_supported():
    assert _pipeline_runner_kwargs(RunnerWithSignalHandling) == {"handle_sigint": False}


def test_pipeline_runner_does_not_pass_unknown_kwargs():
    assert _pipeline_runner_kwargs(LegacyRunner) == {}
