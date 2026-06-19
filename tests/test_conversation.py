"""Tests for Pipecat conversation integration helpers."""

from __future__ import annotations

from dataclasses import dataclass

from voice_notes_agent.agent.conversation import (
    _local_audio_transport_kwargs,
    _pipeline_runner_kwargs,
)


class RunnerWithSignalHandling:
    def __init__(self, *, handle_sigint: bool = True, name: str | None = None):
        self.handle_sigint = handle_sigint
        self.name = name


class LegacyRunner:
    def __init__(self):
        pass


class ParamsWithDeviceIndexes:
    model_fields = {"input_device_index": object(), "output_device_index": object()}


class ParamsWithoutDeviceIndexes:
    model_fields = {"audio_in_enabled": object(), "audio_out_enabled": object()}


@dataclass(frozen=True)
class ConversationCfg:
    input_device_index: int | None = None
    output_device_index: int | None = None


def test_pipeline_runner_disables_sigint_when_supported():
    assert _pipeline_runner_kwargs(RunnerWithSignalHandling) == {"handle_sigint": False}


def test_pipeline_runner_does_not_pass_unknown_kwargs():
    assert _pipeline_runner_kwargs(LegacyRunner) == {}


def test_local_audio_transport_uses_configured_device_indexes():
    cfg = ConversationCfg(input_device_index=31, output_device_index=30)
    assert _local_audio_transport_kwargs(ParamsWithDeviceIndexes, cfg) == {
        "input_device_index": 31,
        "output_device_index": 30,
    }


def test_local_audio_transport_skips_unsupported_device_indexes():
    cfg = ConversationCfg(input_device_index=31, output_device_index=30)
    assert _local_audio_transport_kwargs(ParamsWithoutDeviceIndexes, cfg) == {}


def test_local_audio_transport_omits_unconfigured_device_indexes():
    cfg = ConversationCfg()
    assert _local_audio_transport_kwargs(ParamsWithDeviceIndexes, cfg) == {}
