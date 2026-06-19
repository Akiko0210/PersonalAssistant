"""Tests for config loading + defaults (§14)."""

from __future__ import annotations

import textwrap

from voice_notes_agent.config import Config, load_config


def test_defaults_when_missing(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert isinstance(cfg, Config)
    assert cfg.mute.start_muted is True          # privacy default (C7)
    assert cfg.vad.pre_roll_sec == 0.5
    assert cfg.providers.llm.model == "claude-opus-4-8"


def test_partial_override_keeps_other_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        textwrap.dedent(
            """
            vad:
              threshold: 0.7
            providers:
              llm:
                model: claude-sonnet-4-6
            conversation:
              input_device_index: 31
              output_device_index: 30
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.vad.threshold == 0.7
    assert cfg.vad.hangover_sec == 1.0           # untouched default
    assert cfg.providers.llm.model == "claude-sonnet-4-6"
    assert cfg.providers.stt.vendor == "deepgram"  # untouched nested default
    assert cfg.conversation.input_device_index == 31
    assert cfg.conversation.output_device_index == 30


def test_unknown_keys_ignored(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("vad:\n  bogus: 1\n  threshold: 0.3\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.vad.threshold == 0.3
