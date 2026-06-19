"""Typed configuration loaded from ``config.yaml`` (§14).

The YAML is parsed into frozen dataclasses so the rest of the code reads attributes
(``cfg.vad.threshold``) rather than dict keys. Unknown keys are ignored and missing
keys fall back to the documented defaults, so a partial user config is always valid.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass(frozen=True)
class VadConfig:
    threshold: float = 0.5
    pre_roll_sec: float = 0.5
    hangover_sec: float = 1.0
    tail_pad_sec: float = 0.2
    min_segment_sec: float = 0.4
    energy_floor: float = 0.0


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    input_device: Any = None
    output_device: Any = None
    block_ms: int = 32


@dataclass(frozen=True)
class WhisperConfig:
    model: str = "large-v3-turbo"
    fallback_model: str = "medium"
    compute_type: str = "int8"
    language: str | None = "en"
    beam_size: int = 1
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0


@dataclass(frozen=True)
class ProviderRef:
    vendor: str = ""
    model: str = ""
    voice: str = ""


@dataclass(frozen=True)
class ProvidersConfig:
    llm: ProviderRef = field(default_factory=lambda: ProviderRef("anthropic", "claude-opus-4-8"))
    stt: ProviderRef = field(default_factory=lambda: ProviderRef("deepgram", "nova-2"))
    tts: ProviderRef = field(default_factory=lambda: ProviderRef("deepgram", voice="aura-asteria-en"))
    embeddings: ProviderRef = field(default_factory=lambda: ProviderRef("local", "all-MiniLM-L6-v2"))


@dataclass(frozen=True)
class SummaryConfig:
    spoken_max_sentences: int = 3
    min_speech_sec: float = 5.0
    full_template: str = (
        "# {title}\n\n**Headline:** {headline}\n\n"
        "## Key points\n{key_points}\n\n## Action items\n{action_items}\n\n"
        "## Open questions\n{questions}\n"
    )


@dataclass(frozen=True)
class ConversationConfig:
    bargein_min_ms: int = 250
    turn_gap_ms: int = 300
    system_prompt: str = (
        "You are a hands-free, voice-only personal notes assistant. When asked to take "
        "notes, call start_note_session and then stay silent — capture is handled locally "
        "until stopped. When a session ends, call summarize_session and read back the "
        "spoken summary. To answer questions about past notes, call search_notes. Keep "
        "spoken responses concise; offer to read more detail on request."
    )


@dataclass(frozen=True)
class MuteConfig:
    start_muted: bool = True
    default_model: str = "both"
    wake_word: str = "jarvis"
    short_press: str = "wake_word_sleep"
    long_press: str = "true_mute"
    long_press_ms: int = 600


@dataclass(frozen=True)
class WakeWordsConfig:
    stop_capture: str = "stop notes"
    enabled: bool = True


@dataclass(frozen=True)
class HeadsetConfig:
    """Primary hands-free control surface: earphone/headset buttons (§FR-K2).

    Earbud firmware already maps tap gestures to distinct media keys, so we bind
    those directly — no tap-counting, and it works regardless of screen/focus.
    Long-press is intentionally unused: most headsets hijack it for the OS voice
    assistant, so it can't be relied on (§R-5).

    Action names: ``mute_toggle`` (toggle MUTED<->LISTENING), ``true_mute``
    (hard mute until a manual wake), ``notes_toggle`` (start/stop note capture),
    or ``none``.
    """

    enabled: bool = True
    single_tap: str = "mute_toggle"   # play/pause  -> toggle listening (primary)
    double_tap: str = "notes_toggle"  # next track  -> start/stop notes
    triple_tap: str = "true_mute"     # prev track  -> hard mute


@dataclass(frozen=True)
class HotkeysConfig:
    """Secondary/backup control surface: global keyboard shortcuts."""

    enabled: bool = True
    mute_toggle: str = "ctrl+alt+m"
    notes_toggle: str = "ctrl+alt+n"
    push_to_talk: str = "ctrl+alt+space"


@dataclass(frozen=True)
class FeedbackConfig:
    earcons: bool = True
    spoken_confirmations: bool = True
    working_cue_interval_sec: float = 6.0
    volume: float = 0.5


@dataclass(frozen=True)
class Config:
    vad: VadConfig = field(default_factory=VadConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    mute: MuteConfig = field(default_factory=MuteConfig)
    wake_words: WakeWordsConfig = field(default_factory=WakeWordsConfig)
    headset: HeadsetConfig = field(default_factory=HeadsetConfig)
    hotkeys: HotkeysConfig = field(default_factory=HotkeysConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)


def _build(cls: type, data: Any) -> Any:
    """Recursively construct a (possibly nested) dataclass from a plain dict.

    Only keys that match a declared field are used; everything else is ignored, and
    absent fields keep their defaults. This keeps partial user configs valid.
    """
    if not isinstance(data, dict):
        return data
    # ``from __future__ import annotations`` makes ``f.type`` a string, so resolve the
    # real field types before deciding whether a nested value is itself a dataclass.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build(ftype, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | os.PathLike[str] | None) -> Config:
    """Load configuration from ``path``; return all-defaults if it is missing."""
    if path is None:
        return Config()
    p = Path(path)
    if not p.exists():
        return Config()
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    return _build(Config, raw)
