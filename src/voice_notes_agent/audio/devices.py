"""Audio device helpers and earcon/TTS playback (§7).

Thin wrappers over ``sounddevice`` (PortAudio). Imported lazily by callers so the
package imports without PortAudio present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PyAudioDeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    host_api_name: str = ""
    is_default_input: bool = False
    is_default_output: bool = False


def list_devices() -> str:  # pragma: no cover - hardware dependent
    import sounddevice as sd

    return str(sd.query_devices())


def list_pyaudio_devices() -> str:  # pragma: no cover - hardware dependent
    devices = _query_pyaudio_devices()
    return "\n".join(
        f"{d.index}: {d.name} in={d.max_input_channels} out={d.max_output_channels} "
        f"host={d.host_api_name}"
        for d in devices
    )


def select_pyaudio_device_indexes(
    *,
    input_device_index: int | None = None,
    output_device_index: int | None = None,
    devices: Iterable[PyAudioDeviceInfo] | None = None,
) -> dict[str, int]:
    """Select Pipecat PyAudio input/output indexes, pairing headsets automatically."""
    selected: dict[str, int] = {}
    if input_device_index is not None:
        selected["input_device_index"] = input_device_index
    if output_device_index is not None:
        selected["output_device_index"] = output_device_index
    if len(selected) == 2:
        return selected

    available = list(devices) if devices is not None else _query_pyaudio_devices()
    inputs = [d for d in available if d.max_input_channels > 0]
    outputs = [d for d in available if d.max_output_channels > 0]
    if not inputs or not outputs:
        return selected

    configured_input = _find_device(inputs, input_device_index)
    configured_output = _find_device(outputs, output_device_index)

    if "input_device_index" not in selected:
        selected_input = (
            _best_input_for_output(inputs, configured_output)
            if configured_output is not None
            else _best_pair(inputs, outputs)[0]
        )
        if selected_input is not None:
            selected["input_device_index"] = selected_input.index

    if "output_device_index" not in selected:
        selected_output = (
            _best_output_for_input(outputs, configured_input)
            if configured_input is not None
            else _best_pair(inputs, outputs)[1]
        )
        if selected_output is not None:
            selected["output_device_index"] = selected_output.index

    return selected


def _query_pyaudio_devices() -> list[PyAudioDeviceInfo]:  # pragma: no cover - hardware dependent
    import pyaudio

    py_audio = pyaudio.PyAudio()
    try:
        default_input = _default_pyaudio_index(py_audio, "input")
        default_output = _default_pyaudio_index(py_audio, "output")
        devices = []
        for index in range(py_audio.get_device_count()):
            info = py_audio.get_device_info_by_index(index)
            host_api = py_audio.get_host_api_info_by_index(info.get("hostApi", 0))
            devices.append(
                PyAudioDeviceInfo(
                    index=index,
                    name=str(info.get("name", "")),
                    max_input_channels=int(info.get("maxInputChannels", 0)),
                    max_output_channels=int(info.get("maxOutputChannels", 0)),
                    host_api_name=str(host_api.get("name", "")),
                    is_default_input=index == default_input,
                    is_default_output=index == default_output,
                )
            )
        return devices
    finally:
        py_audio.terminate()


def _default_pyaudio_index(py_audio, kind: str) -> int | None:  # pragma: no cover
    try:
        if kind == "input":
            return int(py_audio.get_default_input_device_info()["index"])
        return int(py_audio.get_default_output_device_info()["index"])
    except Exception:
        return None


def _find_device(devices: list[PyAudioDeviceInfo], index: int | None) -> PyAudioDeviceInfo | None:
    if index is None:
        return None
    return next((d for d in devices if d.index == index), None)


def _best_pair(
    inputs: list[PyAudioDeviceInfo],
    outputs: list[PyAudioDeviceInfo],
) -> tuple[PyAudioDeviceInfo | None, PyAudioDeviceInfo | None]:
    best: tuple[int, PyAudioDeviceInfo | None, PyAudioDeviceInfo | None] = (-10_000, None, None)
    for input_device in inputs:
        for output_device in outputs:
            score = _pair_score(input_device, output_device)
            if score > best[0]:
                best = (score, input_device, output_device)
    if best[0] < 80:
        return _default_or_first(inputs, "input"), _default_or_first(outputs, "output")
    return best[1], best[2]


def _best_output_for_input(
    outputs: list[PyAudioDeviceInfo],
    input_device: PyAudioDeviceInfo | None,
) -> PyAudioDeviceInfo | None:
    if input_device is None:
        return _default_or_first(outputs, "output")
    return max(outputs, key=lambda output: _pair_score(input_device, output), default=None)


def _best_input_for_output(
    inputs: list[PyAudioDeviceInfo],
    output_device: PyAudioDeviceInfo | None,
) -> PyAudioDeviceInfo | None:
    if output_device is None:
        return _default_or_first(inputs, "input")
    return max(inputs, key=lambda input_device: _pair_score(input_device, output_device), default=None)


def _default_or_first(devices: list[PyAudioDeviceInfo], kind: str) -> PyAudioDeviceInfo | None:
    if kind == "input":
        default = next((d for d in devices if d.is_default_input), None)
    else:
        default = next((d for d in devices if d.is_default_output), None)
    return default or (devices[0] if devices else None)


def _pair_score(input_device: PyAudioDeviceInfo, output_device: PyAudioDeviceInfo) -> int:
    input_tokens = _device_tokens(input_device.name)
    output_tokens = _device_tokens(output_device.name)
    shared = input_tokens & output_tokens
    score = len(shared) * 100

    if input_device.host_api_name == output_device.host_api_name:
        score += 25
    score += _host_api_score(input_device.host_api_name)
    score += _host_api_score(output_device.host_api_name)
    score += min(output_device.max_output_channels, 2) * 5

    if _is_generic(input_device.name):
        score -= 80
    if _is_generic(output_device.name):
        score -= 80
    return score


_GENERIC_TOKENS = {
    "audio",
    "capture",
    "driver",
    "hands",
    "headphones",
    "headset",
    "input",
    "mapper",
    "microsoft",
    "output",
    "primary",
    "sound",
}


def _device_tokens(name: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    return {token for token in tokens if len(token) > 1 and token not in _GENERIC_TOKENS}


def _host_api_score(name: str) -> int:
    lowered = name.lower()
    if "mme" in lowered:
        return 40
    if "directsound" in lowered:
        return 30
    if "wasapi" in lowered:
        return 20
    if "wdm" in lowered or "ks" in lowered:
        return 10
    return 0


def _is_generic(name: str) -> bool:
    lowered = name.lower()
    return (
        "microsoft sound mapper" in lowered
        or "primary sound" in lowered
        or lowered.startswith("input (")
        or lowered.startswith("output (")
    )


def resolve_device(spec) -> object:
    """Resolve a device spec (int index, name substring, or None=default)."""
    return spec  # sounddevice accepts index, name, or None directly


def play(audio: np.ndarray, sample_rate: int, *, device=None, blocking: bool = False) -> None:
    """Play a mono float32 buffer on the output device (earcons + TTS)."""
    import sounddevice as sd  # pragma: no cover - hardware dependent

    sd.play(audio, samplerate=sample_rate, device=device, blocking=blocking)


def stop_playback() -> None:  # pragma: no cover - hardware dependent
    import sounddevice as sd

    sd.stop()
