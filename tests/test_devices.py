"""Tests for PyAudio device auto-selection used by Pipecat LISTENING mode."""

from __future__ import annotations

from voice_notes_agent.audio.devices import PyAudioDeviceInfo, select_pyaudio_device_indexes


def test_auto_selects_matching_wasapi_headset_pair():
    devices = [
        PyAudioDeviceInfo(0, "Microsoft Sound Mapper - Input", 2, 0, "MME"),
        PyAudioDeviceInfo(1, "Headset (EarPods)", 1, 0, "MME"),
        PyAudioDeviceInfo(3, "Headset (EarPods)", 0, 2, "MME"),
        PyAudioDeviceInfo(8, "Headset (EarPods)", 1, 0, "Windows DirectSound"),
        PyAudioDeviceInfo(10, "Headset (EarPods)", 0, 2, "Windows DirectSound"),
        PyAudioDeviceInfo(18, "Headset (EarPods)", 2, 0, "Windows WDM-KS"),
        PyAudioDeviceInfo(30, "Output (EarPods)", 0, 2, "Windows WASAPI"),
        PyAudioDeviceInfo(31, "Headset (EarPods)", 1, 0, "Windows WASAPI"),
    ]

    assert select_pyaudio_device_indexes(devices=devices) == {
        "input_device_index": 31,
        "output_device_index": 30,
    }


def test_auto_selects_missing_output_for_configured_input():
    devices = [
        PyAudioDeviceInfo(20, "Microphone (Realtek HD Audio Mic input)", 2, 0, "Windows WDM-KS"),
        PyAudioDeviceInfo(30, "Output (EarPods)", 0, 2, "Windows WASAPI"),
        PyAudioDeviceInfo(31, "Headset (EarPods)", 1, 0, "Windows WASAPI"),
    ]

    assert select_pyaudio_device_indexes(input_device_index=31, devices=devices) == {
        "input_device_index": 31,
        "output_device_index": 30,
    }


def test_auto_falls_back_to_defaults_without_confident_pair():
    devices = [
        PyAudioDeviceInfo(20, "USB Microphone", 2, 0, "Windows WDM-KS"),
        PyAudioDeviceInfo(
            22,
            "External Speakers",
            0,
            2,
            "Windows WDM-KS",
            is_default_output=True,
        ),
        PyAudioDeviceInfo(
            23,
            "Laptop Array Microphone",
            2,
            0,
            "Windows WASAPI",
            is_default_input=True,
        ),
    ]

    assert select_pyaudio_device_indexes(devices=devices) == {
        "input_device_index": 23,
        "output_device_index": 22,
    }
