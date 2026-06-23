"""Local speech-to-text via faster-whisper.

Runs fully on-device. `vad_filter=True` gives a second, model-grade pass over
each segment so any residual silence/noise that slipped through webrtcvad does
not produce hallucinated text.
"""

import logging

import numpy as np
from faster_whisper import WhisperModel

import config as cfg

log = logging.getLogger("stt")


class Transcriber:
    def __init__(self):
        log.info("loading whisper model %s (%s/%s) ...",
                 cfg.WHISPER_MODEL, cfg.WHISPER_DEVICE, cfg.WHISPER_COMPUTE)
        self.model = WhisperModel(
            cfg.WHISPER_MODEL,
            device=cfg.WHISPER_DEVICE,
            compute_type=cfg.WHISPER_COMPUTE,
        )
        log.info("whisper model ready")

    def transcribe(self, audio_int16: np.ndarray) -> str:
        if audio_int16 is None or audio_int16.size == 0:
            return ""
        audio = audio_int16.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            beam_size=5,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text
