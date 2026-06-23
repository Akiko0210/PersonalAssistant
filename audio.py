"""Microphone capture with voice-activity detection and utterance endpointing.

A single always-on input stream pushes fixed-size frames onto a queue. Consumers
pull whole *utterances* via `collect_utterance`, which uses webrtcvad to ignore
silence — so an hour-long notetaking session with only minutes of speech does
almost no work while the user is quiet.
"""

import collections
import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import webrtcvad

import config as cfg

log = logging.getLogger("audio")


class AudioEngine:
    def __init__(self):
        self.q: "queue.Queue[bytes]" = queue.Queue()
        self.vad = webrtcvad.Vad(cfg.VAD_AGGRESSIVENESS)
        self.muted = threading.Event()
        self._stream = None

    # --- stream lifecycle ----------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("input status: %s", status)
        self.q.put(bytes(indata))

    def start(self):
        self._stream = sd.RawInputStream(
            samplerate=cfg.SAMPLE_RATE,
            blocksize=cfg.FRAME_SAMPLES,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )
        self._stream.start()
        log.info("microphone stream started")

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def flush(self):
        """Drop any buffered frames (e.g. echo of our own TTS)."""
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    # --- capture helpers -----------------------------------------------------
    @staticmethod
    def _to_array(frames):
        return np.frombuffer(b"".join(frames), dtype=np.int16)

    def collect_utterance(self, interrupt=None, endpoint_ms=None):
        """Block until one spoken utterance is captured, then return it as an
        int16 numpy array. Returns None if interrupted before/at speech start,
        or while muted. If `interrupt` fires mid-utterance, the partial audio
        captured so far is returned.
        """
        endpoint_ms = endpoint_ms or cfg.CONVO_ENDPOINT_MS
        pad_frames = max(1, cfg.SPEECH_PAD_MS // cfg.FRAME_MS)
        ring = collections.deque(maxlen=pad_frames)
        triggered = False
        voiced = []
        silence_ms = 0
        start = time.monotonic()

        while True:
            if interrupt is not None and interrupt.is_set():
                return self._to_array(voiced) if (triggered and voiced) else None

            try:
                frame = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.muted.is_set():
                ring.clear()
                triggered = False
                voiced = []
                silence_ms = 0
                continue

            is_speech = self.vad.is_speech(frame, cfg.SAMPLE_RATE)

            if not triggered:
                ring.append((frame, is_speech))
                num_voiced = sum(1 for _, s in ring if s)
                if num_voiced > cfg.TRIGGER_RATIO * ring.maxlen:
                    triggered = True
                    voiced.extend(f for f, _ in ring)
                    ring.clear()
                    silence_ms = 0
            else:
                voiced.append(frame)
                if is_speech:
                    silence_ms = 0
                else:
                    silence_ms += cfg.FRAME_MS
                    if silence_ms >= endpoint_ms:
                        return self._to_array(voiced)
                if (time.monotonic() - start) > cfg.MAX_UTTERANCE_S:
                    return self._to_array(voiced)

    def poll_speech(self, timeout=0.1):
        """Pull one frame and classify it. Returns (is_speech, rms) where is_speech
        is the VAD verdict and rms is the frame loudness (int16 RMS), or None if no
        frame arrived within `timeout`. While muted, is_speech is forced False.
        Used to watch for barge-in while the agent is talking."""
        try:
            frame = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        samples = np.frombuffer(frame, dtype=np.int16)
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))) if samples.size else 0.0
        if self.muted.is_set():
            return (False, rms)
        return (self.vad.is_speech(frame, cfg.SAMPLE_RATE), rms)

    def record_seconds(self, seconds):
        """Capture a fixed window of audio (used by --selftest)."""
        self.flush()
        needed = int(seconds * 1000 / cfg.FRAME_MS)
        frames = []
        while len(frames) < needed:
            try:
                frames.append(self.q.get(timeout=1.0))
            except queue.Empty:
                break
        return self._to_array(frames)
