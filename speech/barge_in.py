"""Barge-in detection: decide when the user's voice should interrupt speech.

Extracted from Agent.say() so the tricky parts — echo calibration, the leaky
voiced counter, and retaining the frames consumed while deciding — live in one
testable object. The Agent feeds every mic frame captured while the TTS is
playing into `feed()`; when it returns True, the agent stops speaking and
pushes `detector.run` back to the mic so the user's opening words aren't lost.
"""

import collections
import logging

import config as cfg

log = logging.getLogger("barge_in")


class BargeInDetector:
    """Feed (is_speech, rms, frame) per mic frame; returns True when the user
    has audibly interrupted.

    Calibrates an echo baseline from the first BARGE_IN_CALIB_MS of playback
    (the user almost never barges in that quickly), then requires frames to be
    BOTH voiced (per VAD) and louder than the calibrated threshold — rejecting
    the agent's own echo. Qualifying frames accumulate a counter that *decays*
    (rather than resets) on non-qualifying frames, so brief VAD/energy dropouts
    mid-word don't wipe progress. Fires once the counter reaches BARGE_IN_MS.
    """

    def __init__(self, *, energy_floor=None, energy_ratio=None, calib_ms=None,
                 fire_ms=None, decay=None, frame_ms=None, pad_ms=None):
        self._energy_floor = cfg.BARGE_IN_ENERGY if energy_floor is None else energy_floor
        self._energy_ratio = cfg.BARGE_IN_ENERGY_RATIO if energy_ratio is None else energy_ratio
        self._calib_target_ms = cfg.BARGE_IN_CALIB_MS if calib_ms is None else calib_ms
        self._fire_ms = cfg.BARGE_IN_MS if fire_ms is None else fire_ms
        self._decay = cfg.BARGE_IN_DECAY if decay is None else decay
        self._frame_ms = cfg.FRAME_MS if frame_ms is None else frame_ms
        pad_ms = cfg.SPEECH_PAD_MS if pad_ms is None else pad_ms

        self.threshold = float(self._energy_floor)  # until calibration finishes
        self.peak_speech_rms = 0.0  # loudest voiced frame seen — logged for tuning
        self._calib_ms = 0
        self._echo_samples = []
        self._voiced_ms = 0.0

        # Retain the audio consumed while deciding this is a real barge-in, so
        # the opening words aren't lost. `_recent` keeps a short pre-roll; once
        # a qualifying voiced run starts, `run` accumulates it (pre-roll
        # included) and is pushed back to the mic when the interruption fires.
        pad_frames = max(1, pad_ms // self._frame_ms)
        self._recent = collections.deque(maxlen=pad_frames)
        self.run = None

    @property
    def calibrating(self) -> bool:
        return self._calib_ms < self._calib_target_ms

    def feed(self, is_speech, rms, frame) -> bool:
        """Process one mic frame. Returns True when the interruption fires;
        the caller should then stop TTS and push back `self.run`."""
        self._recent.append(frame)

        # Calibrate the echo floor from the first part of playback, then lock
        # the threshold. No interruption can fire during calibration.
        if self.calibrating:
            self._echo_samples.append(rms)
            self._calib_ms += self._frame_ms
            if not self.calibrating and self._echo_samples:
                self._echo_samples.sort()
                # A low percentile rather than the median: if the user starts
                # talking *during* calibration, their loud frames shouldn't
                # inflate the echo baseline. An inflated baseline pushes the
                # threshold so high the rest of this utterance can't be
                # interrupted — the "sometimes I can't barge in" failure.
                baseline = self._echo_samples[len(self._echo_samples) // 3]
                self.threshold = max(self._energy_floor,
                                     baseline * self._energy_ratio)
                log.info("barge-in armed (echo baseline=%.0f, threshold=%.0f)",
                         baseline, self.threshold)
            return False

        if is_speech:
            self.peak_speech_rms = max(self.peak_speech_rms, rms)

        # Only loud, voiced audio counts — this rejects the agent's own echo.
        if is_speech and rms > self.threshold:
            self.run = list(self._recent) if self.run is None else self.run + [frame]
            self._voiced_ms += self._frame_ms
            if self._voiced_ms >= self._fire_ms:
                return True
        else:
            self._voiced_ms = max(0.0, self._voiced_ms - self._frame_ms * self._decay)
            if self.run is not None:
                self.run.append(frame)        # keep brief gaps within the run
                if self._voiced_ms <= 0:
                    self.run = None            # run fizzled — it was a false start

        return False

    def log_summary(self):
        """After playback finishes without an interruption: if the user clearly
        spoke but we never triggered, the threshold is probably too high —
        surface the numbers so BARGE_IN_ENERGY can be tuned."""
        if self.peak_speech_rms > 0:
            log.info("reply finished; loudest speech rms=%.0f vs threshold=%.0f"
                     " (no barge-in)", self.peak_speech_rms, self.threshold)
