"""POSIX playback variant — one file for macOS AND Linux, because both play
through sounddevice's default output stream (already a dependency for the
mic); separate macos.py/linux.py files would be two copies of the same code.
WAV only, to match what the winsound variant accepts anyway."""


def _load_wav(path):
    """Decode a 16-bit PCM WAV into (numpy array, sample rate) for sd.play.
    Only this one format: the shipped cue is one, and winsound (the other
    variant) accepts nothing else anyway."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels())
        return data, w.getframerate()


# Decoded once per path for the life of the process — the cue plays on every
# model call, and re-reading a Dropbox-synced file each time would be the kind
# of chatty disk access the project avoids.
_cache = {}


def play_loop(path):
    import sounddevice as sd
    if path not in _cache:
        _cache[path] = _load_wav(path)
    data, fs = _cache[path]
    sd.play(data, fs, loop=True)


def stop():
    import sounddevice as sd
    sd.stop()
