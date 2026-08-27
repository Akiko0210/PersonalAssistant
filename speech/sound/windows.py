"""Windows playback variant: the built-in winsound, which loops and stops
cleanly and needs no extra dependencies (WAV only)."""


def play_loop(path):
    import winsound
    winsound.PlaySound(
        str(path),
        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
    )


def stop():
    import winsound
    winsound.PlaySound(None, winsound.SND_PURGE)
