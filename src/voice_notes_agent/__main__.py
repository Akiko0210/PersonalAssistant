"""Entry point: ``python -m voice_notes_agent``.

Loads project-local config, configures file + console logging, and runs the agent in
the foreground until the user stops it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler

from .app import App
from .config import load_config
from .paths import Paths


def _configure_logging(paths: Paths) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        paths.logs / "agent.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice-notes-agent")
    parser.add_argument("--list-devices", action="store_true", help="print sounddevice devices and exit")
    parser.add_argument(
        "--list-pyaudio-devices",
        action="store_true",
        help="print PyAudio devices for Pipecat local audio and exit",
    )
    args = parser.parse_args(argv)

    paths = Paths.resolve()
    _configure_logging(paths)

    if args.list_devices:  # pragma: no cover - hardware dependent
        from .audio.devices import list_devices

        print(list_devices())
        return 0
    if args.list_pyaudio_devices:  # pragma: no cover - hardware dependent
        from .audio.devices import list_pyaudio_devices

        print(list_pyaudio_devices())
        return 0

    cfg = load_config(paths.config_file)
    app = App(cfg, paths)
    app.run_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
