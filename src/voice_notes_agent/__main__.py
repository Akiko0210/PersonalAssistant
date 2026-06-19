"""Entry point: ``python -m voice_notes_agent`` (§A1, phase 7 launch-at-login).

Loads config from the data store, configures file + console logging, and runs the agent
as a persistent background process. ``--install-autostart`` registers the process to
launch at login on Windows (phase 7).
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


def _install_autostart() -> int:  # pragma: no cover - Windows registry
    """Register launch-at-login via the Windows Run registry key (phase 7)."""
    if sys.platform != "win32":
        print("Autostart install is Windows-only.", file=sys.stderr)
        return 2
    import winreg

    cmd = f'"{sys.executable}" -m voice_notes_agent'
    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        0,
        winreg.KEY_SET_VALUE,
    )
    winreg.SetValueEx(key, "VoiceNotesAgent", 0, winreg.REG_SZ, cmd)
    winreg.CloseKey(key)
    print("Registered VoiceNotesAgent to launch at login.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice-notes-agent")
    parser.add_argument("--install-autostart", action="store_true", help="launch at login (Windows)")
    parser.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    args = parser.parse_args(argv)

    if args.install_autostart:
        return _install_autostart()

    paths = Paths.resolve()
    _configure_logging(paths)

    if args.list_devices:  # pragma: no cover - hardware dependent
        from .audio.devices import list_devices

        print(list_devices())
        return 0

    cfg = load_config(paths.config_file)
    app = App(cfg, paths)
    app.run_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
