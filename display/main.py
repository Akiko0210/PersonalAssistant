"""The display window: a visual aid that shows what the latest trading
answer was about, in large high-contrast type (#28).

Holding four legs and their P&L in your head while listening takes a lot of
working memory, and the user can't scan the dashboard for them. So every
answer from a tool registered with `display=` (tools/__init__.py) becomes a
"card" on the live Agent: a heading, an "as of" time, and the blocks the
tool built (display/card.py) in the same loop as its spoken sentence, from
the same figures — never a separate fetch that could disagree with the
spoken answer.

Two halves live here:
- Agent side (make_card, Window): build the card, and launch or relaunch
  the window process. Cheap imports only: voice_agent imports this module.
- Window side (run): its own process, because pywebview must own its main
  thread and the agent's is taken by Agent().run(). It polls the agent's
  dashboard route /api/display and comes to the front only when the card's
  version changes. The isolation means a window crash never touches voice.

Bringing the window forward is OS-specific and lives in the variant files
beside this one: `windows.py` (SetWindowPos without activation) and
`posix.py` (pywebview's on_top toggle, shared by macOS and Linux). The window
is created non-activating (focus=False), so neither can take the keyboard
from whatever the user is typing into.
"""

import argparse
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import config as cfg
from display import card as blocks
from lib.single_instance.main import AlreadyRunning, SingleInstance

if os.name == "nt":
    from display import windows as _impl
else:
    from display import posix as _impl

log = logging.getLogger("display")

PAGE = Path(__file__).with_name("page.html")

# One statement per line: split after a sentence's full stop, never inside a
# number ("1,234.56" has no space after its point).
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


# --- Agent side ---------------------------------------------------------------

def make_card(heading, content, version):
    """The display state for one tool answer. `content` is the card the tool
    built (display/card.py); a bare string — a refusal, a "which account?",
    a broker error — is shown as plain notes, one statement per line."""
    if not isinstance(content, dict):
        content = blocks.card(notes=[
            s.strip() for line in str(content).splitlines()
            for s in _SENTENCE_END.split(line) if s.strip()])
    return {"version": version, "title": heading,
            "as_of": datetime.now().isoformat(timespec="seconds"), **content}


class Window:
    """The window process, from the agent's side. open() starts it when it
    isn't running — at agent start, and again with the next card after the
    user closed it or it crashed. A window left over from an earlier agent
    run holds the display lock, so the new process simply exits."""

    def __init__(self, port):
        self._port = port
        self._proc = None
        self._enabled = cfg.DISPLAY_WINDOW and port is not None
        if self._enabled and importlib.util.find_spec("webview") is None:
            # Otherwise every card would spawn a Python that dies on import.
            log.warning("display window off: pywebview is not installed "
                        "(pip install pywebview)")
            self._enabled = False

    def open(self):
        if not self._enabled or (self._proc and self._proc.poll() is None):
            return self
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "display.main", "--port", str(self._port)],
                cwd=cfg.BASE_DIR, **_impl.POPEN_KWARGS)
        except OSError as e:
            log.warning("could not start the display window: %s", e)
        return self


# --- Window side --------------------------------------------------------------

def fetch(port):
    """What the agent has to show, or the plain fact that it isn't there."""
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/display", timeout=2) as r:
            state = json.load(r)
    except (OSError, ValueError):
        state = {"agent_running": False, "card": None}
    return {**state, "stale_after_s": cfg.DISPLAY_STALE_S}


_OPENING = object()


def is_new(version, last):
    """Come forward for a card that wasn't there at the last poll, and only
    then: a poll with no change must never jump over the user's work."""
    return version is not None and version != last


def _follow(window, port):
    # Not on the first poll: a window that just opened is already on top.
    shown, last = None, _OPENING
    while True:
        state = fetch(port)
        if state != shown:
            # A page error must not end this loop: the window would freeze
            # on its last card, looking current.
            try:
                window.evaluate_js(f"render({json.dumps(state)})")
            except Exception:
                log.exception("display render failed")
            shown = state
        version = (state.get("card") or {}).get("version")
        if last is not _OPENING and is_new(version, last):
            _impl.bring_to_front(window)
        last = version
        time.sleep(cfg.DISPLAY_POLL_S)


def run(port):
    try:
        lock = SingleInstance(cfg.DISPLAY_LOCK_PATH).acquire()
    except AlreadyRunning:
        return  # one window already follows the same port
    import webview
    # Placed, not left to the OS: Windows cascades new windows down the
    # screen, which pushed the bottom under the taskbar. Top-right, every
    # time, so the eye learns where to look.
    screen = webview.screens[0]
    window = webview.create_window(
        cfg.DISPLAY_TITLE, html=PAGE.read_text(encoding="utf-8"),
        width=cfg.DISPLAY_WIDTH, height=screen.height - cfg.DISPLAY_SCREEN_MARGIN,
        x=screen.x + screen.width - cfg.DISPLAY_WIDTH, y=screen.y,
        focus=False, background_color="#000000")
    # Daemon, so closing the window ends the process; pywebview's own func
    # thread would keep it alive.
    threading.Thread(target=_follow, args=(window, port), daemon=True,
                     name="follow").start()
    try:
        webview.start()
    finally:
        lock.release()


def main():
    ap = argparse.ArgumentParser(description="Voice agent display window")
    ap.add_argument("--port", type=int, default=cfg.DASHBOARD_PORT)
    args = ap.parse_args()
    cfg.ensure_dirs()
    # Detached from the agent's console, so its errors go to a file.
    logging.basicConfig(filename=cfg.LOG_DIR / "display.log", level=logging.INFO,
                        format="%(asctime)s %(name)-8s %(levelname)-7s %(message)s")
    try:
        run(args.port)
    except Exception:
        log.exception("display window failed")
        raise


if __name__ == "__main__":
    main()
