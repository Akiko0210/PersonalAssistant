"""macOS and Linux variant of the display window's OS-specific half: both go
through pywebview's own on_top (an NSWindow level on macOS, GTK/Qt keep-above
on Linux), which raises the window without activating it. Untested on either
so far — the issue targets Windows first.
"""

import os

# Its own session, so one rule decides when the window goes — its agent's
# process is gone (agent_alive) — not which terminal signals reach it.
POPEN_KWARGS = {"start_new_session": True}


def agent_alive(pid):
    try:
        os.kill(pid, 0)   # signal 0: existence check only
    except ProcessLookupError:
        return False
    except PermissionError:
        return True       # exists, owned by someone else
    return True


def bring_to_front(window):
    window.on_top = True
    window.on_top = False
