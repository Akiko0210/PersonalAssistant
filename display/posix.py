"""macOS and Linux variant of the display window's OS-specific half: both go
through pywebview's own on_top (an NSWindow level on macOS, GTK/Qt keep-above
on Linux), which raises the window without activating it. Untested on either
so far — the issue targets Windows first.
"""

# Its own session, so a Ctrl+C in the agent's terminal doesn't take the
# window down with it; the window outlives the agent to say so.
POPEN_KWARGS = {"start_new_session": True}


def bring_to_front(window):
    window.on_top = True
    window.on_top = False
