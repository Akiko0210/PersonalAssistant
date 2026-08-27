"""Linux media-button status — nothing OS-specific to run yet.

What Linux gets today is main.py's common channel: the pynput keyboard hook,
which hears XF86AudioPlay from wired headsets and USB dongles under X11.
Known limits, so nobody re-discovers them the hard way:

- Wayland: pynput cannot listen globally — no headset buttons there until a
  desktop-portal/MPRIS path exists.
- Bluetooth-native headsets (AVRCP): presses go to the desktop's MPRIS media
  player, not the key event stream. Receiving them means registering our own
  MPRIS service on D-Bus (org.mpris.MediaPlayer2.Player with PlayPause /
  Next / Previous handlers) — the Linux twin of windows.py's SMTC session.
  Deliberately not built until someone runs the agent on Linux with such a
  headset; it would slot in via main.py exactly like the SMTC channel does.
"""
