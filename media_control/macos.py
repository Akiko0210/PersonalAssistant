"""The macOS channel helpers: media-key decoding for the pynput event tap.

Wired into the agent by main.py, macOS only — this module is the one place
allowed to import AppKit for buttons. On macOS, headset/media keys arrive as
NSSystemDefined events (subtype 8, the
"AUX control buttons"), with the key id and up/down state packed into data1.
pynput decodes these into Key.media_* for on_press, but decoding alone isn't
enough: unless the tap also swallows the event, every press reaches the
system's default handler too — which launches or controls Music.app on top of
whatever the agent does with the click. `darwin_intercept` below is passed to
pynput's darwin listener to consume exactly the keys the agent handles
(play/pause, next, previous) and let everything else through.
"""

NS_SYSTEM_DEFINED = 14  # CGEvent type of NSSystemDefined events
MEDIA_KEY_SUBTYPE = 8   # NSEvent subtype for AUX control buttons
NX_KEYTYPE_PLAY = 16
NX_KEYTYPE_NEXT = 17
NX_KEYTYPE_PREVIOUS = 18

# Only the keys the agent acts on. Deliberately NOT the whole media family
# (fast-forward, volume, ...): swallowing a key we don't handle would break it
# system-wide for as long as the agent runs.
_AGENT_KEYS = {NX_KEYTYPE_PLAY, NX_KEYTYPE_NEXT, NX_KEYTYPE_PREVIOUS}


def is_agent_media_key(subtype: int, data1: int) -> bool:
    """True when an NSSystemDefined event is a media key the agent consumes.
    Both the down and up halves match — suppressing only the down would still
    hand the release to Music.app. Pure, so it's testable without Quartz."""
    return subtype == MEDIA_KEY_SUBTYPE and ((data1 >> 16) & 0xFFFF) in _AGENT_KEYS


def darwin_intercept(event_type, event):
    """pynput `darwin_intercept` hook: return None to swallow our media keys,
    the untouched event for everything else."""
    if event_type != NS_SYSTEM_DEFINED:
        return event
    try:
        from AppKit import NSEvent
        ns = NSEvent.eventWithCGEvent_(event)
        if ns is not None and is_agent_media_key(ns.subtype(), int(ns.data1())):
            return None
    except Exception:  # noqa: BLE001 - decoding trouble must not kill the tap
        pass
    return event
