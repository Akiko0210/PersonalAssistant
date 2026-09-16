"""The Linux channel: an MPRIS media player on the session D-Bus.

Wired into the agent by main.py, Linux only — this module is the one place
allowed to import dbus_fast. It is the Linux twin of windows.py's SMTC
session. A Bluetooth-native headset's presses (AVRCP) never reach the
keyboard hook here: BlueZ turns them into media-key events, the desktop's
media-key daemon (gnome-settings-daemon, KDE, playerctld) grabs those and
forwards them over D-Bus to an MPRIS player — so unless this process IS one,
the presses go to whatever music app is open, or nowhere. Registering
org.mpris.MediaPlayer2.voice_agent and claiming PlaybackStatus=Playing makes
the daemon route them here, already decoded:

    Play / Pause / PlayPause  <- single press
    Next                      <- double press (firmware-decoded, AirPods-class)
    Previous                  <- triple press

Under Wayland this is the ONLY working channel (pynput can't listen globally
there); under X11 a wired headset's press may also arrive via the keyboard
hook, and the gesture decoder counts it once.
"""

import asyncio
import logging
import threading

from dbus_fast import BusType, Variant
from dbus_fast.aio import MessageBus
from dbus_fast.constants import RequestNameReply
from dbus_fast.service import PropertyAccess, ServiceInterface, dbus_property, method

log = logging.getLogger("media")

BUS_NAME = "org.mpris.MediaPlayer2.voice_agent"
OBJECT_PATH = "/org/mpris/MediaPlayer2"


class _Root(ServiceInterface):
    """org.mpris.MediaPlayer2 — the identity half. Daemons only need it to
    exist; every capability is declined."""

    def __init__(self, title):
        super().__init__("org.mpris.MediaPlayer2")
        self._title = title

    @method()
    def Raise(self):
        pass

    @method()
    def Quit(self):
        pass

    @dbus_property(access=PropertyAccess.READ)
    def Identity(self) -> "s":
        return self._title

    @dbus_property(access=PropertyAccess.READ)
    def CanQuit(self) -> "b":
        return False

    @dbus_property(access=PropertyAccess.READ)
    def CanRaise(self) -> "b":
        return False

    @dbus_property(access=PropertyAccess.READ)
    def HasTrackList(self) -> "b":
        return False

    @dbus_property(access=PropertyAccess.READ)
    def SupportedUriSchemes(self) -> "as":
        return []

    @dbus_property(access=PropertyAccess.READ)
    def SupportedMimeTypes(self) -> "as":
        return []


class _Player(ServiceInterface):
    """org.mpris.MediaPlayer2.Player — the button half. Every command the
    spec lists exists (a daemon probing a missing one gets a D-Bus error and
    may drop us); the ones that aren't buttons are no-ops."""

    def __init__(self, on_play_pause, on_next, on_previous, title):
        super().__init__("org.mpris.MediaPlayer2.Player")
        self._on_play_pause = on_play_pause
        self._on_next = on_next
        self._on_previous = on_previous
        self._title = title

    def _hit(self, name, cb):
        # Runs on the D-Bus thread: callbacks must be quick and thread-safe,
        # exactly as with the SMTC channel.
        log.info("media button (MPRIS): %s", name)
        if cb is None:
            return
        try:
            cb()
        except Exception:  # noqa: BLE001 - never bounce an error back to the daemon
            log.exception("media button callback failed")

    # A headset alternates between AVRCP PLAY and PAUSE with its own idea of
    # the state, and daemons forward those as Play/Pause rather than
    # PlayPause — all three are one physical click.
    @method()
    def PlayPause(self):
        self._hit("PlayPause", self._on_play_pause)

    @method()
    def Play(self):
        self._hit("Play", self._on_play_pause)

    @method()
    def Pause(self):
        self._hit("Pause", self._on_play_pause)

    @method()
    def Next(self):
        self._hit("Next", self._on_next)

    @method()
    def Previous(self):
        self._hit("Previous", self._on_previous)

    @method()
    def Stop(self):
        pass

    @method()
    def Seek(self, offset: "x"):
        pass

    @method()
    def SetPosition(self, track_id: "o", position: "x"):
        pass

    @method()
    def OpenUri(self, uri: "s"):
        pass

    # Always "Playing": the media-key daemons hand presses to the player that
    # is playing (GNOME: the one that most recently said so) — the same trick
    # the silent keepalive plays on Windows, minus the audio.
    @dbus_property(access=PropertyAccess.READ)
    def PlaybackStatus(self) -> "s":
        return "Playing"

    @dbus_property(access=PropertyAccess.READ)
    def Metadata(self) -> "a{sv}":
        return {"mpris:trackid": Variant("o", OBJECT_PATH + "/Track/0"),
                "xesam:title": Variant("s", self._title)}

    @dbus_property(access=PropertyAccess.READ)
    def LoopStatus(self) -> "s":
        return "None"

    @dbus_property(access=PropertyAccess.READ)
    def Shuffle(self) -> "b":
        return False

    @dbus_property(access=PropertyAccess.READ)
    def Volume(self) -> "d":
        return 1.0

    @dbus_property(access=PropertyAccess.READ)
    def Position(self) -> "x":
        return 0

    @dbus_property(access=PropertyAccess.READ)
    def Rate(self) -> "d":
        return 1.0

    @dbus_property(access=PropertyAccess.READ)
    def MinimumRate(self) -> "d":
        return 1.0

    @dbus_property(access=PropertyAccess.READ)
    def MaximumRate(self) -> "d":
        return 1.0

    @dbus_property(access=PropertyAccess.READ)
    def CanGoNext(self) -> "b":
        return True

    @dbus_property(access=PropertyAccess.READ)
    def CanGoPrevious(self) -> "b":
        return True

    @dbus_property(access=PropertyAccess.READ)
    def CanPlay(self) -> "b":
        return True

    @dbus_property(access=PropertyAccess.READ)
    def CanPause(self) -> "b":
        return True

    @dbus_property(access=PropertyAccess.READ)
    def CanSeek(self) -> "b":
        return False

    @dbus_property(access=PropertyAccess.READ)
    def CanControl(self) -> "b":
        return True


class MediaButtonListener:
    """Owns the MPRIS player, served from its own daemon thread (dbus_fast is
    asyncio; the agent is not), and dispatches presses to callbacks."""

    def __init__(self, on_play_pause=None, on_next=None, on_previous=None,
                 session_title="Voice Agent"):
        self._root = _Root(session_title)
        self._player = _Player(on_play_pause, on_next, on_previous, session_title)
        self._loop = None
        self._bus = None
        self._error = None
        self._ready = threading.Event()

    def start(self):
        """Raises when the session bus can't be reached or the name is already
        owned — main.py turns that into the honest warning."""
        threading.Thread(target=self._run, name="mpris", daemon=True).start()
        if not self._ready.wait(5.0):
            raise RuntimeError("session bus did not answer")
        if self._error is not None:
            raise self._error
        log.info("MPRIS player active (%s)", BUS_NAME)

    def _run(self):
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self):
        bus = None
        try:
            bus = await MessageBus(bus_type=BusType.SESSION).connect()
            bus.export(OBJECT_PATH, self._root)
            bus.export(OBJECT_PATH, self._player)
            reply = await bus.request_name(BUS_NAME)
            if reply != RequestNameReply.PRIMARY_OWNER:
                raise RuntimeError(f"{BUS_NAME} is already owned ({reply.name})")
            # Announce Playing explicitly: daemons that track the active
            # player listen for this signal rather than re-reading properties.
            self._player.emit_properties_changed({"PlaybackStatus": "Playing"})
        except Exception as e:  # noqa: BLE001 - reported to start() on the caller's thread
            if bus is not None:
                bus.disconnect()
            self._error = e
            self._ready.set()
            return
        self._bus = bus
        self._ready.set()
        await bus.wait_for_disconnect()

    def stop(self):
        if self._bus is not None:
            self._loop.call_soon_threadsafe(self._bus.disconnect)
