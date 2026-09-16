"""The Linux button channel (media_control/linux.py + its wiring in main.py).

The wiring test needs nothing installed. The MPRIS test is live: it spawns a
private session bus with dbus-daemon, serves the player on it, and drives it
the way a desktop media-key daemon does (D-Bus method calls) — skipped where
dbus-fast or dbus-daemon is missing, and never touching the real session bus.
"""

import asyncio
import os
import shutil
import subprocess
import unittest
from unittest import mock

from media_control import main

try:
    from dbus_fast.aio import MessageBus
    from media_control import linux
except ImportError:  # dbus_fast not installed on this machine
    MessageBus = None


class LinuxWiringTests(unittest.TestCase):
    def test_buttons_survive_a_missing_bus_and_a_missing_display(self):
        # The honest-failure path: no dbus_fast (or no session bus) and no X
        # display (Wayland) must each leave the other channel standing, never
        # take start() down.
        mb = main.MediaButtons(on_click=lambda: None, on_note=lambda: None,
                               on_quit=lambda: None)
        with mock.patch.object(main.sys, "platform", "linux"), \
                mock.patch.object(mb, "_make_mpris",
                                  side_effect=ImportError("no dbus_fast")), \
                mock.patch.object(mb, "_start_keyboard_hook",
                                  side_effect=RuntimeError("no DISPLAY")):
            mb.start()
        self.assertIsNone(mb._mpris)
        mb.stop()  # nothing started, nothing to stop, no error

    def test_linux_starts_the_player_before_the_hook(self):
        mb = main.MediaButtons(on_click=lambda: None, on_note=lambda: None,
                               on_quit=lambda: None)
        order = []
        with mock.patch.object(main.sys, "platform", "linux"), \
                mock.patch.object(mb, "_start_session",
                                  side_effect=lambda make, label: order.append(label)), \
                mock.patch.object(mb, "_start_keyboard_hook",
                                  side_effect=lambda: order.append("hook")):
            mb.start()
        self.assertEqual(order, ["MPRIS", "hook"])


@unittest.skipIf(MessageBus is None or not shutil.which("dbus-daemon"),
                 "needs dbus-fast and dbus-daemon")
class MprisLiveTests(unittest.TestCase):
    def setUp(self):
        self.daemon = subprocess.Popen(
            ["dbus-daemon", "--session", "--nofork", "--print-address"],
            stdout=subprocess.PIPE, text=True)
        address = self.daemon.stdout.readline().strip()
        self.addCleanup(self.daemon.kill)
        env = mock.patch.dict(os.environ, {"DBUS_SESSION_BUS_ADDRESS": address})
        env.start()
        self.addCleanup(env.stop)

    async def _press(self, *buttons):
        """What gnome-settings-daemon does with a headset press."""
        bus = await MessageBus().connect()
        intro = await bus.introspect(linux.BUS_NAME, linux.OBJECT_PATH)
        player = bus.get_proxy_object(linux.BUS_NAME, linux.OBJECT_PATH, intro) \
            .get_interface("org.mpris.MediaPlayer2.Player")
        for button in buttons:  # dbus-fast snake_cases the member names
            await getattr(player, "call_" + button)()
        status = await player.get_playback_status()
        bus.disconnect()
        return status

    def test_presses_reach_the_callbacks_already_decoded(self):
        hits = []
        listener = linux.MediaButtonListener(
            on_play_pause=lambda: hits.append("click"),
            on_next=lambda: hits.append("note"),
            on_previous=lambda: hits.append("quit"))
        listener.start()
        self.addCleanup(listener.stop)
        status = asyncio.run(self._press("play_pause", "play", "pause",
                                         "next", "previous"))
        self.assertEqual(hits, ["click", "click", "click", "note", "quit"])
        self.assertEqual(status, "Playing")  # what makes daemons route to us

    def test_a_second_owner_is_refused_honestly(self):
        first = linux.MediaButtonListener()
        first.start()
        self.addCleanup(first.stop)
        with self.assertRaises(RuntimeError):
            linux.MediaButtonListener().start()


if __name__ == "__main__":
    unittest.main()
