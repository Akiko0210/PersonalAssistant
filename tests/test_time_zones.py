"""Zone names on spoken clock times. A clock time with no zone got read as
Eastern ("market closed just now", 2026-10-16), so every one says its zone."""

import unittest
from datetime import datetime, timedelta, timezone

from lib.dates import zone_name
from tools import ToolContext, dispatch


def zone(name):
    return datetime(2026, 10, 16, tzinfo=timezone(timedelta(hours=-7), name))


class TestZoneName(unittest.TestCase):
    def test_windows_full_names_become_the_usual_abbreviation(self):
        # Windows says "Pacific Daylight Time" where macOS/Linux say "PDT".
        self.assertEqual(zone_name(zone("Pacific Daylight Time")), "PDT")
        self.assertEqual(zone_name(zone("Pacific Standard Time")), "PST")
        self.assertEqual(zone_name(zone("PDT")), "PDT")


class TestCurrentTime(unittest.TestCase):
    def test_the_clock_names_the_local_zone(self):
        out = dispatch(ToolContext(), "get_current_time", {})
        self.assertTrue(out.endswith(" " + zone_name(datetime.now().astimezone())), out)


if __name__ == "__main__":
    unittest.main()
