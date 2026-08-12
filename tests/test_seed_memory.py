"""Tests for the seed script's log parser and date window
(scripts/seed_agent_memory.py) — pure logic, no Chroma, no model, no files
beyond a string."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.seed_agent_memory import (DEFAULT_DAYS, chunk_by_persona_day,
                                       log_date, parse_logs, select_logs)

LOG = """\
2026-08-01 09:00:00,100 tts      INFO    TTS backend: Windows SAPI
2026-08-01 09:00:02,000 agent    INFO    startup took 2.5s
2026-08-01 09:00:10,000 agent    INFO    you: Good morning Alice.
2026-08-01 09:00:12,000 agent    INFO    agent: Morning. What's on your mind?
2026-08-01 09:01:00,000 agent    INFO    === talking to Tom (Sonnet 5) ===
2026-08-01 09:01:05,000 agent    INFO    you (typed): how did the butterfly go?
2026-08-01 09:01:08,000 agent    INFO    agent: The butterfly closed green:
- up 340 on the day
- the tail sleeve carried it
2026-08-01 09:02:00,000 agent    INFO    delegating to Bob: save a note
2026-08-02 10:00:00,000 agent    INFO    startup took 1.9s
2026-08-02 10:00:05,000 agent    INFO    you: still there?
2026-08-02 10:00:07,000 agent    INFO    agent: Here.
"""


def write_log(tmp, text=LOG, name="session_2026-08-01.log"):
    path = Path(tmp) / name
    path.write_text(text, encoding="utf-8")
    return path


class TestParseLogs(unittest.TestCase):
    def parse(self, text=LOG):
        with tempfile.TemporaryDirectory() as tmp:
            return parse_logs([write_log(tmp, text)])

    def test_turns_before_any_switch_belong_to_the_default_agent(self):
        turns = self.parse()
        self.assertEqual(turns[0], ["2026-08-01", "alice",
                                    "user: Good morning Alice."])

    def test_a_switch_banner_flips_the_persona(self):
        turns = self.parse()
        tom = [t for t in turns if t[1] == "tom"]
        self.assertEqual(tom[0][2], "user: how did the butterfly go?")

    def test_a_boot_marker_resets_to_the_default_agent(self):
        # Day 2 boots fresh: the process always starts on DEFAULT_AGENT, so
        # attribution must not carry Tom over from the previous session.
        turns = self.parse()
        self.assertEqual(turns[-1], ["2026-08-02", "alice", "assistant: Here."])

    def test_multiline_replies_are_joined(self):
        turns = self.parse()
        (reply,) = [t[2] for t in turns if "butterfly closed" in t[2]]
        self.assertIn("tail sleeve carried it", reply)

    def test_non_turn_lines_are_skipped(self):
        turns = self.parse()
        self.assertFalse(any("delegating" in t[2] for t in turns))
        self.assertFalse(any("TTS backend" in t[2] for t in turns))

    def test_typed_variant_counts_as_a_user_turn(self):
        turns = self.parse()
        self.assertTrue(any(t[2] == "user: how did the butterfly go?"
                            for t in turns))


class TestLogWindow(unittest.TestCase):
    """Only recent logs may be seeded: personas did not exist before
    2026-07-20, so an older log has nothing to attribute turns to and every
    one of them would be filed under the default persona (a first run did
    exactly that — 26 Alice records reaching back to June)."""

    def paths(self, *names):
        return [Path("logs") / n for n in names]

    def test_logs_inside_the_window_are_kept(self):
        kept, skipped, cutoff = select_logs(
            self.paths("session_2026-08-06.log", "session_2026-08-10.log"),
            days=7, today=date(2026, 8, 10))
        self.assertEqual(cutoff, date(2026, 8, 3))
        self.assertEqual([p.name for p in kept],
                         ["session_2026-08-06.log", "session_2026-08-10.log"])
        self.assertEqual(skipped, [])

    def test_older_logs_are_skipped(self):
        kept, skipped, _ = select_logs(
            self.paths("session_2026-06-22.log", "session_2026-07-19.log",
                       "session_2026-08-09.log"),
            days=7, today=date(2026, 8, 10))
        self.assertEqual([p.name for p in kept], ["session_2026-08-09.log"])
        self.assertEqual(len(skipped), 2)

    def test_the_cutoff_day_itself_is_included(self):
        kept, _, _ = select_logs(self.paths("session_2026-08-03.log"),
                                 days=7, today=date(2026, 8, 10))
        self.assertEqual(len(kept), 1)

    def test_an_unparseable_name_is_skipped_not_guessed_at(self):
        kept, skipped, _ = select_logs(self.paths("session_backup.log"),
                                       days=7, today=date(2026, 8, 10))
        self.assertEqual(kept, [])
        self.assertEqual(len(skipped), 1)

    def test_log_date_reads_the_filename(self):
        self.assertEqual(log_date(Path("logs/session_2026-08-10.log")),
                         date(2026, 8, 10))
        self.assertIsNone(log_date(Path("logs/notes.log")))

    def test_default_window_is_a_week(self):
        self.assertEqual(DEFAULT_DAYS, 7)


class TestChunking(unittest.TestCase):
    def test_chunks_group_by_persona_and_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks = chunk_by_persona_day(parse_logs([write_log(tmp)]))
        self.assertEqual(set(chunks), {("alice", "2026-08-01"),
                                       ("tom", "2026-08-01"),
                                       ("alice", "2026-08-02")})
        self.assertEqual(len(chunks[("tom", "2026-08-01")]), 2)


if __name__ == "__main__":
    unittest.main()
