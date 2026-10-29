"""Tests for the post-session analyzer (scripts/analyze_session.py): which
slice of a daily log counts as "the session that just ended", what is stripped
before the model sees it, and that every CLI failure still leaves an honest
_analyzed file. The CLI itself is faked — no model call, no network."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import analyze_session as an

SESSION_1 = """\
2026-10-15 14:12:43,312 agent    INFO    scanning knowledge base...
2026-10-15 14:13:14,796 agent    INFO    Ready. Headset button: 1-click=mute  2-click=note  3-click=quit
2026-10-15 14:14:00,000 agent    INFO    you: what's on the agenda?
2026-10-15 16:37:55,317 agent    INFO    shut down
"""
SESSION_2 = """\
2026-10-15 16:39:46,709 agent    INFO    scanning knowledge base...
2026-10-15 16:39:56,070 agent    INFO    Ready. Headset button: 1-click=mute  2-click=note  3-click=quit
2026-10-15 16:41:38,115 agent    INFO    you: Let's continue what we were working on.
2026-10-15 16:41:39,000 httpx    INFO    HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
2026-10-15 16:41:40,000 context  DEBUG   Background block:
  - candidate table row one
  - candidate table row two
2026-10-15 16:41:41,419 llm      INFO    tool_use recall {'period': 'today'}
2026-10-15 16:41:44,355 agent    INFO    agent: Nothing's been closed out since this morning.
2026-10-15 16:42:01,823 agent    INFO    you: no, you said that yesterday
2026-10-15 16:42:24,927 httpx    WARNING connection reset by peer
2026-10-15 17:14:30,872 agent    INFO    shut down
"""
# An --ingest pass or a refused second copy: logs, but never says "Ready."
INGEST_TAIL = """\
2026-10-15 17:20:00,000 knowledge INFO    Knowledge base up to date.
"""


def lines(text):
    return text.splitlines(True)


class TestLastSession(unittest.TestCase):
    def test_last_of_several_sessions(self):
        first, s = an.last_session(lines(SESSION_1 + SESSION_2))
        self.assertEqual(first, 5)
        self.assertEqual(s, lines(SESSION_2))

    def test_trailing_launch_that_never_started_is_skipped(self):
        # run.bat's next plain launch ingests first; that tail must not
        # displace the session that actually ran.
        _, s = an.last_session(lines(SESSION_1 + SESSION_2 + INGEST_TAIL))
        self.assertEqual(s, lines(SESSION_2))

    def test_crash_without_shut_down_is_the_session(self):
        crashed = SESSION_2.replace(
            "2026-10-15 17:14:30,872 agent    INFO    shut down\n", "")
        _, s = an.last_session(lines(SESSION_1 + crashed))
        self.assertEqual(s, lines(crashed))


class TestCondense(unittest.TestCase):
    def test_drops_debug_with_its_continuation_and_chatter(self):
        out = an.condense(lines(SESSION_2))
        self.assertNotIn("candidate table", out)
        self.assertNotIn("HTTP Request", out)
        self.assertIn("tool_use recall", out)
        self.assertIn("you: Let's continue", out)

    def test_keeps_warnings_from_noisy_loggers(self):
        self.assertIn("connection reset by peer", an.condense(lines(SESSION_2)))


class TestAnalyze(unittest.TestCase):
    def run_analyze(self, which="/bin/claude", result=None, raises=None):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "session_2026-10-15.log"
            log.write_text(SESSION_1 + SESSION_2, encoding="utf-8")
            run = mock.Mock(return_value=result, side_effect=raises)
            with mock.patch.object(an.shutil, "which", return_value=which), \
                    mock.patch.object(an.subprocess, "run", run):
                out, ok = an.analyze(log)
            return out.name, ok, out.read_text(encoding="utf-8"), run

    def test_writes_report_with_computed_metrics(self):
        done = subprocess.CompletedProcess([], 0, "## Session overview\nFine.", "")
        name, ok, text, run = self.run_analyze(result=done)
        self.assertTrue(ok)
        self.assertEqual(name, "session_2026-10-15_1639_analyzed.md")
        self.assertIn("## Session overview", text)
        # Numbers come from the audit parser over this session only — the
        # 14:14 turn belongs to the earlier session.
        self.assertIn("user turns: 2", text)
        self.assertIn('"no, you said that yesterday"', text)
        # Prompt and log travel on stdin (Windows .cmd shims mangle long args),
        # already condensed, with the raw line range for the model to Grep.
        stdin = run.call_args.kwargs["input"]
        self.assertNotIn("HTTP Request", stdin)
        self.assertIn("lines 5-", stdin)
        self.assertNotIn("Let's", " ".join(run.call_args.args[0]))

    def test_missing_cli_still_writes_an_honest_file(self):
        name, ok, text, run = self.run_analyze(which=None)
        self.assertFalse(ok)
        self.assertIn("## Analysis failed", text)
        self.assertIn("not on PATH", text)
        self.assertIn("user turns: 2", text)  # the metrics survive the failure
        run.assert_not_called()

    def test_not_logged_in_is_reported_with_cli_output(self):
        bad = subprocess.CompletedProcess([], 1, "Not logged in · Please run /login", "")
        _, ok, text, _ = self.run_analyze(result=bad)
        self.assertFalse(ok)
        self.assertIn("exited 1", text)
        self.assertIn("Not logged in", text)

    def test_timeout_is_reported(self):
        _, ok, text, _ = self.run_analyze(
            raises=subprocess.TimeoutExpired("claude", 1))
        self.assertFalse(ok)
        self.assertIn("timed out", text)


if __name__ == "__main__":
    unittest.main()
