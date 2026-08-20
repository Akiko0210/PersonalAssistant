"""Unit tests for what leaves for the API: client_for's provider routing, and
cached()'s wire copy of a stored history.

Uses a bare shell object instead of a full Claude (whose __init__ loads note
stores and embedding models): client_for only touches .client and ._deepseek,
and that narrow surface is exactly what these tests pin down.
"""

import os
import unittest
from unittest.mock import patch

import config as cfg
from brain.llm import Claude, cached


class _Shell:
    """Just the attributes client_for reads/writes."""
    def __init__(self):
        self.client = object()
        self._deepseek = None


class TestClientFor(unittest.TestCase):
    def test_claude_ids_use_the_shared_client(self):
        shell = _Shell()
        for mid in ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"):
            self.assertIs(Claude.client_for(shell, mid), shell.client)
        self.assertIsNone(shell._deepseek)  # never built for Claude models

    def test_deepseek_without_key_raises_with_guidance(self):
        shell = _Shell()
        env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                Claude.client_for(shell, "deepseek-v4-flash")
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))

    def test_deepseek_client_built_once_and_reused(self):
        shell = _Shell()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            first = Claude.client_for(shell, "deepseek-v4-flash")
            second = Claude.client_for(shell, "deepseek-v4-pro")
        self.assertIs(first, second)          # one client serves both models
        self.assertIsNot(first, shell.client)  # and it isn't the Anthropic one
        self.assertEqual(str(first.base_url).rstrip("/"),
                         cfg.DEEPSEEK_BASE_URL)


class TestCached(unittest.TestCase):
    """A persisted message carries a local `ts` the Messages API would reject
    as an unknown field, so the wire copy must keep role and content only."""

    HISTORY = [
        {"role": "user", "content": "hi", "ts": "2026-08-11T09:00:00+09:00"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}],
         "ts": "2026-08-11T09:00:04+09:00"},
    ]

    def test_only_role_and_content_go_out(self):
        for m in cached(self.HISTORY):
            self.assertEqual(set(m), {"role", "content"})

    def test_last_block_still_carries_the_cache_breakpoint(self):
        wire = cached(self.HISTORY)
        self.assertIn("cache_control", wire[-1]["content"][-1])

    def test_stored_history_is_left_alone(self):
        snapshot = [dict(m) for m in self.HISTORY]
        cached(self.HISTORY)
        self.assertEqual(self.HISTORY, snapshot)

    def test_empty(self):
        self.assertEqual(cached([]), [])

    def test_spoken_user_turn_carries_its_time_as_text(self):
        # The ts field itself never goes out; it is rendered into the text so
        # the model can see time pass between messages (the 9am "nothing to do
        # tonight" failure, session_2026-08-11.log 2026-08-20 09:04).
        wire = cached(self.HISTORY)
        self.assertEqual(wire[0]["content"], "(9:00am 8/11/2026) hi")

    def test_assistant_and_unstamped_messages_stay_bare(self):
        wire = cached([
            {"role": "user", "content": "old"},  # pre-feature: no ts
            {"role": "assistant", "content": "sure", "ts": "2026-08-11T09:00:04+09:00"},
            {"role": "user", "content": [  # tool_result turn: blocks, not speech
                {"type": "tool_result", "tool_use_id": "t", "content": "ok"}],
             "ts": "2026-08-11T09:00:05+09:00"},
        ])
        self.assertEqual(wire[0]["content"], "old")
        self.assertEqual(wire[1]["content"], "sure")
        self.assertEqual(wire[2]["content"][0]["type"], "tool_result")


if __name__ == "__main__":
    unittest.main()
