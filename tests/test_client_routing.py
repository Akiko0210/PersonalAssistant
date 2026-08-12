"""Unit tests for Claude.client_for — provider routing to API clients.

Uses a bare shell object instead of a full Claude (whose __init__ loads note
stores and embedding models): client_for only touches .client and ._deepseek,
and that narrow surface is exactly what these tests pin down.
"""

import os
import unittest
from unittest.mock import patch

import config as cfg
from brain.llm import Claude


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


if __name__ == "__main__":
    unittest.main()
