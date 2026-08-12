"""Tests for truncated-reply handling in llm.converse.

The guarantee under test: when a reply hits CONVO_MAX_TOKENS mid-tool-call
(stop_reason "max_tokens"), the tool was NEVER executed — converse must say so
out loud instead of returning "". A silent empty return once cost a whole note
save (session_2026-07-19.log 19:53-20:05): the truncated save_conversation_note
was dropped, sanitize erased the evidence from history, and the model kept
announcing "saving now" for 12 minutes with the user hearing nothing.
"""

import functools
import unittest
from types import SimpleNamespace

from tests.llm_fixtures import FakeBlock, make_claude as _make_claude

# convo_model=None: these tests pin the registry-default resolution path.
make_claude = functools.partial(_make_claude, convo_model=None)


class TestTruncatedReply(unittest.TestCase):
    def test_truncated_tool_call_is_reported_not_silent(self):
        resp = SimpleNamespace(
            stop_reason="max_tokens",
            content=[FakeBlock(type="tool_use", id="t1",
                               name="save_conversation_note",
                               input={"title": "Design doc", "content": "trunca"})],
        )
        reply = make_claude([resp]).converse("write it up and save it as a note")
        self.assertTrue(reply)                    # never a silent empty string
        self.assertIn("did not complete", reply)  # honest: the save did NOT happen

    def test_truncated_text_still_returned_with_notice(self):
        resp = SimpleNamespace(
            stop_reason="max_tokens",
            content=[FakeBlock(type="text", text="Here's the design: first,")],
        )
        reply = make_claude([resp]).converse("explain the design")
        self.assertIn("Here's the design", reply)  # partial text is kept

    def test_normal_reply_unchanged(self):
        resp = SimpleNamespace(
            stop_reason="end_turn",
            content=[FakeBlock(type="text", text="All good.")],
        )
        reply = make_claude([resp]).converse("hello")
        self.assertEqual(reply, "All good.")


if __name__ == "__main__":
    unittest.main()
