"""Tests for tool-activity events being folded into conversation history.

The guarantee under test: work a tool does beyond its return string — a deferred
save, or a sub-dialogue that runs in separate model memory — is recorded and
folded into the main history as an assistant self-note, so the next turn's model
knows what its tools actually did (rather than answering from the stale
placeholder the tool returned mid-turn).

Claude is built via __new__ so its heavy __init__ (API client, stores, embedding
model) is skipped; the methods here only touch _ctx, history, and _save_history.
"""

import unittest

import history as hist
from llm import Claude
from tools import ToolContext


def make_claude(history=None):
    c = Claude.__new__(Claude)
    c._ctx = ToolContext()
    c.history = list(history or [])
    c.saved = []
    c._save_history = lambda: c.saved.append(list(c.history))
    return c


class TestRecordEvent(unittest.TestCase):
    def test_record_and_empty(self):
        ctx = ToolContext()
        self.assertEqual(ctx.events, [])
        ctx.record_event("did a thing")
        ctx.record_event("")        # empty is ignored
        ctx.record_event(None)      # None is ignored
        self.assertEqual(ctx.events, ["did a thing"])


class TestFlushToolEvents(unittest.TestCase):
    def test_no_events_is_noop(self):
        c = make_claude(history=[{"role": "user", "content": "hi"}])
        c.flush_tool_events()
        self.assertEqual(c.history, [{"role": "user", "content": "hi"}])
        self.assertEqual(c.saved, [])  # nothing persisted when nothing to fold

    def test_event_folded_as_assistant_note(self):
        c = make_claude(history=[{"role": "user", "content": "save that"}])
        c.record_tool_event("filed the note into General")
        c.flush_tool_events()
        self.assertEqual(c.history[-1]["role"], "assistant")
        self.assertIn("filed the note into General", c.history[-1]["content"])
        self.assertEqual(c._ctx.events, [])  # buffer cleared after flush

    def test_persist_saves_history(self):
        c = make_claude(history=[{"role": "user", "content": "save that"}])
        c.record_tool_event("filed into General")
        c.flush_tool_events(persist=True)
        self.assertEqual(len(c.saved), 1)

    def test_no_persist_does_not_save(self):
        c = make_claude(history=[{"role": "user", "content": "save that"}])
        c.record_tool_event("filed into General")
        c.flush_tool_events(persist=False)
        self.assertEqual(c.saved, [])

    def test_note_coalesces_into_preceding_assistant_reply(self):
        # After a turn the history ends on the assistant's reply; the self-note
        # is a second assistant turn, which sanitize folds into one so roles keep
        # alternating and the API stays happy.
        c = make_claude(history=[
            {"role": "user", "content": "save that as a note"},
            {"role": "assistant", "content": [{"type": "text", "text": "Sure, saving it."}]},
        ])
        c.record_tool_event("filed into General as note_123")
        c.flush_tool_events()
        # Exactly one assistant turn remains after the user turn.
        roles = [m["role"] for m in c.history]
        self.assertEqual(roles, ["user", "assistant"])
        blocks = c.history[-1]["content"]
        joined = " ".join(b["text"] for b in blocks if b.get("type") == "text")
        self.assertIn("Sure, saving it.", joined)
        self.assertIn("filed into General as note_123", joined)

    def test_folded_history_survives_sanitize_roundtrip(self):
        # A downstream save() re-sanitizes; the folded history must stay valid.
        c = make_claude(history=[{"role": "user", "content": "save that"}])
        c.record_tool_event("filed into General")
        c.flush_tool_events()
        # Idempotent: sanitizing again changes nothing structurally.
        self.assertEqual(hist.sanitize(c.history), c.history)


if __name__ == "__main__":
    unittest.main()
