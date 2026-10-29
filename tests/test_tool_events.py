"""Tests for tool-activity events being folded into conversation history.

The guarantee under test: work a tool does beyond its return string — a deferred
save, or a sub-dialogue that runs in separate model memory — is recorded and
folded into the main history as an assistant self-note, so the next turn's model
knows what its tools actually did (rather than answering from the stale
placeholder the tool returned mid-turn).

Claude is built via __new__ so its heavy __init__ (API client, stores, embedding
model) is skipped; the methods here only touch _ctx, history, and _save_history.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config as cfg
from brain import history as hist
from tools import ToolContext
from tests.llm_fixtures import make_claude


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


class TestNoteTo(unittest.TestCase):
    """A dashboard event lands in the thread of the persona it's for (#38)."""

    def test_the_active_persona_gets_it_in_memory_and_saved(self):
        c = make_claude(active="tom", history=[{"role": "user", "content": "hi"}])
        c.note_to("tom", "statement imported")
        self.assertIn("statement imported", c.history[-1]["content"])
        self.assertEqual(len(c.saved), 1)

    def test_an_inactive_persona_gets_it_in_its_saved_thread(self):
        c = make_claude(active="alice", history=[{"role": "user", "content": "hi"}])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(cfg, "HISTORY_DIR", Path(tmp)):
            hist.save(cfg.history_path("tom"), [
                {"role": "user", "content": "any fills?"},
                {"role": "assistant", "content": "None today."}])
            c.note_to("tom", "statement imported")
            tom = hist.load(cfg.history_path("tom"))
        self.assertEqual(len(tom), 2)        # folded into Tom's last reply
        self.assertIn("statement imported", tom[-1]["content"])
        self.assertEqual(c.history, [{"role": "user", "content": "hi"}])
        self.assertEqual(c.saved, [])        # Alice's thread is untouched


if __name__ == "__main__":
    unittest.main()
