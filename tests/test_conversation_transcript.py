"""Tests for conversation-note transcripts.

The guarantee under test: a note saved FROM CONVERSATION (the
save_conversation_note flow) gets a transcript containing the actual spoken
exchange it was drawn from — not a second copy of the model-authored note
body. Early versions wrote the note content into both files, so transcript ==
summary and the user's own words (the one thing that can't be regenerated)
were silently discarded.
"""

import logging
import unittest
from types import SimpleNamespace

from llm import Claude
from voice_agent import Agent


class TestConversationExcerpt(unittest.TestCase):
    def make(self, history):
        c = Claude.__new__(Claude)
        c.history = history
        return c

    def test_flattens_user_and_assistant_text(self):
        c = self.make([
            {"role": "user", "content": "the OT story about Dr. Greenfield"},
            {"role": "assistant", "content": [{"type": "text", "text": "Got it."}]},
        ])
        excerpt = c.conversation_excerpt()
        self.assertIn("user: the OT story about Dr. Greenfield", excerpt)
        self.assertIn("assistant: Got it.", excerpt)

    def test_tool_traffic_is_skipped(self):
        c = self.make([
            {"role": "user", "content": "save that as a note"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "save_conversation_note",
                 "input": {"title": "x", "content": "y"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        ])
        excerpt = c.conversation_excerpt()
        self.assertIn("save that as a note", excerpt)
        self.assertNotIn("tool_use", excerpt)
        self.assertNotIn("save_conversation_note", excerpt)

    def test_empty_history(self):
        self.assertEqual(self.make([]).conversation_excerpt(), "")


class FakeStore:
    def __init__(self):
        self.transcripts = {}
        self.summaries = {}

    def _match_category(self, name):
        return None

    def new_session(self):
        return "note_test"

    def append_transcript(self, note_id, text):
        self.transcripts[note_id] = text

    def save_summary(self, note_id, title, content, category):
        self.summaries[note_id] = (title, content, category)


class TestSavePendingNoteTranscript(unittest.TestCase):
    def make_agent(self, excerpt):
        agent = Agent.__new__(Agent)
        agent.log = logging.getLogger("test")
        agent.store = FakeStore()
        agent.audio = SimpleNamespace(flush=lambda: None)
        agent.say = lambda text, **kw: False
        agent._confirm_category = lambda suggested, title, summary: "general"
        agent.llm = SimpleNamespace(
            conversation_excerpt=lambda: excerpt,
            record_tool_event=lambda text: None,
            flush_tool_events=lambda persist=False: None,
        )
        return agent

    def test_transcript_is_the_conversation_not_the_note(self):
        agent = self.make_agent(
            "user: seven providers came out, mostly useless\n\n"
            "assistant: That sounds exhausting."
        )
        agent._save_pending_note({"title": "OT issues",
                                  "content": "## Summary\nThe polished note."})
        tx = agent.store.transcripts["note_test"]
        self.assertIn("seven providers came out", tx)       # the user's words
        self.assertNotIn("The polished note", tx)           # not the summary
        # ...while the summary file still gets the note body.
        self.assertIn("The polished note", agent.store.summaries["note_test"][1])

    def test_empty_history_falls_back_to_content(self):
        agent = self.make_agent("")
        agent._save_pending_note({"title": "t", "content": "note body"})
        self.assertIn("note body", agent.store.transcripts["note_test"])


if __name__ == "__main__":
    unittest.main()
