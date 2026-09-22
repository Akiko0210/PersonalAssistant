"""Tests for query understanding (brain/query.py): the model's JSON becomes a
query, a window and known entities; anything else — an exception, prose, a
bad date, an invented key — falls back to what every turn retrieved on
before: the raw utterance."""

import json
import unittest

from brain.query import Query, understand
from lib.dates import window_epochs

NOW = "2026-09-18T09:35:32-07:00"


class TestUnderstand(unittest.TestCase):
    def test_the_reply_becomes_query_window_and_entities(self):
        reply = json.dumps({"query": "the user's to-do list for 2026-09-18",
                            "since": "2026-09-18T09:00", "until": "2026-09-18T10:00",
                            "entities": ["todo:2026-09-18", "made:up"]})
        q = understand(lambda s, p: reply, "what's on my list", ["user: hi"],
                       now=NOW, known_entities=["todo:2026-09-18", "person:tom"])
        self.assertEqual(q.text, "the user's to-do list for 2026-09-18")
        self.assertEqual(q.window, window_epochs(since="2026-09-18T09:00",
                                                 until="2026-09-18T10:00"))
        self.assertEqual(q.entities, ("todo:2026-09-18",))  # never an invented key
        self.assertEqual(q.source, "model")

    def test_no_time_words_means_no_window(self):
        q = understand(lambda s, p: '{"query": "SPX", "since": null, "until": null}',
                       "how did SPX do", [], now=NOW)
        self.assertIsNone(q.window)

    def test_failures_fall_back_to_the_utterance(self):
        def boom(s, p):
            raise TimeoutError("slow")
        self.assertEqual(understand(boom, "what about the other one?", [], now=NOW),
                         Query(text="what about the other one?"))
        q = understand(lambda s, p: "Happy to help! What would you like?",
                       "hello there", [], now=NOW)
        self.assertEqual((q.text, q.window, q.source), ("hello there", None, "raw"))

    def test_a_malformed_date_is_no_date(self):
        q = understand(lambda s, p: '{"query": "x", "since": "this morning"}',
                       "x", [], now=NOW)
        self.assertEqual((q.text, q.window), ("x", None))

    def test_known_keys_are_shown_dated_most_recent_first(self):
        seen = {}

        def ask(system, prompt):
            seen["prompt"] = prompt
            return '{"query": "my list", "entities": ["todo:list-2026-09-17"]}'
        q = understand(ask, "tell me about my list", [], now=NOW,
                       known_entities={"todo:list-2026-09-17": "2026-09-17T09:45:23-07:00",
                                       "todo:list": "2026-09-15T09:06:00-07:00"})
        self.assertIn("todo:list-2026-09-17 (as of 9:45am 9/17/2026), "
                      "todo:list (as of 9:06am 9/15/2026)", seen["prompt"])
        self.assertEqual(q.entities, ("todo:list-2026-09-17",))

    def test_the_prompt_carries_the_recent_turns_time_and_keys(self):
        seen = {}

        def ask(system, prompt):
            seen["prompt"] = prompt
            return "{}"
        understand(ask, "and the other one?",
                   ["user: how did the SPX butterfly do", "assistant: fine"],
                   now=NOW, known_entities=["trade:spx-butterfly"])
        self.assertIn("user: how did the SPX butterfly do", seen["prompt"])
        self.assertIn(NOW, seen["prompt"])
        self.assertIn("trade:spx-butterfly", seen["prompt"])
        self.assertTrue(seen["prompt"].rstrip().endswith("and the other one?"))


if __name__ == "__main__":
    unittest.main()
