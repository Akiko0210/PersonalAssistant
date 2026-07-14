"""Unit tests for history.py — the API-invariant repair logic.

These encode the exact failure that once bricked the app: a history saved
mid-tool-loop with an unanswered tool_use, which replays into a 400 on every
launch. Run with:  python -m unittest discover tests
"""

import os
import tempfile
import unittest

import history


def user(text):
    return {"role": "user", "content": text}


def assistant_text(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def assistant_tool_use(tid):
    return {"role": "assistant",
            "content": [{"type": "tool_use", "id": tid, "name": "t", "input": {}}]}


def tool_result(tid):
    return {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tid, "content": "ok"}]}


class TestSanitize(unittest.TestCase):
    def test_valid_history_unchanged(self):
        h = [user("hi"), assistant_tool_use("a"), tool_result("a"),
             assistant_text("done"), user("thanks")]
        self.assertEqual(history.sanitize(h), h)

    def test_orphaned_tool_use_dropped_and_users_merged(self):
        # The exact shape that caused the 400-brick: tool_use with no result,
        # then the user spoke again.
        h = [user("Yes."), assistant_tool_use("x"), user("Are you done?")]
        s = history.sanitize(h)
        self.assertEqual(len(s), 1)
        self.assertEqual(s[0]["role"], "user")
        self.assertEqual(s[0]["content"], "Yes.\n\nAre you done?")

    def test_orphaned_tool_result_dropped(self):
        h = [user("hi"), tool_result("ghost"), assistant_text("hello")]
        s = history.sanitize(h)
        self.assertEqual([m["role"] for m in s], ["user", "assistant"])
        self.assertEqual(s[0]["content"], "hi")

    def test_mixed_block_message_keeps_good_blocks(self):
        h = [user("hi"),
             {"role": "assistant", "content": [
                 {"type": "text", "text": "let me check"},
                 {"type": "tool_use", "id": "x", "name": "t", "input": {}},
             ]},
             user("hello?")]
        s = history.sanitize(h)
        # the text block survives; the orphaned tool_use is dropped
        self.assertEqual(len(s), 3)
        self.assertEqual(s[1]["content"], [{"type": "text", "text": "let me check"}])

    def test_adjacent_assistant_turns_merge_as_blocks(self):
        h = [user("hi"), assistant_text("one"), assistant_text("two")]
        s = history.sanitize(h)
        self.assertEqual(len(s), 2)
        self.assertEqual(len(s[1]["content"]), 2)

    def test_never_mutates_input(self):
        h = [user("Yes."), assistant_tool_use("x"), user("done?")]
        snapshot = [dict(m) for m in h]
        history.sanitize(h)
        self.assertEqual(h, snapshot)

    def test_empty(self):
        self.assertEqual(history.sanitize([]), [])


class TestTrim(unittest.TestCase):
    def test_starts_on_plain_user_message(self):
        h = [user("a"), assistant_tool_use("t1"), tool_result("t1"),
             assistant_text("r"), user("b"), assistant_text("r2")]
        t = history.trim(h, 4)  # window lands on the tool_result
        self.assertTrue(t[0]["role"] == "user" and isinstance(t[0]["content"], str))
        self.assertEqual(t[0]["content"], "b")

    def test_short_history_kept_whole(self):
        h = [user("a"), assistant_text("r")]
        self.assertEqual(history.trim(h, 40), h)


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        for name in os.listdir(self.dir):
            try:
                os.unlink(os.path.join(self.dir, name))
            except OSError:
                pass
        try:
            os.rmdir(self.dir)
        except OSError:
            pass

    def test_roundtrip(self):
        from pathlib import Path
        path = Path(self.dir) / "history.json"
        h = [user("hi"), assistant_text("hello"), user("bye")]
        history.save(path, h)
        self.assertEqual(history.load(path), h)

    def test_load_missing_returns_empty(self):
        from pathlib import Path
        self.assertEqual(history.load(Path(self.dir) / "nope.json"), [])

    def test_save_is_atomic_no_temp_left(self):
        from pathlib import Path
        history.save(Path(self.dir) / "history.json", [user("x")])
        leftovers = [n for n in os.listdir(self.dir) if n.startswith(".tmp-")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
