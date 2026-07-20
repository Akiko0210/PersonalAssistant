"""Tests for notes.parse_frontmatter — the parser resync's orphan adoption
relies on to rebuild an index entry from a note file alone."""

import unittest

from notes import parse_frontmatter


class TestParseFrontmatter(unittest.TestCase):
    def test_save_summary_format_roundtrips(self):
        text = ("---\ntitle: Audio architecture\ndate: 2026-07-19T20:19:05\n"
                "id: note_2026-07-19_201905\ncategory: ideas\n---\n\n## Summary\nBody here.\n")
        fields, body = parse_frontmatter(text)
        self.assertEqual(fields["title"], "Audio architecture")
        self.assertEqual(fields["date"], "2026-07-19T20:19:05")
        self.assertEqual(fields["category"], "ideas")
        self.assertEqual(body.strip(), "## Summary\nBody here.")

    def test_title_containing_colon(self):
        fields, _ = parse_frontmatter("---\ntitle: Plan: phase two\n---\nx")
        self.assertEqual(fields["title"], "Plan: phase two")

    def test_no_frontmatter_returns_whole_text(self):
        fields, body = parse_frontmatter("just a plain note body")
        self.assertEqual(fields, {})
        self.assertEqual(body, "just a plain note body")

    def test_malformed_frontmatter_is_tolerated(self):
        fields, body = parse_frontmatter("---\nno closing fence\n\nbody")
        self.assertEqual(fields, {})
        self.assertIn("body", body)


if __name__ == "__main__":
    unittest.main()
