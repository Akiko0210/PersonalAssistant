"""Unit tests for Claude._parse_summary (pure text parsing)."""

import unittest

import categories
from llm import Claude


class TestParseSummary(unittest.TestCase):
    def test_full_format(self):
        text = (
            "TITLE: Grocery run\n"
            "SPOKEN: You listed milk and eggs.\n"
            "CATEGORY: general\n"
            "---\n"
            "## Summary\nA grocery list.\n\n## Key Points\n- milk\n- eggs"
        )
        title, spoken, full, category = Claude._parse_summary(text)
        self.assertEqual(title, "Grocery run")
        self.assertEqual(spoken, "You listed milk and eggs.")
        self.assertEqual(category, "general")
        self.assertTrue(full.startswith("## Summary"))

    def test_unknown_category_falls_back_to_default(self):
        text = "TITLE: T\nSPOKEN: S\nCATEGORY: nonsense-slug\n---\nbody"
        _, _, _, category = Claude._parse_summary(text)
        self.assertEqual(category, categories.DEFAULT_CATEGORY)

    def test_missing_header_gets_defaults(self):
        title, spoken, full, category = Claude._parse_summary("just some text")
        self.assertEqual(title, "Untitled note")
        self.assertEqual(spoken, "I've saved your note.")
        self.assertEqual(category, categories.DEFAULT_CATEGORY)
        self.assertTrue(full)


if __name__ == "__main__":
    unittest.main()
