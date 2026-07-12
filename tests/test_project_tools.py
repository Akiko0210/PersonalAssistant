"""Unit tests for the describe_project tool."""

import unittest

import config as cfg
from tools import ToolContext, dispatch, api_tools
import tools.project_tools as project_tools


class TestDescribeProject(unittest.TestCase):
    def test_registered(self):
        names = {t["name"] for t in api_tools()}
        self.assertIn("describe_project", names)

    def test_returns_project_doc(self):
        out = dispatch(ToolContext(), "describe_project", {})
        # The real PROJECT.md ships with the repo; the tool should return it.
        self.assertIn("Voice AI Notetaking Agent", out)
        self.assertIn("tool registry", out.lower())

    def test_topic_arg_is_optional_and_ignored_for_content(self):
        a = dispatch(ToolContext(), "describe_project", {})
        b = dispatch(ToolContext(), "describe_project", {"topic": "barge-in"})
        self.assertEqual(a, b)

    def test_missing_doc_degrades_gracefully(self):
        # Point the tool at a nonexistent path and bypass the session cache.
        project_tools._CACHE["text"] = None
        original = cfg.PROJECT_DOC_PATH
        try:
            cfg.PROJECT_DOC_PATH = cfg.BASE_DIR / "does-not-exist.md"
            out = dispatch(ToolContext(), "describe_project", {})
            self.assertIn("unavailable", out.lower())
        finally:
            cfg.PROJECT_DOC_PATH = original
            project_tools._CACHE["text"] = None  # don't poison other tests


if __name__ == "__main__":
    unittest.main()
