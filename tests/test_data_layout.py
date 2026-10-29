"""Tests for config.migrate_layout: the one-time move of a flat data/ into
its topic folders (#49). The guarantees: everything lands where the live
paths now look, nothing is overwritten, a second run is a no-op, and a note
folder that shares its name with another data/ folder gives up only its
notes."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config as cfg
from stores import categories


def _layout(root):
    """Every path constant, rebuilt under `root` the way config.py builds it."""
    notes, memory, settings, gmail = (root / "notes", root / "memory",
                                      root / "settings", root / "gmail")
    return {"DATA_DIR": root, "NOTES_DIR": notes,
            "HISTORY_DIR": root / "conversation_history",
            "INDEX_PATH": notes / "index.json",
            "CATEGORIES_PATH": notes / "categories.json",
            "PENDING_DIR": notes / "_pending", "CHROMA_DIR": memory / "chroma",
            "WORKING_STATE_PATH": memory / "working_state.db",
            "AGENTS_PATH": settings / "agents.json",
            "AGENT_STATE_PATH": settings / "agent_state.json",
            "OVERRIDES_PATH": settings / "config_overrides.json",
            "GMAIL_CLIENT_SECRET_PATH": gmail / "client_secret.json",
            "GMAIL_TOKEN_PATH": gmail / "token.json"}


def _write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestMigrateLayout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        patcher = mock.patch.multiple(cfg, **_layout(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)
        saved = {k: dict(v) for k, v in categories.NOTE_CATEGORIES.items()}
        self.addCleanup(lambda: (categories.NOTE_CATEGORIES.clear(),
                                 categories.NOTE_CATEGORIES.update(saved)))
        self.addCleanup(self.tmp.cleanup)

    def _flat_install(self):
        r = self.root
        _write(r / "index.json", '{"note_1": {"category": "general"}}')
        # "trading" files its notes in data/trading, the brokers' own folder —
        # what the seed "Trading" folder is on a case-insensitive disk.
        _write(r / "categories.json", json.dumps({
            "general": {"display": "General", "folder": "General"},
            "trading": {"display": "Trading", "folder": "trading"},
            "coach": {"display": "Coach", "folder": "CoachingNotes"}}))
        _write(r / "General" / "note_1.md", "summary")
        _write(r / "General" / "note_1.transcript.md", "words")
        _write(r / "CoachingNotes" / "note_2.md")
        _write(r / "trading" / "note_3.md", "a trade idea")
        _write(r / "trading" / "schwab" / "tokens.json", "{}")
        _write(r / "pending" / "note_4.md")
        _write(r / "chroma" / "chroma.sqlite3", "db")
        _write(r / "working_state.db", "ws")
        _write(r / "agents.json", "{}")
        _write(r / "config_overrides.json", '{"TTS_RATE": 190}')
        _write(r / "gmail_token.json", "tok")
        _write(r / "history_tom.json", '[{"role": "user", "content": "hi"}]')
        _write(r / "history_alice.json", "[]")

    def test_moves_every_file_to_where_the_live_paths_look(self):
        self._flat_install()
        self.assertGreater(cfg.migrate_layout(), 0)
        n = self.root / "notes"
        self.assertEqual((n / "General" / "note_1.md").read_text(), "summary")
        self.assertTrue((n / "General" / "note_1.transcript.md").exists())
        self.assertTrue((n / "CoachingNotes" / "note_2.md").exists())
        self.assertTrue((cfg.PENDING_DIR / "note_4.md").exists())
        self.assertIn("note_1", json.loads(cfg.INDEX_PATH.read_text()))
        self.assertTrue(cfg.CATEGORIES_PATH.exists())
        self.assertEqual((cfg.CHROMA_DIR / "chroma.sqlite3").read_text(), "db")
        self.assertEqual(cfg.WORKING_STATE_PATH.read_text(), "ws")
        self.assertEqual(json.loads(cfg.OVERRIDES_PATH.read_text()),
                         {"TTS_RATE": 190})
        self.assertTrue(cfg.AGENTS_PATH.exists())
        self.assertEqual(cfg.GMAIL_TOKEN_PATH.read_text(), "tok")
        self.assertEqual(json.loads(cfg.history_path("tom").read_text()),
                         [{"role": "user", "content": "hi"}])
        self.assertTrue(cfg.history_path("alice").exists())
        # Old homes are gone, emptied note folders with them.
        for old in ("index.json", "pending", "chroma", "history_tom.json",
                    "General", "CoachingNotes", "gmail_token.json"):
            self.assertFalse((self.root / old).exists(), old)

    def test_a_shared_folder_gives_up_only_its_notes(self):
        self._flat_install()
        cfg.migrate_layout()
        self.assertEqual((self.root / "notes" / "trading" / "note_3.md")
                         .read_text(), "a trade idea")
        self.assertFalse((self.root / "trading" / "note_3.md").exists())
        self.assertTrue((self.root / "trading" / "schwab" / "tokens.json")
                        .exists())

    def test_a_second_run_moves_nothing(self):
        self._flat_install()
        cfg.migrate_layout()
        self.assertEqual(cfg.migrate_layout(), 0)

    def test_an_empty_destination_made_early_is_filled(self):
        # setup_logging's ensure_dirs() runs before the lock, so it has
        # already made memory/chroma and the note folders, empty.
        self._flat_install()
        cfg.CHROMA_DIR.mkdir(parents=True)
        (self.root / "notes" / "General").mkdir(parents=True)
        cfg.migrate_layout()
        self.assertTrue((cfg.CHROMA_DIR / "chroma.sqlite3").exists())
        self.assertTrue((self.root / "notes" / "General" / "note_1.md").exists())

    def test_never_overwrites_a_destination_with_content(self):
        self._flat_install()
        _write(cfg.history_path("tom"), "[]")       # already there, newer
        with self.assertLogs("agent", "WARNING") as logs:
            cfg.migrate_layout()
        self.assertEqual(cfg.history_path("tom").read_text(), "[]")
        self.assertTrue((self.root / "history_tom.json").exists())  # kept aside
        self.assertIn("history_tom.json", "\n".join(logs.output))

    def test_a_fresh_install_has_nothing_to_move(self):
        self.assertEqual(cfg.migrate_layout(), 0)


if __name__ == "__main__":
    unittest.main()
