"""Exporting any table Tom shows (#53): every display tool takes export=,
and its card — what the screen showed, totals and notes included — is
saved as a CSV. query_trade_log keeps its own every-row export (#50)."""

import csv
import unittest
from unittest import mock

import config as cfg
import tools
from display import card as blocks
from lib import csv_export
from tests.test_trade_log import LogBase


def _read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


class TestCardTable(unittest.TestCase):
    def test_every_row_total_and_note_becomes_a_line(self):
        card = blocks.card(
            [blocks.section("IRA", [blocks.row(
                "SPX 7750 Put · Fri Nov 6", "+120.00", blocks.PNL,
                badge="SHORT", lines=["Quantity 1", "", "avg 3.10"])],
                total=blocks.total("IRA unrealized", "+120.00"),
                notes=["1 position unmarked"]),
             blocks.section("Blue", [blocks.row("Fees", "4.88")])],
            total=blocks.total("All accounts", "+120.00"), notes=["stale"])
        self.assertEqual(blocks.table(card), [
            ("", "All accounts", "TOTAL", "+120.00", ""),
            ("IRA", "IRA unrealized", "TOTAL", "+120.00", ""),
            ("IRA", "SPX 7750 Put · Fri Nov 6", "SHORT", "+120.00",
             "Quantity 1 · avg 3.10"),
            ("IRA", "1 position unmarked", "NOTE", "", ""),
            ("Blue", "Fees", "", "4.88", ""),
            ("", "stale", "NOTE", "", "")])


class TestRegistry(unittest.TestCase):
    def test_every_display_tool_can_export(self):
        for name in tools._DISPLAY:
            schema, _ = tools._REGISTRY[name]
            self.assertIn("export", schema["input_schema"]["properties"], name)
        self.assertIn("trade_log_positions", tools._CARD_EXPORT)
        self.assertIn("get_positions", tools._CARD_EXPORT)
        # Its own export writes every trade, not the card's shown rows.
        self.assertNotIn("query_trade_log", tools._CARD_EXPORT)


class TestCardExport(LogBase):
    def setUp(self):
        super().setUp()
        self.call("import_statement", path=str(self.csv))

    def test_the_position_table_exports_as_shown(self):
        reply = self.call("trade_log_positions", account="699", export=True)
        [path] = cfg.EXPORTS_DIR.glob("trade_log_positions_*.csv")
        rows = _read(path)
        self.assertEqual(tuple(rows[0]), blocks.TABLE_HEADER)
        self.assertTrue(reply.startswith(
            f"Exported {len(rows) - 1} rows to {path.name}"))
        self.assertIn("P&L -1,879.88", reply)       # the answer is still there
        self.assertIn("-1,879.88", [r[3] for r in rows])

    def test_no_file_unless_asked(self):
        self.call("trade_log_positions", account="699")
        self.assertFalse(cfg.EXPORTS_DIR.exists())

    def test_an_answer_with_no_table_says_nothing_was_exported(self):
        reply = self.call("trade_log_positions", account="699", symbol="NDX",
                          export=True)
        self.assertIn("No positions in the log match that", reply)
        self.assertIn("Nothing was exported", reply)
        self.assertFalse(cfg.EXPORTS_DIR.exists())

    def test_a_failed_write_is_said_and_the_answer_kept(self):
        with mock.patch.object(csv_export, "write",
                               side_effect=PermissionError("disk is locked")), \
                self.assertLogs("tools", "ERROR"):
            reply = self.call("trade_log_positions", account="699", export=True)
        self.assertIn("The export failed: disk is locked", reply)
        self.assertIn("P&L -1,879.88", reply)


if __name__ == "__main__":
    unittest.main()
