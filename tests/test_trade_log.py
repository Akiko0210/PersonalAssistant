"""Tom's trade log: the thinkorswim fill grammar, the statement import and its
tie-out, alert/statement reconciliation, minimal zero-net position grouping,
and the tools' honest-failure paths (unparseable lines, which account?)."""

import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from brain import agents
from tools import ToolContext, dispatch
from trading import config as tcfg
from trading import statement as tstatement
from trading import trade_log
from trading.trade_positions import group_positions

# A real alert (2026-09-25), market-data noise and all.
ALERT = ("#131646602945 BOT +1 SPX 100 (Weeklys) 25 SEP 26 7600 PUT @1.85LAST="
         "7715.14 BID=7713.04 ASK=7717.21 MARK=7715.14 VOL INDEX=null IMPL "
         "VOL=15.92% DELTA=null NEWS=null , ACCOUNT *****868SCHW")
ALERT_DATE = "Fri, 25 Sep 2026 12:03:11 -0400"

# Rows lifted from the 2026-09-20 statement: a butterfly closed by its three
# expirations, one order filled in two split rows, and a non-trade cash row.
STATEMENT = """\
Account Statement for *****699SCHW (Rollover IRA) since 12/31/25 through 9/20/26

Cash Balance
DATE,TIME,TYPE,REF #,DESCRIPTION,Misc Fees,Commissions & Fees,AMOUNT,BALANCE
4/2/26,09:40:16,TRD,="1005898731860",BOT +1 BUTTERFLY SPX 100 (Weeklys) 10 APR 26 6700/6600/6500 PUT @18.75 CBOE,-2.28,-2.60,"-1,875.00","33,016.95"
4/10/26,23:25:44,RAD,="116357787992",Removed due to Expiration PUT S & P 500 INDEX $6500 EXP 04/10/26: EXP: -1.0 .SPXW260410P6500,,,,"33,016.95"
4/10/26,23:25:55,RAD,="116357788001",Removed due to Expiration PUT S & P 500 INDEX $6700 EXP 04/10/26: EXP: -1.0 .SPXW260410P6700,,,,"33,016.95"
4/10/26,23:26:06,RAD,="116357788015",Removed due to Expiration PUT S & P 500 INDEX $6600 EXP 04/10/26: EXP: 2.0 .SPXW260410P6600,,,,"33,016.95"
6/5/26,09:20:40,TRD,="1006632596914",BOT +2 SPX 100 (Weeklys) 5 JUN 26 7500 CALL @2.90 CBOE,-1.14,-1.30,-580.00,"23,191.96"
6/5/26,09:20:40,TRD,="1006632596914",BOT +2 SPX 100 (Weeklys) 5 JUN 26 7500 CALL @2.90 CBOE,-1.15,-1.30,-580.00,"22,609.51"
6/30/26,00:00:00,DOI,="999",INTEREST,,,1.25,"22,610.76"
,,,,TOTAL,($4.57),($5.20),"($3,033.75)","$22,610.76"
"""


class TestParseFill(unittest.TestCase):
    CASES = [  # line -> legs (expiration, strike, type, signed qty), cash
        ("BOT +1 SPX 100 (Weeklys) 20 FEB 26 7040 CALL @.30 CBOE",
         [("2026-02-20", 7040, "C", 1)], -30.0),
        ("BOT +1 VERTICAL SPX 100 (Weeklys) 6 MAR 26 6870/6880 CALL @7.70 CBOE",
         [("2026-03-06", 6870, "C", 1), ("2026-03-06", 6880, "C", -1)], -770.0),
        ("BOT +2 CALENDAR SPX 100 (Weeklys) 3 FEB 26/30 JAN 26 6860 PUT @7.00 CBOE",
         [("2026-02-03", 6860, "P", 2), ("2026-01-30", 6860, "P", -2)], -1400.0),
        ("SOLD -2 DIAGONAL SPX 100 (Weeklys) 24 FEB 26/20 FEB 26 6830/6810 PUT @27.20 CBOE",
         [("2026-02-24", 6830, "P", -2), ("2026-02-20", 6810, "P", 2)], 5440.0),
        ("SOLD -1 BUTTERFLY SPX 100 (Weeklys) 6 MAR 26 6840/6940/7030 CALL @32.15 CBOE",
         [("2026-03-06", 6840, "C", -1), ("2026-03-06", 6940, "C", 2),
          ("2026-03-06", 7030, "C", -1)], 3215.0),
        ("SOLD -1 CONDOR SPX 100 (Weeklys) 6 MAR 26 6770/6870/6880/6970 CALL @36.05 CBOE",
         [("2026-03-06", 6770, "C", -1), ("2026-03-06", 6870, "C", 1),
          ("2026-03-06", 6880, "C", 1), ("2026-03-06", 6970, "C", -1)], 3605.0),
        ("SOLD -3 DBL DIAG SPX 100 (Weeklys) 20 JAN 26/16 JAN 26 "
         "6990/6830/6990/6830 CALL/PUT/CALL/PUT @12.05 CBOE",
         [("2026-01-20", 6990, "C", -3), ("2026-01-20", 6830, "P", -3),
          ("2026-01-16", 6990, "C", 3), ("2026-01-16", 6830, "P", 3)], 3615.0),
        ("SOLD -4 DBL DIAG SPX 100 (Quarterlys) 30 JUN 26/26 JUN 26 "
         "7520/7350/7480/7340 PUT @67.75 CBOE",
         [("2026-06-30", 7520, "P", -4), ("2026-06-30", 7350, "P", -4),
          ("2026-06-26", 7480, "P", 4), ("2026-06-26", 7340, "P", 4)], 27100.0),
    ]

    def test_statement_lines(self):
        for line, legs, cash in self.CASES:
            with self.subTest(line=line):
                fill = trade_log.parse_fill(line)
                self.assertEqual(fill["legs"], legs)
                self.assertEqual(fill["amount"], cash)

    def test_alert_line_stops_at_the_price(self):
        fill = trade_log.parse_fill(ALERT.split(" ", 1)[1])
        self.assertEqual(fill["legs"], [("2026-09-25", 7600, "P", 1)])
        self.assertEqual(fill["amount"], -185.0)
        self.assertTrue(fill["text"].endswith("@1.85"))

    def test_unknown_lines_are_refused_not_guessed(self):
        for line in (
                "BOT +1 IRON CONDOR SPX 100 (Weeklys) 25 SEP 26 "
                "7800/7850/7500/7450 CALL/PUT @3.00 CBOE",
                "BOT +2 BUTTERFLY /ESZ26 1/50 SEP 26 (Wk5) /EW5U26 "
                "7750/7650/7550 PUT @21.25 CME",
                "BOT -1 SPX 100 (Weeklys) 20 FEB 26 7040 CALL @.30 CBOE"):
            with self.subTest(line=line):
                self.assertIsNone(trade_log.parse_fill(line))


class TestStatement(unittest.TestCase):
    def test_orders_merge_split_fills_and_read_expirations(self):
        st = tstatement.parse(STATEMENT)
        self.assertEqual(st["problems"], [])
        self.assertEqual((st["account"], st["since"], st["skipped"]),
                         ("*****699SCHW", "2025-12-31", 1))
        self.assertEqual(len(st["orders"]), 5)
        split = st["orders"][-1]
        self.assertEqual((split["quantity"], split["fills"], split["amount"],
                          split["fees"]), (4, 2, -1160.0, -4.89))
        self.assertEqual(split["legs"], [("2026-06-05", 7500, "C", 4)])
        body = st["orders"][3]
        self.assertEqual((body["symbol"], body["strategy"], body["legs"]),
                         ("SPX", "expiration", [("2026-04-10", 6600, "P", 2)]))

    def test_a_misread_total_refuses_the_import(self):
        st = tstatement.parse(STATEMENT.replace("($4.57)", "($4.00)"))
        self.assertIn("TOTAL", st["problems"][0])


class LogBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = tcfg.TRADE_LOG_PATH
        tcfg.TRADE_LOG_PATH = Path(self.tmp.name) / "trade_log.db"
        self.csv = Path(self.tmp.name) / "2026-09-20-AccountStatement.csv"
        self.csv.write_text(STATEMENT, encoding="utf-8")
        self.ctx = ToolContext(active_agent="tom")

    def tearDown(self):
        tcfg.TRADE_LOG_PATH = self._saved
        self.tmp.cleanup()

    def call(self, name, **args):
        return dispatch(self.ctx, name, args)


class TestTools(LogBase):
    def test_log_trade_from_an_alert_is_idempotent(self):
        self.assertIn("Logged", self.call("log_trade", alert=ALERT,
                                          email_date=ALERT_DATE))
        again = self.call("log_trade", alert=ALERT, email_date=ALERT_DATE)
        self.assertIn("Already in the log", again)
        [t] = trade_log.query()
        local = datetime(2026, 9, 25, 12, 3, 11,
                         tzinfo=timezone(timedelta(hours=-4))).astimezone()
        self.assertEqual((t["account"], t["ref"], t["amount"], t["fees"]),
                         ("*****868SCHW", "131646602945", -185.0, None))
        self.assertEqual(t["executed_at"],
                         local.replace(tzinfo=None).isoformat(timespec="seconds"))

    def test_statement_supersedes_alert_rows_in_its_span_only(self):
        fill = trade_log.parse_fill(
            "BOT +2 SPX 100 (Weeklys) 5 JUN 26 7500 CALL @2.90")
        for ref, when in (("1", "2026-06-05T09:21:00"),    # inside the grace
                          ("2", "2026-06-05T12:00:00")):   # after the statement
            trade_log.save([trade_log.order(
                fill, account="*****699SCHW", ref=ref, executed_at=when,
                amount=fill["amount"], fees=None, description="x",
                source="email")])
        reply = self.call("import_statement", path=str(self.csv))
        self.assertIn("ties out", reply)
        self.assertIn("Replaced 1 alert-logged", reply)
        self.call("import_statement", path=str(self.csv))       # repeatable
        refs = [t["ref"] for t in trade_log.query()]
        self.assertEqual(len(refs), 6)
        self.assertIn("2", refs)
        self.assertNotIn("1", refs)

    def test_unparseable_alert_logs_nothing(self):
        reply = self.call("log_trade", email_date=ALERT_DATE, alert=(
            "#1 BOT +1 IRON CONDOR SPX 100 25 SEP 26 1/2/3/4 CALL/PUT @3 , "
            "ACCOUNT *****868SCHW"))
        self.assertIn("nothing was logged", reply)
        self.assertEqual(trade_log.query(), [])

    def test_readers_ask_which_account_when_several(self):
        self.call("import_statement", path=str(self.csv))
        self.call("log_trade", alert=ALERT, email_date=ALERT_DATE)
        ask = self.call("query_trade_log")
        self.assertIn("Ask the user which one", ask)
        self.assertIn("*****868SCHW", ask)
        one = self.call("query_trade_log", account="868")
        self.assertIn("131646602945", one)
        self.assertIn("fees still pending on 1", one)

    def test_positions_price_closed_and_flag_expired_leftovers(self):
        self.call("import_statement", path=str(self.csv))
        out = self.call("trade_log_positions", account="699")
        self.assertIn("P&L -1,879.88", out)     # the fly, closed by expiry
        self.assertIn("unresolved", out)        # 7500C: expired, not logged
        closed = self.call("trade_log_positions", account="699",
                           status="closed")
        self.assertNotIn("unresolved", closed)

    def test_tom_writes_linda_only_reads(self):
        readers = {"query_trade_log", "trade_log_positions"}
        writers = {"log_trade", "import_statement"}
        self.assertLessEqual(readers | writers, agents.AGENTS["tom"]["tools"])
        self.assertLessEqual(readers, agents.AGENTS["linda"]["tools"])
        self.assertFalse(writers & agents.AGENTS["linda"]["tools"])


def _t(ref, when, legs, amount):
    return {"ref": ref, "executed_at": when, "symbol": "SPX", "amount": amount,
            "fees": -1.0, "strategy": "diagonal", "legs": legs}


class TestGrouping(unittest.TestCase):
    def test_equal_candidates_pair_fifo_into_minimal_positions(self):
        # The 2/13-2/19 diagonals: two buys, two sells, each sell able to
        # close either buy. Minimal sets are pairs, oldest buy first.
        buy = [("2026-02-24", 6830, "P", 2), ("2026-02-20", 6810, "P", -2)]
        sell = [(e, k, cp, -q) for e, k, cp, q in buy]
        ps = group_positions([_t("b1", "2026-02-13", buy, -3840),
                              _t("b2", "2026-02-17", buy, -4230),
                              _t("s1", "2026-02-19T12", sell, 5440),
                              _t("s2", "2026-02-19T13", sell, 5130)],
                             today=date(2026, 9, 25))
        self.assertEqual([[t["ref"] for t in p["trades"]] for p in ps],
                         [["b1", "s1"], ["b2", "s2"]])
        self.assertEqual([p["net"] for p in ps], [1598.0, 898.0])

    def test_leftovers_are_open_until_every_leg_has_expired(self):
        cal = [_t("c", "2026-09-18", [("2026-10-01", 7550, "P", 1),
                                      ("2026-09-25", 7550, "P", -1)], -1500)]
        self.assertEqual(group_positions(cal, today=date(2026, 9, 25))[0]
                         ["status"], "open")
        self.assertEqual(group_positions(cal, today=date(2026, 10, 2))[0]
                         ["status"], "unresolved")

    def test_search_cap_groups_a_tangle_honestly(self):
        legs = [("2026-10-01", 7600, "P", 1)]
        saved, tcfg.POSITION_SEARCH_MAX_TRADES = tcfg.POSITION_SEARCH_MAX_TRADES, 1
        try:
            ps = group_positions([_t(str(i), f"2026-09-0{i}", legs, -100)
                                  for i in range(1, 4)], today=date(2026, 9, 25))
        finally:
            tcfg.POSITION_SEARCH_MAX_TRADES = saved
        self.assertEqual(len(ps), 1)
        self.assertTrue(ps[0]["tangled"])


if __name__ == "__main__":
    unittest.main()
