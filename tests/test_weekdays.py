"""Weekdays written out next to every date a tool hands the model. Given
bare ISO dates the agents worked weekdays out themselves and got five
expirations wrong in a week ("Oct 20 7650 put expires Monday", a Tuesday;
2026-10-13..20)."""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from lib.dates import with_weekday
from tests.test_trade_log import ALERT, ALERT_DATE, LogBase
from tests.trading_fixtures import spx_chain
from tools import ToolContext, dispatch
from trading.models import Leg
from trading.ticket import Ticket


class TestWithWeekday(unittest.TestCase):
    def test_the_audits_dates(self):
        self.assertEqual(with_weekday("2026-10-20"), "2026-10-20 Tue")
        self.assertEqual(with_weekday("2026-10-21"), "2026-10-21 Wed")

    def test_junk_comes_back_unchanged_without_a_guessed_weekday(self):
        self.assertEqual(with_weekday("Oct 20"), "Oct 20")
        self.assertIsNone(with_weekday(None))


class TestTradeLogLine(LogBase):
    def test_fill_time_and_leg_carry_their_weekdays(self):
        self.call("log_trade", alert=ALERT, email_date=ALERT_DATE)
        out = self.call("query_trade_log", account="868")
        local = datetime(2026, 9, 25, 12, 3,
                         tzinfo=timezone(timedelta(hours=-4))).astimezone()
        self.assertIn(f"131646602945 {local:%Y-%m-%d %a %H:%M} SPX", out)
        self.assertIn("| +1 2026-09-25 Fri 7600P", out)


class TestExpirationList(unittest.TestCase):
    def test_each_expiration_names_its_weekday(self):
        eng = SimpleNamespace(chain=lambda symbol: spx_chain())
        with mock.patch("tools.trading_tools._engine", return_value=(eng, None)):
            out = dispatch(ToolContext(active_agent="tom"), "list_expirations",
                           {"symbol": "SPX", "count": 1})
        self.assertIn("Nearest SPX expirations: 2026-07-24 Fri.", out)


class TestTicketSummary(unittest.TestCase):
    def test_leg_names_its_expirations_weekday(self):
        tk = Ticket(underlying="SPX", legs=[
            Leg(7650.0, "Put", "2026-10-20", "Short")])
        self.assertIn("STO 1 2026-10-20 Tue 7650 Put", tk.describe())


if __name__ == "__main__":
    unittest.main()
