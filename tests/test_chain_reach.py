"""Reaching past the loaded chain (#32). Schwab's whole SPX chain is too big
for its API (502 TooBigBody), so the normal load is a week at 80 strikes;
a later date or a far strike is fetched on demand, and only a contract that
isn't listed at all is refused — never snapped to the edge of what's loaded.
"""

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

from tests.test_schwab import SchwabBase, TestAccountsAndOrders
from tests.trading_fixtures import FakeEngine, FakeResponse, WindowedBroker, spx_chain
from tools import ToolContext, dispatch
from tools import trading_tools
from trading import chains as tchains
from trading import config as tcfg
from trading import schwab
from trading import ticket as tticket

LISTED = ["2026-10-21", "2026-10-22", "2026-11-20", "2027-02-19"]


class ReachTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = tcfg.TRADING_DIR
        tcfg.TRADING_DIR = Path(self.tmp.name)
        tchains.clear_cache()
        # Loaded: Oct 21-22 at 6340..6360. Listed: four expirations, 6300..6400.
        self.broker = WindowedBroker(spx_chain(expirations=LISTED))
        self.eng = FakeEngine("schwab", variant=self.broker, price=6350.0)

    def tearDown(self):
        tcfg.TRADING_DIR = self._saved
        tchains.clear_cache()
        self.tmp.cleanup()

    def call(self, tool, **args):
        with mock.patch("tools.trading_tools._engine_for",
                        return_value=(self.eng, None)):
            return dispatch(ToolContext(active_agent="tom"), tool, args)

    def ticket(self):
        return tticket.load("schwab")

    def test_a_date_past_the_loaded_week_is_fetched_not_snapped(self):
        out = self.call("build_strategy", strategy="Short Put", symbol="SPX",
                        expiration="2027-02-17")
        self.assertIn("Built", out)
        self.assertEqual(self.ticket().legs[0].expiration, "2027-02-19")
        self.assertIn(("2027-02-03", "2027-03-03", False), self.broker.asked)

    def test_a_date_under_two_weeks_out_never_asks_for_the_past(self):
        # Schwab 400s a window starting before today (#42); ten days out,
        # date - 14 would start four days ago.
        class Nov10(date):
            @classmethod
            def today(cls):
                return date(2026, 11, 10)
        with mock.patch("tools.trading_tools.date", Nov10):
            out = self.call("build_strategy", strategy="Short Put", symbol="SPX",
                            expiration="2026-11-20")
        self.assertIn("Built", out)
        self.assertIn(("2026-11-10", "2026-12-04", False), self.broker.asked)

    def test_a_date_with_nothing_listed_near_it_is_refused(self):
        out = self.call("build_strategy", strategy="Short Put", symbol="SPX",
                        expiration="2026-12-20")
        self.assertIn("no expiration listed within 14 days of 2026-12-20 Sun", out)
        # Nov 20 is listed but wasn't loaded: no "nearest" to get wrong.
        self.assertNotIn("nearest", out)
        self.assertIsNone(self.ticket())

    def test_inside_the_loaded_week_it_still_snaps_without_a_fetch(self):
        self.call("build_strategy", strategy="Short Put", symbol="SPX",
                  expiration="2026-10-20")
        self.assertEqual(self.ticket().legs[0].expiration, "2026-10-21")
        self.assertEqual(self.broker.asked, [None])     # the normal load only

    def test_a_strike_past_the_loaded_ones_fetches_that_expiration(self):
        self.call("build_strategy", strategy="Short Put", symbol="SPX")
        out = self.call("adjust_leg", leg_number=1, strike=6390)
        self.assertIn("Leg 1 updated", out)
        self.assertEqual(self.ticket().legs[0].strike, 6390)
        self.assertIn(("2026-10-21", "2026-10-21", True), self.broker.asked)

    def test_a_strike_that_isnt_listed_is_refused_and_the_leg_kept(self):
        self.call("build_strategy", strategy="Short Put", symbol="SPX")
        before = self.ticket().legs[0].strike
        out = self.call("adjust_leg", leg_number=1, strike=6500)
        self.assertIn("strikes run from 6300 to 6400; 6500 isn't listed", out)
        self.assertEqual(self.ticket().legs[0].strike, before)

    def test_a_ticket_on_a_fetched_stretch_outlives_the_cache(self):
        self.call("build_strategy", strategy="Short Put", symbol="SPX",
                  expiration="2027-02-19")
        tchains.clear_cache()                       # fifteen minutes later
        tk, chain, problem = trading_tools._ticket_chain(self.eng)
        self.assertIsNone(problem)
        self.assertEqual(tk.validate(chain), [])


class SchwabRequestTests(SchwabBase):
    def test_the_normal_load_is_a_week_at_eighty_strikes(self):
        c = TestAccountsAndOrders.signed_in(self)
        self.http.script["GET /marketdata/v1/chains"] = FakeResponse(200, {})
        schwab.fetch_chain(c, "SPX")
        today = date.fromtimestamp(self.now)
        params = self.http.calls[-1][2]["params"]
        self.assertEqual((params["symbol"], params["fromDate"], params["toDate"],
                          params["strikeCount"]),
                         ("$SPX", today.isoformat(),
                          (today + timedelta(days=tcfg.SCHWAB_CHAIN_DAYS)).isoformat(),
                          tcfg.SCHWAB_CHAIN_STRIKES))

    def test_a_window_with_every_strike_drops_the_strike_count(self):
        c = TestAccountsAndOrders.signed_in(self)
        self.http.script["GET /marketdata/v1/chains"] = FakeResponse(200, {})
        schwab.fetch_chain(c, "SPX", start="2027-02-19", end="2027-02-19",
                           every_strike=True)
        params = self.http.calls[-1][2]["params"]
        self.assertEqual((params["fromDate"], params["toDate"]),
                         ("2027-02-19", "2027-02-19"))
        self.assertNotIn("strikeCount", params)


if __name__ == "__main__":
    unittest.main()
