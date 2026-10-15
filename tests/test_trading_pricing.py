"""Pricing apart from the broker's dry run, with greeks.

On 2026-10-05 review_order's dry run was rejected for buying power (the
tastytrade account can't carry the user's thinkorswim-sized positions), the
rejection dropped the mid it had just computed, and Tom concluded he could
not price option legs at all. price_order reads quotes only, so size and
buying power can't stop it, and it reports the greeks dxFeed streams.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from tests.trading_fixtures import spx_chain
from tools import ToolContext, dispatch
from trading.models import Leg, Quote
from trading.quotes import QuoteService
from trading.ticket import Ticket

SYM = ".SPXW261009C7850"


def calendar():
    """Short the 07-24 6350 call, long the 08-07 6350 call, two each."""
    return Ticket(underlying="SPX", strategy="Calendar Spread", legs=[
        Leg(6350.0, "Call", "2026-07-24", "Short", size=2),
        Leg(6350.0, "Call", "2026-08-07", "Long", size=2),
    ])


def quote(bid, ask, **greeks):
    return Quote(symbol="x", bid=bid, ask=ask, updated_at=1.0, **greeks)


class TestGreeksFeed(unittest.TestCase):
    def setUp(self):
        self.svc = QuoteService(client=None)  # no start(): no thread, no network
        self.sent = []
        self.svc._send = self.sent.append

    def test_compact_greeks_land_on_the_quote(self):
        self.svc._feed_data(["Quote", [SYM, "20.1", "20.5"],
                             "Greeks", [SYM, "0.142", "0.41", "0.0031",
                                        "-4.1", "NaN"]])
        q = self.svc.get(SYM)
        self.assertEqual((q.bid, q.ask), (20.1, 20.5))
        self.assertEqual((q.iv, q.delta, q.gamma, q.theta), (0.142, 0.41, 0.0031, -4.1))
        self.assertIsNone(q.vega)  # NaN is "unknown", not a number

    def test_a_greeks_wait_is_not_satisfied_by_a_price_alone(self):
        self.svc._feed_data(["Quote", [SYM, "20.1", "20.5"]])
        self.assertIsNone(self.svc.get(SYM, greeks=True).delta)  # returned at the deadline

    def test_greeks_are_subscribed_only_when_asked(self):
        self.svc._feed_ready.set()
        self.svc.subscribe([SYM])
        self.svc.subscribe([SYM], greeks=True)
        types = [a["type"] for msg in self.sent for a in msg["add"]]
        self.assertEqual(types, ["Quote", "Greeks"])


class TestNetGreeks(unittest.TestCase):
    def test_short_legs_flip_sign_and_size_multiplies(self):
        tk = calendar()
        quotes = [(tk.legs[0], quote(20, 21, delta=0.50, gamma=0.004, theta=-6.0, vega=2.0)),
                  (tk.legs[1], quote(30, 31, delta=0.52, gamma=0.002, theta=-3.0, vega=5.0))]
        ng = tk.net_greeks(quotes)
        self.assertAlmostEqual(ng["delta"], 0.04)
        self.assertAlmostEqual(ng["theta"], 6.0)    # the short's decay is yours
        self.assertAlmostEqual(ng["vega"], 6.0)
        self.assertAlmostEqual(tk.net_of(quotes), -20.0)  # 2 x (20.5 - 30.5) debit

    def test_one_missing_greek_voids_that_net(self):
        tk = calendar()
        ng = tk.net_greeks([(tk.legs[0], quote(20, 21, delta=0.5)),
                            (tk.legs[1], quote(30, 31))])
        self.assertIsNone(ng["delta"])


class TestPriceOrderTool(unittest.TestCase):
    def price(self, quotes_by_expiration):
        chain, tk = spx_chain(), calendar()
        by_streamer = {chain.option(l.expiration, l.strike, "C").streamer:
                       quotes_by_expiration[l.expiration] for l in tk.legs}
        eng = SimpleNamespace(leg_quote_fn=lambda ch, greeks=False: by_streamer.get,
                              orders=None)  # any broker call would raise
        with mock.patch("tools.trading_tools._engine_and_ticket",
                        return_value=(eng, tk, chain, None)):
            return dispatch(ToolContext(active_agent="tom"), "price_order", {})

    def test_prices_and_greeks_with_no_broker_involved(self):
        out = self.price({
            "2026-07-24": quote(20.1, 20.5, iv=0.14, delta=0.5, gamma=0.004, theta=-6.0, vega=2.0),
            "2026-08-07": quote(30.1, 30.5, iv=0.15, delta=0.52, gamma=0.002, theta=-3.0, vega=5.0)})
        self.assertIn("STO 2 2026-07-24 6350 Call: bid 20.10, ask 20.50, mid 20.30", out)
        self.assertIn("IV 14.0%, delta +0.500", out)
        self.assertIn("Net mid 20.00 debit", out)
        self.assertIn("theta +6.00 a day", out)
        self.assertIn("buying power not checked", out)

    def test_missing_greeks_are_said_not_guessed(self):
        out = self.price({"2026-07-24": quote(20.1, 20.5),
                          "2026-08-07": quote(30.1, 30.5)})
        self.assertIn("greeks not available", out)
        self.assertIn("No net greeks", out)
        self.assertIn("Net mid 20.00 debit", out)  # the price still comes back

    def test_tom_has_it(self):
        from brain import agents
        self.assertIn("price_order", agents.AGENTS["tom"]["tools"])


if __name__ == "__main__":
    unittest.main()
