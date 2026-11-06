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
from trading import web_api
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
        self.assertIn("STO 2 2026-07-24 Fri 6350 Call: bid 20.10, ask 20.50, mid 20.30", out)
        self.assertIn("IV 14.0%, delta +0.500", out)
        self.assertIn("Net mid 10.00 debit per spread; 2 spreads, $2,000.00 total", out)
        self.assertIn("theta +6.00 a day", out)
        self.assertIn("buying power not checked", out)

    def test_missing_greeks_are_said_not_guessed(self):
        out = self.price({"2026-07-24": quote(20.1, 20.5),
                          "2026-08-07": quote(30.1, 30.5)})
        self.assertIn("greeks not available", out)
        self.assertIn("No net greeks", out)
        self.assertIn("Net mid 10.00 debit", out)  # the price still comes back

    def test_tom_has_it(self):
        from brain import agents
        self.assertIn("price_order", agents.AGENTS["tom"]["tools"])


class TestPerSpreadPrice(unittest.TestCase):
    """The ticket total read as a per-spread price gave four wrong spoken
    answers on 2-lots (#29), and every path below would have set it as the
    order's limit: a debit overpaying up to the ask, a credit never filling."""

    MIDS = {"2026-07-24": quote(10.0, 10.0),   # the short: a 3.95 credit a spread
            "2026-08-07": quote(6.05, 6.05)}

    def setUp(self):
        self.chain, self.tk = spx_chain(), calendar()
        by_streamer = {self.chain.option(l.expiration, l.strike, "C").streamer:
                       self.MIDS[l.expiration] for l in self.tk.legs}
        self.eng = SimpleNamespace(
            key="tastytrade", chain=lambda u: self.chain, orders=None,
            leg_quote_fn=lambda ch, greeks=False: by_streamer.get)
        patcher = mock.patch("trading.ticket.save")
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_tool(self, name, args=None):
        with mock.patch("tools.trading_tools._engine_and_ticket",
                        return_value=(self.eng, self.tk, self.chain, None)):
            return dispatch(ToolContext(active_agent="tom"), name, args or {})

    def assert_limit_is_per_spread(self):
        self.assertEqual((self.tk.limit_price, self.tk.price_effect), (3.95, "Credit"))
        self.assertEqual([l.size for l in self.tk.legs], [2, 2])  # size is untouched

    def test_price_order_says_per_spread_count_and_dollars(self):
        out = self.run_tool("price_order")
        self.assertIn("Net mid 3.95 credit per spread; 2 spreads, $790.00 total", out)

    def test_set_at_mid(self):
        out = self.run_tool("set_order_terms", {"use_mid": True})
        self.assert_limit_is_per_spread()
        self.assertIn("3.95 Credit per spread", out)

    def test_voice_review_defaults_to_the_per_spread_mid(self):
        with mock.patch("tools.trading_tools._one_account",
                        return_value=(None, "stop here")):
            self.run_tool("review_order")
        self.assert_limit_is_per_spread()

    def test_dashboard_review_defaults_to_the_per_spread_mid(self):
        with mock.patch("trading.web_api._engine", return_value=(self.eng, None)), \
             mock.patch("trading.web_api.tticket.load", return_value=self.tk), \
             mock.patch("trading.web_api._account", return_value=(None, "stop here")):
            web_api.dry_run({"broker": "tastytrade"})
        self.assert_limit_is_per_spread()

    def test_a_limit_already_set_is_left_alone(self):
        self.tk.limit_price, self.tk.price_effect = 4.10, "Credit"
        with mock.patch("tools.trading_tools._one_account",
                        return_value=(None, "stop here")):
            self.run_tool("review_order")
        self.assertEqual(self.tk.limit_price, 4.10)


class TestSpreadCount(unittest.TestCase):
    def net(self, sizes, mids, sides):
        tk = Ticket(underlying="SPX", legs=[
            Leg(6350.0 + 5 * i, "Call", "2026-07-24", side, size=size)
            for i, (size, side) in enumerate(zip(sizes, sides))])
        return tk.spread_count(), tk.net_per_spread(
            [(leg, quote(m, m)) for leg, m in zip(tk.legs, mids)])

    def test_one_lot_is_its_own_total(self):
        count, net = self.net([1, 1], [10.0, 6.05], ["Short", "Long"])
        self.assertEqual(count, 1)
        self.assertAlmostEqual(net, 3.95)

    def test_a_butterfly_is_one_spread_not_two(self):
        count, net = self.net([1, 2, 1], [12.0, 8.0, 5.0], ["Long", "Short", "Long"])
        self.assertEqual(count, 1)
        self.assertAlmostEqual(net, -1.0)    # -12 + 16 - 5

    def test_two_butterflies(self):
        count, net = self.net([2, 4, 2], [12.0, 8.0, 5.0], ["Long", "Short", "Long"])
        self.assertEqual(count, 2)
        self.assertAlmostEqual(net, -1.0)

    def test_a_single_option_prices_per_contract(self):
        count, net = self.net([3], [10.2], ["Short"])
        self.assertEqual(count, 3)
        self.assertAlmostEqual(net, 10.2)

    def test_a_missing_quote_stays_none(self):
        tk = calendar()
        self.assertIsNone(tk.net_per_spread([(tk.legs[0], None),
                                             (tk.legs[1], quote(1, 2))]))


if __name__ == "__main__":
    unittest.main()
