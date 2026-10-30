"""option_chain (#59): a strip of strikes quoted in one batch, picked by
delta, by range or around the money — and never a ticket built to do it.
On 2026-10-29 Tom found "the 30-delta put" by re-pricing a scratch ticket
four times, unasked.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from tests.test_broker_routing import run
from tests.trading_fixtures import spx_chain
from trading import config as tcfg
from trading.models import Quote

SPOT = 6350.0


def put_delta(strike, floor=0.10):
    """A toy put delta: -0.50 at the money, 0.01 per point, floored."""
    d = 0.5 - (SPOT - strike) / 100
    return -min(0.98, max(floor, d))


class StripEngine:
    """Enough engine for option_chain: the fixture SPX chain, SPX at 6350,
    and leg quotes from `quote_of(strike)`; every prefetch is recorded."""
    key = "schwab"

    def __init__(self, quote_of=None, error=None):
        self.chain_data = spx_chain()
        self.quote_of = quote_of or (lambda k: Quote(
            "x", bid=1.0, ask=1.2, delta=put_delta(k), iv=0.15, as_of=None))
        self.quotes = SimpleNamespace(last_error=error)
        self.prefetched = []

    def chain(self, underlying, force=False):
        return self.chain_data

    def extend_chain(self, chain, start, end, every_strike=False):
        return chain

    def underlying_quote(self, underlying):
        return Quote("SPX", last=SPOT)

    def leg_quote_fn(self, chain, greeks=False):
        by_streamer = {i.streamer: k for (e, k, cp), i in chain.options.items()}

        def get(streamer):
            return self.quote_of(by_streamer[streamer])
        get.prefetch = self.prefetched.append
        return get


def strip(eng, **args):
    with mock.patch("trading.ticket.save") as save:
        out = run("tom", "option_chain", {"symbol": "SPX", "side": "put", **args},
                  engines={"schwab": eng})
    save.assert_not_called()                     # quotes only: no ticket
    return out


class TestOptionChain(unittest.TestCase):
    def test_a_delta_finds_the_closest_strike_in_one_batch(self):
        eng = StripEngine()
        out = strip(eng, delta=30)
        self.assertIn("Closest to 0.30 delta: the 6330 at -0.30.", out)
        self.assertIn("6330: bid 1.00, ask 1.20, mid 1.10, delta -0.30, IV 15.0%", out)
        self.assertEqual(out.count("bid "), 5)           # default five strikes
        self.assertEqual(len(eng.prefetched), 1)          # one request
        scanned = eng.prefetched[0]
        self.assertEqual(len(scanned), 11)                # 6300..6350: the put side
        self.assertNotIn("Nothing nearer", out)

    def test_a_delta_out_of_reach_says_so(self):
        out = strip(StripEngine(), delta=0.02)            # the floor is 0.10
        self.assertIn("Closest to 0.02 delta: the 6300 at -0.10.", out)
        self.assertIn("Nothing nearer among the strikes quoted", out)

    def test_a_strike_range_is_quoted_whole(self):
        out = strip(StripEngine(), from_strike=6320, to_strike=6310)
        self.assertEqual([s.split(":")[0] for s in out.split("; ")][1:],
                         ["6315", "6320"])
        self.assertIn("6310: bid", out)

    def test_a_long_range_is_capped_and_says_so(self):
        eng = StripEngine()
        eng.chain_data = spx_chain(strikes=[6000 + 5 * i for i in range(41)])
        out = strip(eng, from_strike=6000, to_strike=6200)
        self.assertIn(f"The first {tcfg.OPTION_CHAIN_MAX_STRIKES} of the 41 strikes", out)

    def test_no_arguments_is_the_strikes_nearest_the_money(self):
        out = strip(StripEngine(), count=3)
        self.assertIn("SPX at 6,350.00", out)
        for k in ("6345:", "6350:", "6355:"):
            self.assertIn(k, out)
        self.assertEqual(out.count("bid "), 3)

    def test_no_quotes_says_why(self):
        eng = StripEngine(quote_of=lambda k: None, error="502 Bad Gateway")
        out = strip(eng, delta=30)
        self.assertIn("No live quotes for the SPX", out)
        self.assertIn("502", out)

    def test_quotes_without_delta_cannot_pick_by_it(self):
        eng = StripEngine(quote_of=lambda k: Quote("x", bid=1.0, ask=1.2))
        self.assertIn("carried no delta", strip(eng, delta=30))

    def test_side_is_required_in_words(self):
        out = run("tom", "option_chain", {"symbol": "SPX", "side": "both"},
                  engines={"schwab": StripEngine()})
        self.assertIn("puts or calls", out)


if __name__ == "__main__":
    unittest.main()
