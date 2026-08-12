"""snapshot_merge must treat 0.0 as a price, not as a missing field.

A deep-OTM option with no buyer legitimately bids 0.00; the old `or`-merge
read that as "absent" and kept a stale non-zero bid standing — which then fed
Quote.mid and the price the user is asked to confirm.
"""

import unittest

from trading.quotes import QuoteService


def merge(svc, **fields):
    svc.snapshot_merge([{"symbol": "SPXW  260810P06100000", **fields}],
                       {"SPXW  260810P06100000": ".SPXW260810P6100"})
    return svc._quotes[".SPXW260810P6100"]


class SnapshotZeroTests(unittest.TestCase):
    def setUp(self):
        self.svc = QuoteService(client=None)  # no start(): no thread, no network

    def test_a_zero_bid_overwrites_a_stale_price(self):
        merge(self.svc, bid="0.10", ask="0.15")
        q = merge(self.svc, bid="0.0", ask="0.05")
        self.assertEqual((q.bid, q.ask), (0.0, 0.05))
        self.assertEqual(q.mid, 0.025)  # not (0.10 + 0.05) / 2

    def test_a_first_snapshot_zero_bid_lands(self):
        q = merge(self.svc, bid="0.0", ask="0.05")
        self.assertEqual(q.bid, 0.0)

    def test_missing_fields_still_keep_the_cached_value(self):
        merge(self.svc, bid="0.10", ask="0.15", last="0.12")
        q = merge(self.svc, ask="0.20")  # partial snapshot: no bid, no last
        self.assertEqual((q.bid, q.ask, q.last), (0.10, 0.20, 0.12))

    def test_last_falls_back_to_mark(self):
        q = merge(self.svc, mark="0.12")
        self.assertEqual(q.last, 0.12)


if __name__ == "__main__":
    unittest.main()
