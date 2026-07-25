import unittest

from trading import pnl as tpnl


def tx(ttype="Trade", sub="", action="", symbol="SPXW  260724C06350000",
       underlying="SPX", qty=1, value=None, net=None, at="2026-07-20T15:00:00Z"):
    """Build a transaction dict the way the API shapes them: value/net-value
    magnitudes with Credit/Debit effects."""
    d = {"transaction-type": ttype, "transaction-sub-type": sub,
         "action": action, "symbol": symbol, "underlying-symbol": underlying,
         "quantity": str(qty), "executed-at": at}
    for key, amount in (("value", value), ("net-value", net)):
        if amount is not None:
            d[key] = str(abs(amount))
            d[f"{key}-effect"] = "Credit" if amount > 0 else "Debit"
    return d


class TestFifo(unittest.TestCase):
    def test_long_round_trip(self):
        events, fees, settle, unmatched = tpnl.realized_events([
            tx(action="Buy to Open", qty=2, value=-200, net=-201,
               at="2026-07-20T15:00:00Z"),
            tx(action="Sell to Close", qty=2, value=300, net=298.8,
               at="2026-07-21T15:00:00Z"),
        ])
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].pnl, 97.8)     # 298.8 - 201
        self.assertAlmostEqual(fees, 2.2)               # 1 + 1.2
        self.assertEqual(unmatched, 0)

    def test_short_round_trip(self):
        events, *_ = tpnl.realized_events([
            tx(action="Sell to Open", qty=1, net=150),
            tx(action="Buy to Close", qty=1, net=-60,
               at="2026-07-21T15:00:00Z"),
        ])
        self.assertAlmostEqual(events[0].pnl, 90.0)

    def test_partial_close_fifo(self):
        events, *_ = tpnl.realized_events([
            tx(action="Buy to Open", qty=2, net=-200, at="2026-07-20T15:00:00Z"),
            tx(action="Buy to Open", qty=1, net=-90, at="2026-07-20T16:00:00Z"),
            tx(action="Sell to Close", qty=2, net=260, at="2026-07-21T15:00:00Z"),
        ])
        # First lot (2 @ -100 each) matches: 260 - 200 = 60.
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].pnl, 60.0)
        self.assertAlmostEqual(events[0].quantity, 2)

    def test_expiration_closes_at_zero(self):
        events, *_ = tpnl.realized_events([
            tx(action="Buy to Open", qty=1, net=-120),
            tx(ttype="Receive Deliver", sub="Expiration", qty=1, net=0,
               at="2026-07-24T21:00:00Z"),
        ])
        self.assertAlmostEqual(events[0].pnl, -120.0)

    def test_bare_futures_actions_use_inventory(self):
        events, *_ = tpnl.realized_events([
            tx(action="Buy", symbol="/ESU6", underlying="/ES",
               qty=1, net=-10, at="2026-07-20T15:00:00Z"),
            tx(action="Sell", symbol="/ESU6", underlying="/ES",
               qty=1, net=260, at="2026-07-21T15:00:00Z"),
        ])
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].pnl, 250.0)

    def test_futures_settlement_is_collected(self):
        _, _, settle, _ = tpnl.realized_events([
            tx(ttype="Money Movement", sub="Futures Settlement",
               symbol="/ESU6", underlying="/ES", qty=0, net=137.5),
        ])
        self.assertAlmostEqual(settle["/ES"][0][1], 137.5)

    def test_unmatched_close_flagged(self):
        _, _, _, unmatched = tpnl.realized_events([
            tx(action="Sell to Close", qty=1, net=100),
        ])
        self.assertEqual(unmatched, 1)


class TestReport(unittest.TestCase):
    def test_period_filters_closes_not_opens(self):
        txs = [
            tx(action="Buy to Open", qty=1, net=-100, at="2026-06-01T15:00:00Z"),
            tx(action="Sell to Close", qty=1, net=180, at="2026-07-02T15:00:00Z"),
            tx(action="Buy to Open", qty=1, net=-100, at="2026-07-03T15:00:00Z"),
            tx(action="Sell to Close", qty=1, net=90, at="2026-08-01T15:00:00Z"),
        ]
        rep = tpnl.report(txs, "2026-07-01", "2026-07-31")
        # Only the July close counts (opened in June is fine); August's not.
        self.assertAlmostEqual(rep.realized, 80.0)
        self.assertEqual(len(rep.events), 1)
        self.assertAlmostEqual(rep.by_underlying["SPX"], 80.0)

    def test_settlements_in_period(self):
        txs = [tx(ttype="Money Movement", sub="Mark to Market",
                  symbol="/ESU6", underlying="/ES", qty=0, net=-62.5,
                  at="2026-07-10T22:00:00Z")]
        rep = tpnl.report(txs, "2026-07-01", "2026-07-31")
        self.assertAlmostEqual(rep.realized, -62.5)
        self.assertAlmostEqual(rep.by_underlying["/ES"], -62.5)

    def test_unrealized_with_marks(self):
        positions = tpnl.parse_positions([
            {"symbol": "SPXW  260724C06350000", "underlying-symbol": "SPX",
             "instrument-type": "Equity Option", "quantity": "2",
             "quantity-direction": "Long", "average-open-price": "10.0",
             "multiplier": "100", "close-price": "11.0"},
            {"symbol": "SPXW  260724P06300000", "underlying-symbol": "SPX",
             "instrument-type": "Equity Option", "quantity": "1",
             "quantity-direction": "Short", "average-open-price": "5.0",
             "multiplier": "100", "close-price": "4.0"},
        ])
        self.assertEqual([p.quantity for p in positions], [2.0, -1.0])
        rep = tpnl.report([], None, None, positions=positions,
                          mark_fn=lambda p: 12.0 if p.quantity > 0 else 4.0)
        # Long: (12-10)*2*100 = +400; short: (4-5)*(-1)*100 = +100.
        self.assertAlmostEqual(rep.unrealized, 500.0)
        self.assertEqual(len(rep.unrealized_by_position), 2)

    def test_missing_mark_noted(self):
        positions = tpnl.parse_positions([
            {"symbol": "X", "underlying-symbol": "X",
             "instrument-type": "Equity", "quantity": "1",
             "quantity-direction": "Long", "average-open-price": "10",
             "multiplier": "1"}])
        rep = tpnl.report([], None, None, positions=positions,
                          mark_fn=lambda p: None)
        self.assertIn("no live mark", rep.note)
        self.assertAlmostEqual(rep.unrealized, 0.0)

    def test_zero_quantity_positions_dropped(self):
        self.assertEqual(tpnl.parse_positions([
            {"symbol": "X", "quantity": "0", "quantity-direction": "Zero",
             "average-open-price": "1", "multiplier": "1"}]), [])


if __name__ == "__main__":
    unittest.main()
