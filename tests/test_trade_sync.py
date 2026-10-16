"""The trade log synced from the Schwab API (#12). Fixtures copy the real
response shapes (account numbers masked): one transaction per leg per fill
with its own fee items, expirations as RECEIVE_AND_DELIVER, spread type
from /orders. On 2026-10-15 these rows matched 83 of 83 statement rows to
the cent; the tests pin that shape and the sync's write path.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from tools import ToolContext, dispatch
from trading import accounts as taccounts
from trading import config as tcfg
from trading import schwab, trade_log

N = "12345839"
ACCT = "*****839SCHW"


def fee(kind, cost):
    return {"instrument": {"assetType": "CURRENCY", "symbol": "CURRENCY_USD"},
            "amount": -cost, "cost": cost, "feeType": kind}


def leg(occ, amount, cost, price, effect):
    return {"instrument": {"assetType": "OPTION", "symbol": occ, "underlyingSymbol": "SPXW",
                           "optionPremiumMultiplier": 100},
            "amount": amount, "cost": cost, "price": price, "positionEffect": effect}


def trade(order_id, activity, when, item, net):
    return {"activityId": activity, "time": when, "type": "TRADE", "orderId": order_id,
            "netAmount": net, "transferItems": [
                fee("COMMISSION", -0.65), fee("OPT_REG_FEE", -0.01),
                fee("INDEX_OPTION_FEE", -0.56), item]}


# A diagonal bought for a 64.17 debit: long the 10-30 7740 put, short the
# 10-20 7700 put — the user's order 1008301726753, both legs at 20:13:42 UTC.
DIAGONAL = [
    trade(1008301726753, 133520332594, "2026-10-14T20:13:42+0000",
          leg("SPXW  261030P07740000", 1.0, -8523.0, 85.23, "OPENING"), -8524.22),
    trade(1008301726753, 133520332595, "2026-10-14T20:13:42+0000",
          leg("SPXW  261020P07700000", -1.0, 2106.0, 21.06, "OPENING"), 2104.78),
]
EXPIRATION = {"activityId": 132349031935, "time": "2026-10-02T06:39:23+0000",
              "type": "RECEIVE_AND_DELIVER", "netAmount": 0.0,
              "description": "Removed due to Expiration PUT S & P 500 INDEX $7550 EXP 10/01/26",
              "transferItems": [leg("SPXW  261001P07550000", -1.0, 0.0, 0.0, "CLOSING")]}
ORDERS = [{"orderId": 1008301726753, "complexOrderStrategyType": "DIAGONAL",
           "filledQuantity": 1.0, "status": "FILLED"}]


def local(ts):
    return datetime.fromisoformat(ts.replace("+0000", "+00:00")).astimezone() \
        .replace(tzinfo=None).isoformat(timespec="seconds")


class FakeClient:
    def __init__(self, txs, orders=ORDERS):
        self.txs, self.orders, self.calls = txs, orders, []

    def transactions_all(self, account, start, end, types="TRADE"):
        self.calls.append((account, start, end, types))
        return list(self.txs)

    def list_orders(self, account, start_date=None):
        return list(self.orders)


class TestRows(unittest.TestCase):
    def test_a_diagonals_two_leg_transactions_are_one_row_with_its_fees(self):
        rows, skipped = schwab.trade_log_rows(FakeClient(DIAGONAL), N, "2026-10-01", "2026-10-15")
        self.assertEqual(skipped, 0)
        (r,) = rows
        self.assertEqual((r["account"], r["ref"], r["source"]), (ACCT, "1008301726753", "api"))
        self.assertEqual(r["executed_at"], local("2026-10-14T20:13:42+0000"))
        self.assertEqual((r["symbol"], r["strategy"], r["quantity"]), ("SPX", "diagonal", 1))
        self.assertEqual(r["amount"], -6417.0)          # the statement's AMOUNT
        self.assertEqual(r["fees"], -2.44)              # both legs' three fees
        self.assertEqual(r["price"], 64.17)
        self.assertEqual(sorted(r["legs"]), [("2026-10-20", 7700.0, "P", -1),
                                             ("2026-10-30", 7740.0, "P", 1)])

    def test_expirations_keep_their_activity_id_and_signed_change(self):
        rows, _ = schwab.trade_log_rows(FakeClient([EXPIRATION]), N, "2026-10-01", "2026-10-15")
        (r,) = rows
        self.assertEqual((r["ref"], r["strategy"], r["quantity"], r["amount"]),
                         ("132349031935", "expiration", None, 0.0))
        self.assertEqual(r["legs"], [("2026-10-01", 7550.0, "P", -1)])

    def test_a_credit_is_a_sell_and_unknown_shapes_are_custom(self):
        sold = [trade(9, 1, "2026-10-14T20:00:00+0000",
                      leg("SPXW  261020P07700000", -2.0, 2000.0, 10.0, "OPENING"), 1998.78)]
        rows, _ = schwab.trade_log_rows(FakeClient(sold, orders=[]), N, "a", "b")
        self.assertEqual((rows[0]["strategy"], rows[0]["quantity"], rows[0]["price"]),
                         ("single", -2, 10.0))
        three = [trade(8, i, "2026-10-14T20:00:00+0000",
                       leg(f"SPXW  26102{i}P07700000", 1.0, -100.0, 1.0, "OPENING"), -101.22)
                 for i in (1, 2, 3)]
        rows, _ = schwab.trade_log_rows(FakeClient(three, orders=[]), N, "a", "b")
        self.assertEqual(rows[0]["strategy"], "custom")

    def test_stock_orders_are_skipped_and_counted(self):
        stock = [trade(7, 1, "2026-10-14T20:00:00+0000",
                       {"instrument": {"assetType": "EQUITY", "symbol": "AAPL"},
                        "amount": 10.0, "cost": -2500.0}, -2500.0)]
        rows, skipped = schwab.trade_log_rows(FakeClient(stock), N, "a", "b")
        self.assertEqual((rows, skipped), ([], 1))

    def test_it_asks_for_expirations_too(self):
        c = FakeClient([])
        schwab.trade_log_rows(c, N, "2026-10-01", "2026-10-15")
        self.assertEqual(c.calls[0][3], "TRADE,RECEIVE_AND_DELIVER")


class TestSyncTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for name, value in (("TRADE_LOG_PATH", Path(self.tmp.name) / "log.db"),
                            ("TRADING_DIR", Path(self.tmp.name))):
            p = mock.patch.object(tcfg, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = FakeClient(DIAGONAL + [EXPIRATION])
        self.eng = mock.Mock(client=self.client)
        self.eng.accounts.return_value = [N, "99999699"]
        self.eng.label.side_effect = lambda n: taccounts.label("schwab", n)

    def sync(self, **args):
        with mock.patch("trading.engine.get_engine", return_value=(self.eng, None)):
            return dispatch(ToolContext(active_agent="tom"), "sync_trades", {"account": "839", **args})

    def alert(self):
        """The fill alert that announced the diagonal: an alert #, no fees."""
        trade_log.save([{"account": ACCT, "ref": "133520332593",
                         "executed_at": local("2026-10-14T20:13:42+0000"),
                         "symbol": "SPX", "strategy": "diagonal", "quantity": 1,
                         "price": 64.17, "amount": -6417.0, "fees": None, "fills": 1,
                         "description": "BOT +1 DIAGONAL ...", "source": "email",
                         "legs": [("2026-10-30", 7740.0, "P", 1), ("2026-10-20", 7700.0, "P", -1)]}])

    def test_sync_writes_exact_rows_and_replaces_the_alert_keeping_its_position(self):
        self.alert()
        pid = trade_log.create_position(ACCT, "the put diagonal")
        trade_log.assign(ACCT, ["133520332593"], pid)
        out = self.sync(start_date="2026-10-01")
        self.assertIn("2 new rows (1 orders, 1 expirations)", out)
        self.assertIn("Replaced 1 alert-logged fills", out)
        rows = {t["ref"]: t for t in trade_log.query(account=ACCT)}
        self.assertEqual(set(rows), {"1008301726753", "132349031935"})
        self.assertEqual(rows["1008301726753"]["fees"], -2.44)
        self.assertEqual(rows["1008301726753"]["position_id"], pid)   # carried over

    def test_a_second_sync_writes_nothing_new_and_resumes_where_the_log_ends(self):
        self.sync(start_date="2026-10-01")
        out = self.sync()
        self.assertIn("0 new rows", out)
        self.assertIn("2 already in the log", out)
        # Resumed a day before the newest synced row, not a year back.
        self.assertEqual(self.client.calls[-1][1], "2026-10-13")
        self.assertEqual(len(trade_log.query(account=ACCT)), 2)

    def test_an_account_schwab_doesnt_share_is_said(self):
        with mock.patch("trading.engine.get_engine", return_value=(self.eng, None)):
            out = dispatch(ToolContext(active_agent="tom"), "sync_trades", {"account": "868"})
        self.assertIn("No account called '868'", out)
        self.assertEqual(self.client.calls, [])

    def test_an_expired_sign_in_is_said_not_raised(self):
        with mock.patch("trading.engine.get_engine",
                        return_value=(None, "Schwab needs you to sign in again")):
            out = dispatch(ToolContext(active_agent="tom"), "sync_trades", {})
        self.assertIn("sign in again", out)

    def test_tom_has_it_and_linda_does_not(self):
        from brain import agents
        self.assertIn("sync_trades", agents.AGENTS["tom"]["tools"])
        self.assertNotIn("sync_trades", agents.AGENTS["linda"]["tools"])


if __name__ == "__main__":
    unittest.main()
