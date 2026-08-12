import tempfile
import time
import unittest
from datetime import date, timedelta

from trading import config as tcfg
from pathlib import Path

from tests.trading_fixtures import spx_chain
from trading import config as tcfg
from trading import orders as torders
from trading import ticket as tticket
from trading.models import Leg
from trading.orders import OrderManager
from trading.tasty_client import TastyError
from trading.ticket import Ticket


class FakeClient:
    """Canned tastytrade responses; records what was sent."""
    def __init__(self):
        self.dry_runs, self.placed, self.cancelled = [], [], []
        self.reject_dry_run = None
        self.account_id = "5WT00001"

    def dry_run_order(self, payload):
        self.dry_runs.append(payload)
        if self.reject_dry_run:
            raise TastyError("preflight", status=422,
                             errors=[{"message": self.reject_dry_run}])
        return {
            "order": {"id": "0", "status": "Received"},
            "warnings": [{"message": "resulting position is a spread"}],
            "buying-power-effect": {
                "change-in-buying-power": "251.6",
                "change-in-buying-power-effect": "Debit",
                "new-buying-power": "9748.4",
                "new-buying-power-effect": "Credit",
                "isolated-order-margin-requirement": "500.0",
                "isolated-order-margin-requirement-effect": "Debit"},
            "fee-calculation": {"total-fees": "3.4",
                                "total-fees-effect": "Debit"},
        }

    def place_order(self, payload):
        self.placed.append(payload)
        return {"order": {"id": "123", "status": "Routed",
                          "underlying-symbol": "SPX", "price": "1.50",
                          "price-effect": "Credit", "time-in-force": "Day",
                          "received-at": "2026-07-24T15:00:00Z",
                          "legs": [{"symbol": "S", "action": "Sell to Open",
                                    "quantity": 1}]}}

    def get_order(self, order_id):
        return {"id": order_id, "status": "Live", "underlying-symbol": "SPX"}

    def list_orders(self, **kw):
        return {"items": [
            {"id": "123", "status": "Live", "underlying-symbol": "SPX"},
            {"id": "99", "status": "Filled", "underlying-symbol": "SPX"}]}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "Cancel Requested",
                "underlying-symbol": "SPX"}


class OrdersBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = (tcfg.TRADING_DIR, tcfg.TICKET_PATH,
                       tcfg.ORDERS_LOG_PATH)
        tcfg.TRADING_DIR = Path(self.tmp.name)
        tcfg.TICKET_PATH = Path(self.tmp.name) / "ticket.json"
        tcfg.ORDERS_LOG_PATH = Path(self.tmp.name) / "orders_log.json"
        self.chain = spx_chain()
        self.client = FakeClient()
        self.mgr = OrderManager(self.client)

    def tearDown(self):
        (tcfg.TRADING_DIR, tcfg.TICKET_PATH,
         tcfg.ORDERS_LOG_PATH) = self._saved
        self.tmp.cleanup()

    def ticket(self):
        tk = Ticket(underlying="SPX", strategy="Bull Put Spread", legs=[
            Leg(6345.0, "Put", "2026-07-24", "Short"),
            Leg(6340.0, "Put", "2026-07-24", "Long")])
        tk.limit_price, tk.price_effect = 1.50, "Credit"
        return tk


class TestReview(OrdersBase):
    def test_review_parses_and_stamps(self):
        tk = self.ticket()
        r = self.mgr.review(tk, self.chain)
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.buying_power_change, -251.6)
        self.assertAlmostEqual(r.new_buying_power, 9748.4)
        self.assertAlmostEqual(r.total_fees, 3.4)
        self.assertEqual(r.warnings, ["resulting position is a spread"])
        self.assertEqual(tk.review["fingerprint"], tk.fingerprint())
        self.assertEqual(tticket.load().review["fingerprint"],
                         tk.fingerprint())

    def test_review_rejection_surfaces_broker_message(self):
        self.client.reject_dry_run = "insufficient buying power"
        r = self.mgr.review(self.ticket(), self.chain)
        self.assertFalse(r.ok)
        self.assertIn("insufficient buying power", r.errors[0])

    def test_review_of_invalid_ticket_never_reaches_broker(self):
        tk = self.ticket()
        tk.legs[0].strike = 1.0
        r = self.mgr.review(tk, self.chain)
        self.assertFalse(r.ok)
        self.assertEqual(self.client.dry_runs, [])


class TestSubmitGate(OrdersBase):
    def test_no_confirm_no_order(self):
        tk = self.ticket()
        self.mgr.review(tk, self.chain)
        order, reason = self.mgr.submit(tk, self.chain, confirmed=False)
        self.assertIsNone(order)
        self.assertIn("confirmation", reason)
        self.assertEqual(self.client.placed, [])

    def test_no_review_no_order(self):
        order, reason = self.mgr.submit(self.ticket(), self.chain,
                                        confirmed=True)
        self.assertIsNone(order)
        self.assertIn("not been reviewed", reason)

    def test_changed_ticket_no_order(self):
        tk = self.ticket()
        self.mgr.review(tk, self.chain)
        tk.legs[0].strike = 6350.0   # edit after review
        order, reason = self.mgr.submit(tk, self.chain, confirmed=True)
        self.assertIsNone(order)
        self.assertIn("changed after its review", reason)
        self.assertEqual(self.client.placed, [])

    def test_stale_review_no_order(self):
        tk = self.ticket()
        self.mgr.review(tk, self.chain)
        tk.review["at"] = time.time() - torders.REVIEW_MAX_AGE_S - 1
        order, reason = self.mgr.submit(tk, self.chain, confirmed=True)
        self.assertIsNone(order)
        self.assertIn("stale", reason)

    def test_reviewed_and_confirmed_submits(self):
        tk = self.ticket()
        self.mgr.review(tk, self.chain)
        order, reason = self.mgr.submit(tk, self.chain, confirmed=True)
        self.assertIsNone(reason)
        self.assertEqual(order.id, "123")
        self.assertEqual(order.status, "Routed")
        self.assertTrue(order.is_working)
        # One review authorizes one submit.
        self.assertIsNone(tk.review)
        payload = self.client.placed[0]
        self.assertEqual(payload["price"], "1.50")
        self.assertEqual(payload["source"], "voice-agent")

    def test_audit_log_written(self):
        tk = self.ticket()
        self.mgr.review(tk, self.chain)
        self.mgr.submit(tk, self.chain, confirmed=True)
        kinds = [e["kind"] for e in torders.read_log()]
        self.assertEqual(kinds, ["dry-run", "submitted"])


class TestCancelAndList(OrdersBase):
    def test_cancel(self):
        order, reason = self.mgr.cancel("123")
        self.assertIsNone(reason)
        self.assertEqual(order.status, "Cancel Requested")
        self.assertEqual(self.client.cancelled, ["123"])

    def test_working_orders_filter(self):
        working = self.mgr.working_orders()
        self.assertEqual([o.id for o in working], ["123"])


class TestDefaultOrdersWindow(unittest.TestCase):
    """Two working SPX orders placed 2026-08-06 were invisible on 08-09 in both
    the dashboard and "any working orders?" — the default start date was today,
    so a resting order dropped off the moment the date rolled over."""

    def test_default_start_reaches_back_past_a_weekend(self):
        start = date.fromisoformat(tcfg.default_orders_start())
        self.assertLessEqual(start, date.today() - timedelta(days=7),
                             "a Friday order must still be listed on Monday")

    def test_default_start_is_not_today(self):
        self.assertNotEqual(tcfg.default_orders_start(), date.today().isoformat())


# Last in the file on purpose: unittest.main() only sees classes already
# defined, so anything added below it is silently skipped on a direct run.
if __name__ == "__main__":
    unittest.main()
