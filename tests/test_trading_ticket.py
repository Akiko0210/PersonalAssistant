import tempfile
import unittest
from pathlib import Path

from tests.trading_fixtures import es_chain, quotes_for, spx_chain
from trading import config as tcfg
from trading import ticket as tticket
from trading.models import Leg
from trading.ticket import Ticket


class TicketBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = (tcfg.TRADING_DIR, tcfg.TICKET_PATH)
        tcfg.TRADING_DIR = Path(self.tmp.name)
        tcfg.TICKET_PATH = Path(self.tmp.name) / "ticket.json"
        self.chain = spx_chain()

    def tearDown(self):
        tcfg.TRADING_DIR, tcfg.TICKET_PATH = self._saved
        self.tmp.cleanup()

    def condor(self):
        return Ticket(underlying="SPX", strategy="Iron Condor", legs=[
            Leg(6345.0, "Put", "2026-07-24", "Short"),
            Leg(6340.0, "Put", "2026-07-24", "Long"),
            Leg(6355.0, "Call", "2026-07-24", "Short"),
            Leg(6360.0, "Call", "2026-07-24", "Long"),
        ])


class TestValidate(TicketBase):
    def test_valid_ticket_passes(self):
        self.assertEqual(self.condor().validate(self.chain), [])

    def test_bad_strike(self):
        tk = self.condor()
        tk.legs[0].strike = 6342.0
        problems = tk.validate(self.chain)
        self.assertEqual(len(problems), 1)
        self.assertIn("6342", problems[0])
        self.assertIn("6340", problems[0])  # closest-strike hint

    def test_bad_expiration(self):
        tk = self.condor()
        tk.legs[2].expiration = "2026-07-25"
        self.assertIn("not a listed expiration",
                      " ".join(tk.validate(self.chain)))

    def test_bad_size_and_side(self):
        tk = self.condor()
        tk.legs[0].size = 0
        tk.legs[1].side = "Sideways"
        problems = " ".join(tk.validate(self.chain))
        self.assertIn("size", problems)
        self.assertIn("side", problems)

    def test_too_many_legs(self):
        tk = self.condor()
        tk.legs.append(Leg(6350.0, "Call", "2026-07-24", "Long"))
        self.assertIn("at most 4",
                      " ".join(tk.validate(self.chain)))

    def test_single_expiry_strategy_rejects_spread_expirations(self):
        tk = self.condor()
        tk.legs[3].expiration = "2026-07-31"
        self.assertIn("single expiration",
                      " ".join(tk.validate(self.chain)))

    def test_multi_expiry_strategy_allows_it(self):
        tk = Ticket(underlying="SPX", strategy="Calendar Spread", legs=[
            Leg(6350.0, "Call", "2026-07-24", "Short"),
            Leg(6350.0, "Call", "2026-08-07", "Long"),
        ])
        self.assertEqual(tk.validate(self.chain), [])


class TestEditing(TicketBase):
    def test_strike_lock_moves_partners(self):
        tk = Ticket(underlying="SPX", strategy="Long Straddle", legs=[
            Leg(6350.0, "Call", "2026-07-24", "Long"),
            Leg(6350.0, "Put", "2026-07-24", "Long"),
        ])
        moved = tk.set_strike(0, 6362.0, self.chain)  # snaps to 6360
        self.assertEqual(moved, [0, 1])
        self.assertEqual([l.strike for l in tk.legs], [6360.0, 6360.0])

    def test_unlocked_strike_moves_alone(self):
        tk = self.condor()
        tk.set_strike(1, 6335.0, self.chain)
        self.assertEqual(tk.legs[1].strike, 6335.0)
        self.assertEqual(tk.legs[0].strike, 6345.0)

    def test_edit_invalidates_review(self):
        tk = self.condor()
        tk.review = {"fingerprint": tk.fingerprint()}
        tk.set_strike(1, 6335.0, self.chain)
        self.assertIsNone(tk.review)


class TestPricingAndPayload(TicketBase):
    def test_net_mid_credit_positive(self):
        tk = self.condor()
        # Same mid (10.2) on all legs: 2 short - 2 long = 0.
        self.assertAlmostEqual(
            tk.net_mid(self.chain, quotes_for(self.chain)), 0.0)
        # Single short leg -> positive (credit).
        short = Ticket(underlying="SPX", legs=[
            Leg(6350.0, "Put", "2026-07-24", "Short")])
        self.assertAlmostEqual(
            short.net_mid(self.chain, quotes_for(self.chain)), 10.2)

    def test_net_mid_none_when_quote_missing(self):
        tk = self.condor()
        self.assertIsNone(tk.net_mid(self.chain, lambda s: None))

    def test_apply_net_price_sets_effect(self):
        tk = self.condor()
        tk.apply_net_price(1.234)
        self.assertEqual((tk.limit_price, tk.price_effect), (1.23, "Credit"))
        tk.apply_net_price(-2.5)
        self.assertEqual((tk.limit_price, tk.price_effect), (2.5, "Debit"))

    def test_payload_shape(self):
        tk = self.condor()
        tk.limit_price, tk.price_effect = 1.50, "Credit"
        p = tk.payload(self.chain)
        self.assertEqual(p["order-type"], "Limit")
        self.assertEqual(p["price"], "1.50")
        self.assertEqual(p["price-effect"], "Credit")
        self.assertEqual(len(p["legs"]), 4)
        leg0 = p["legs"][0]
        self.assertEqual(leg0["instrument-type"], "Equity Option")
        self.assertEqual(leg0["action"], "Sell to Open")
        self.assertEqual(leg0["symbol"], "SPXW  260724P06345000")

    def test_payload_detects_future_options(self):
        chain = es_chain()
        tk = Ticket(underlying="/ES", legs=[
            Leg(6400.0, "Call", "2026-08-21", "Long")])
        tk.limit_price, tk.price_effect = 10.0, "Debit"
        p = tk.payload(chain)
        self.assertEqual(p["legs"][0]["instrument-type"], "Future Option")

    def test_payload_requires_price(self):
        with self.assertRaises(ValueError):
            self.condor().payload(self.chain)


class TestPersistence(TicketBase):
    def test_round_trip(self):
        tk = self.condor()
        tk.limit_price, tk.price_effect = 1.10, "Credit"
        tticket.save(tk)
        back = tticket.load()
        self.assertEqual(back.fingerprint(), tk.fingerprint())
        self.assertEqual(back.describe(), tk.describe())

    def test_load_missing(self):
        self.assertIsNone(tticket.load())

    def test_clear(self):
        tticket.save(self.condor())
        tticket.clear()
        self.assertIsNone(tticket.load())


if __name__ == "__main__":
    unittest.main()
