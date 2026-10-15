"""The Schwab variant, against a scripted HTTP fake: the 7-day sign-in and
its honest failures, account-hash resolution, and every parser that turns
Schwab's shapes into the package's dataclasses (so pnl.py stays one copy).
"""

import tempfile
import unittest
from pathlib import Path

from tests.trading_fixtures import FakeHTTP, FakeResponse, spx_chain
from trading import config as tcfg
from trading import pnl as tpnl
from trading import schwab
from trading.models import BrokerError, Leg
from trading.ticket import Ticket

DAY = 24 * 3600


def token_body(access="A1", refresh="R1"):
    return FakeResponse(200, {"access_token": access, "expires_in": 1800,
                              "refresh_token": refresh, "token_type": "Bearer"})


class SchwabBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = tcfg.TRADING_DIR
        tcfg.TRADING_DIR = Path(self.tmp.name)
        self.now = 1_000_000.0
        self.http = FakeHTTP()

    def tearDown(self):
        tcfg.TRADING_DIR = self._saved
        self.tmp.cleanup()

    def client(self, **kw):
        return schwab.Client(client_id="id", client_secret="sec", session=self.http,
                             clock=lambda: self.now, **kw)


class TestSignIn(SchwabBase):
    def test_no_tokens_means_sign_in_not_a_stack_trace(self):
        c = self.client()
        with self.assertRaises(BrokerError) as cm:
            c._token()
        self.assertEqual(cm.exception.code, "auth")
        self.assertIn("sign in", str(cm.exception))
        self.assertEqual(self.http.calls, [])   # nothing was tried

    def test_exchange_code_from_the_landing_url(self):
        self.http.script["POST /v1/oauth/token"] = token_body()
        c = self.client()
        c.exchange_code("https://127.0.0.1/?code=C%40xyz&session=s1")
        method, url, kw = self.http.calls[0]
        self.assertEqual(kw["data"]["grant_type"], "authorization_code")
        self.assertEqual(kw["data"]["code"], "C@xyz")   # URL-decoded, as the docs require
        self.assertTrue(kw["headers"]["Authorization"].startswith("Basic "))
        state = c.auth_state()
        self.assertTrue(state["signed_in"])
        self.assertAlmostEqual(state["hours_left"], 7 * 24, delta=0.1)
        # Survives a restart: the next client reads the file.
        self.assertTrue(self.client().auth_state()["signed_in"])

    def test_an_expired_code_says_why_and_how_to_retry(self):
        # The user's first sign-in failed with only "failed (400)" to go on.
        self.http.script["POST /v1/oauth/token"] = FakeResponse(400, {
            "error": "invalid_grant",
            "error_description": "Bad authorization code: AuthorizationCode has expired"})
        c = self.client()
        with self.assertLogs("trading", "WARNING") as logs, \
                self.assertRaises(BrokerError) as cm:
            c.exchange_code("https://127.0.0.1/?code=C%40old&session=s")
        msg = str(cm.exception)
        self.assertIn("AuthorizationCode has expired", msg)   # Schwab's own words
        self.assertIn("30 seconds", msg)
        self.assertIn("Sign in to Schwab again", msg)
        self.assertEqual(cm.exception.code, "auth")
        self.assertIn("AuthorizationCode has expired", logs.output[0])
        self.assertFalse(c.auth_state()["signed_in"])        # nothing half-saved

    def test_a_refused_refresh_points_to_sign_in_not_the_code(self):
        self.http.script["POST /v1/oauth/token"] = [token_body(), FakeResponse(
            401, {"error": "invalid_client", "error_description": "refresh token revoked"})]
        c = self.client()
        c.exchange_code("c")
        self.now += DAY
        with self.assertLogs("trading", "WARNING"), self.assertRaises(BrokerError) as cm:
            c._token()
        self.assertIn("refresh token revoked", str(cm.exception))
        self.assertNotIn("30 seconds", str(cm.exception))
        self.assertIn("sign in again", str(cm.exception))

    def test_access_refresh_keeps_the_seven_day_clock(self):
        self.http.script["POST /v1/oauth/token"] = [token_body("A1"), token_body("A2")]
        c = self.client()
        c.exchange_code("code1")
        self.now += 2 * DAY           # access long gone, refresh fine
        self.assertEqual(c._token(), "A2")
        self.assertEqual(self.http.calls[-1][2]["data"]["grant_type"], "refresh_token")
        self.assertAlmostEqual(c.auth_state()["hours_left"], 5 * 24, delta=0.1)

    def test_warns_a_day_ahead_and_refuses_after(self):
        self.http.script["POST /v1/oauth/token"] = token_body()
        c = self.client()
        c.exchange_code("code1")
        self.now += 6 * DAY + 3600
        self.assertIn("expires in", c.auth_state()["warning"])
        self.now += DAY
        state = c.auth_state()
        self.assertFalse(state["signed_in"])
        self.assertIn("expired", state["problem"])
        with self.assertRaises(BrokerError) as cm:
            c._token()
        self.assertEqual(cm.exception.code, "auth")


class TestAccountsAndOrders(SchwabBase):
    def signed_in(self):
        self.http.script["POST /v1/oauth/token"] = token_body()
        c = self.client()
        c.exchange_code("c")
        return c

    def test_every_shared_account_is_listed_and_reached_by_its_hash(self):
        # Two accounts used to be a hard stop ("set SCHWAB_ACCOUNT_ID");
        # now both are listed and each call names the one it means.
        self.http.script["GET /accounts/accountNumbers"] = FakeResponse(200, [
            {"accountNumber": "11112222", "hashValue": "H1"},
            {"accountNumber": "33334444", "hashValue": "H2"}])
        self.http.script["GET /accounts/H2"] = FakeResponse(200, {"securitiesAccount": {
            "currentBalances": {"liquidationValue": 5000.0, "cashBalance": 1200.0,
                                "cashAvailableForTrading": 1100.0}}})
        c = self.signed_in()
        self.assertEqual(c.account_numbers(), ["11112222", "33334444"])
        b = schwab.parse_balance(c.balances("33334444"))
        self.assertEqual((b.net_liq, b.cash, b.buying_power), (5000.0, 1200.0, 1100.0))
        with self.assertRaises(BrokerError):
            c.positions("99999999")             # not shared: said, not guessed

    def test_place_order_reads_the_id_from_location(self):
        self.http.script["GET /accounts/accountNumbers"] = FakeResponse(
            200, [{"accountNumber": "1", "hashValue": "H"}])
        self.http.script["POST /accounts/H/orders"] = FakeResponse(
            201, None, {"Location": "https://api.schwabapi.com/trader/v1/accounts/H/orders/987"})
        self.http.script["GET /accounts/H/orders/987"] = FakeResponse(
            200, {"orderId": 987, "status": "WORKING", "orderType": "NET_CREDIT",
                  "price": 1.5, "duration": "DAY",
                  "orderLegCollection": [{"instruction": "SELL_TO_OPEN", "quantity": 1,
                                          "instrument": {"symbol": "SPXW  260724P06345000",
                                                         "underlyingSymbol": "$SPX"}}]})
        order = schwab.parse_order(self.signed_in().place_order("1", {}))
        self.assertEqual((order.id, order.status, order.underlying), ("987", "Working", "SPX"))
        self.assertTrue(order.is_working)
        self.assertEqual(order.price_effect, "Credit")


class TestParsers(unittest.TestCase):
    def test_chain_prefers_standard_over_weekly_and_keys_quotes_by_occ(self):
        data = {"callExpDateMap": {"2026-08-21:30": {"6400.0": [
            {"symbol": "SPXW  260821C06400000", "strikePrice": 6400.0, "expirationType": "W"},
            {"symbol": "SPX   260821C06400000", "strikePrice": 6400.0, "expirationType": "S"}]}},
            "putExpDateMap": {"2026-08-21:30": {"6400.0": [
                {"symbol": "SPX   260821P06400000", "strikePrice": 6400.0, "expirationType": "S"}]},
                "2026-07-24:2": {"6350.0": [
                    {"symbol": "SPXW  260724P06350000", "strikePrice": 6350.0, "expirationType": "W"}]}}}
        chain = schwab._normalize_chain("SPX", data)
        self.assertEqual(chain.expirations, ["2026-07-24", "2026-08-21"])
        self.assertEqual(chain.option("2026-08-21", 6400, "C").occ, "SPX   260821C06400000")
        self.assertEqual(chain.option("2026-07-24", 6350, "P").streamer, "SPXW  260724P06350000")
        self.assertEqual(chain.underlying_streamer, "$SPX")

    def test_futures_are_refused_by_name(self):
        with self.assertRaises(ValueError) as cm:
            schwab.fetch_chain(None, "/ES")
        self.assertIn("Bonny", str(cm.exception))

    def test_order_payload_single_and_spread(self):
        chain = spx_chain()
        one = Ticket(underlying="SPX", legs=[Leg(6350.0, "Put", "2026-07-24", "Short")])
        one.limit_price, one.price_effect = 2.0, "Credit"
        p = schwab.order_payload(one, chain)
        self.assertEqual(p["orderType"], "LIMIT")
        self.assertEqual(p["orderLegCollection"][0]["instruction"], "SELL_TO_OPEN")
        self.assertEqual(p["orderLegCollection"][0]["instrument"],
                         {"symbol": "SPXW  260724P06350000", "assetType": "OPTION"})
        spread = Ticket(underlying="SPX", strategy="Bull Put Spread", tif="GTC", legs=[
            Leg(6345.0, "Put", "2026-07-24", "Short"), Leg(6340.0, "Put", "2026-07-24", "Long")])
        spread.limit_price, spread.price_effect = 1.5, "Credit"
        p = schwab.order_payload(spread, chain)
        self.assertEqual((p["orderType"], p["price"], p["duration"]),
                         ("NET_CREDIT", "1.50", "GOOD_TILL_CANCEL"))
        self.assertEqual([l["instruction"] for l in p["orderLegCollection"]],
                         ["SELL_TO_OPEN", "BUY_TO_OPEN"])

    def test_preview_rejects_become_errors_and_fees_add_up(self):
        r = schwab.parse_dry_run({
            "orderStrategy": {"orderBalance": {"projectedBuyingPower": 9000.5}},
            "orderValidationResult": {"rejects": [{"activityMessage": "Insufficient buying power"}],
                                      "warns": [{"message": "wide market"}]},
            "commissionAndFee": {
                "commission": {"commissionLegs": [{"commissionValues": [{"value": 0.65}]}]},
                "fee": {"feeLegs": [{"feeValues": [{"value": 0.05}, {"value": 0.02}]}]}}})
        self.assertFalse(r.ok)
        self.assertEqual(r.errors, ["Insufficient buying power"])
        self.assertEqual(r.warnings, ["wide market"])
        self.assertAlmostEqual(r.total_fees, 0.72)
        self.assertEqual(r.new_buying_power, 9000.5)
        self.assertIsNone(r.buying_power_change)   # Schwab projects, never a delta
        self.assertTrue(schwab.parse_dry_run({"orderValidationResult": {}}).ok)

    def test_order_statuses_map_onto_the_terminal_set(self):
        for raw, status, working in (("FILLED", "Filled", False), ("CANCELED", "Cancelled", False),
                                     ("PENDING_ACTIVATION", "Pending Activation", True)):
            o = schwab.parse_order({"orderId": 1, "status": raw, "orderType": "LIMIT",
                                    "orderLegCollection": [{"instruction": "BUY_TO_OPEN",
                                                            "instrument": {"symbol": "SPXW  260724C06350000"}}]})
            self.assertEqual((o.status, o.is_working), (status, working), raw)
            self.assertEqual(o.underlying, "SPXW")   # from the OCC when the leg names none
            self.assertEqual(o.price_effect, "Debit")

    def test_positions_sign_multiplier_and_schwabs_own_mark(self):
        pos = schwab.parse_positions([
            {"longQuantity": 0, "shortQuantity": 2, "averagePrice": 1.5, "marketValue": -260.0,
             "instrument": {"assetType": "OPTION", "symbol": "SPXW  260724P06345000",
                            "underlyingSymbol": "$SPX"}},
            {"longQuantity": 10, "shortQuantity": 0, "averagePrice": 100.0, "marketValue": 1050.0,
             "instrument": {"assetType": "EQUITY", "symbol": "AAPL"}},
            {"longQuantity": 0, "shortQuantity": 0, "instrument": {"assetType": "EQUITY", "symbol": "X"}}])
        self.assertEqual(len(pos), 2)
        p = pos[0]
        self.assertEqual((p.quantity, p.multiplier, p.underlying), (-2, 100.0, "SPX"))
        self.assertAlmostEqual(p.close_price, 1.3)       # -260 / (-2 * 100)
        self.assertAlmostEqual(p.unrealized(1.3), 40.0)  # sold at 1.50, now 1.30, two contracts
        self.assertAlmostEqual(pos[1].close_price, 105.0)

    def test_transactions_normalize_into_the_shared_pnl_math(self):
        def trade(ts, amount, cost, net, effect):
            return {"type": "TRADE", "time": ts, "netAmount": net, "transferItems": [
                {"instrument": {"assetType": "OPTION", "symbol": "SPXW  260724P06345000",
                                "underlyingSymbol": "$SPX"},
                 "amount": amount, "cost": cost, "positionEffect": effect},
                {"instrument": {"assetType": "CURRENCY", "symbol": "CURRENCY_USD"},
                 "amount": 0.65, "feeType": "COMMISSION"}]}

        class C:
            def transactions_all(self, account, start, end):
                return [trade("2026-07-20T15:00:00+0000", -1, 150.0, 149.35, "OPENING"),
                        trade("2026-07-22T15:00:00+0000", 1, -50.0, -50.65, "CLOSING")]
        txs = schwab.transactions(C(), "1", "2026-07-01", "2026-07-31")
        self.assertEqual([t["action"] for t in txs], ["Sell to Open", "Buy to Close"])
        self.assertEqual(txs[0]["underlying-symbol"], "SPX")
        rep = tpnl.report(txs, "2026-07-01", "2026-07-31")
        self.assertAlmostEqual(rep.realized, 98.70)     # 149.35 - 50.65
        self.assertAlmostEqual(rep.fees, 1.30)

    def test_transactions_window_is_clamped_to_a_year(self):
        seen = {}

        class C:
            def transactions_all(self, account, start, end):
                seen.update(start=start, end=end)
                return []
        schwab.transactions(C(), "1", "2020-01-01", "2026-07-31")
        self.assertEqual(seen["start"], "2025-07-31")


class TestRestQuotes(unittest.TestCase):
    def test_prefetch_is_one_call_and_cached_within_ttl(self):
        class C:
            def __init__(self):
                self.calls = []

            def quotes(self, symbols):
                self.calls.append(list(symbols))
                return {s: {"quote": {"bidPrice": 1.0, "askPrice": 1.2, "volatility": 14.2,
                                      "delta": 0.4, "gamma": 0.01, "theta": -0.5, "vega": 0.3}}
                        for s in symbols}
        c = C()
        q = schwab.make_quotes(c)

        def get(symbol):          # the engine's closure, with its batching hook
            return q.get(symbol)
        get.prefetch = q.prefetch
        tk = Ticket(underlying="SPX", strategy="Bull Put Spread", legs=[
            Leg(6345.0, "Put", "2026-07-24", "Short"), Leg(6340.0, "Put", "2026-07-24", "Long")])
        quotes = tk.leg_quotes(spx_chain(), get)
        self.assertEqual(len(c.calls), 1)               # both legs in one request
        self.assertEqual(len(c.calls[0]), 2)
        self.assertAlmostEqual(quotes[0][1].iv, 0.142)  # percent -> fraction
        self.assertAlmostEqual(tk.net_of(quotes), 0.0)
        q.get(quotes[0][1].symbol)
        self.assertEqual(len(c.calls), 1)               # fresh enough: no second call


class TestLegacyFiles(unittest.TestCase):
    def test_tastytrades_old_ticket_and_log_move_under_its_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = tcfg.TRADING_DIR
            tcfg.TRADING_DIR = Path(tmp)
            try:
                (Path(tmp) / "orders_log.json").write_text("[]")
                tcfg.ensure_dirs("tastytrade")
                self.assertTrue(tcfg.orders_log_path("tastytrade").is_file())
                self.assertFalse((Path(tmp) / "orders_log.json").exists())
                tcfg.ensure_dirs("schwab")
                self.assertFalse(tcfg.orders_log_path("schwab").exists())
            finally:
                tcfg.TRADING_DIR = saved


if __name__ == "__main__":
    unittest.main()
