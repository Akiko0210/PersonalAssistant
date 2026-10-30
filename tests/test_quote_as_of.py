"""A quote says when it is from (#55): Schwab's market time and "Closed"
status reach the Quote, dxFeed's trade time does too, a failed Schwab
refresh is no quote rather than the old one, and get_quote puts it in words
— the close is never read out as "right now" again.
"""

import time
import unittest

from tests.test_broker_routing import run
from tests.trading_fixtures import FakeEngine
from trading import config as tcfg
from trading import schwab
from trading.models import Quote
from trading.quotes import QuoteService

# Shapes from the live /quotes call of 2026-10-29 (values trimmed).
SPX_AFTER_CLOSE = {"lastPrice": 7720.39, "closePrice": 7743.98,
                   "securityStatus": "Closed", "tradeTime": 1793305967111}
ES_LIVE = {"bidPrice": 7703.75, "askPrice": 7704.25, "lastPrice": 7704.0,
           "securityStatus": "Normal", "tradeTime": 1793319334675,
           "quoteTime": 1793319335156}


class SchwabClient:
    def __init__(self, quotes):
        self.answer, self.calls = quotes, 0

    def quotes(self, symbols):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return {s: {"quote": self.answer[s]} for s in symbols if s in self.answer}


class TestSchwabMarketTime(unittest.TestCase):
    def test_closed_index_carries_status_and_trade_time(self):
        q = schwab.make_quotes(SchwabClient({"$SPX": SPX_AFTER_CLOSE})).get("$SPX")
        self.assertTrue(q.closed)
        self.assertAlmostEqual(q.as_of, 1793305967.111)

    def test_futures_take_the_newer_of_trade_and_quote_time(self):
        q = schwab.make_quotes(SchwabClient({"/ESZ26": ES_LIVE})).get("/ESZ26")
        self.assertFalse(q.closed)
        self.assertAlmostEqual(q.as_of, 1793319335.156)

    def test_a_failed_refresh_is_no_quote_not_the_old_one(self):
        client = SchwabClient({"$SPX": SPX_AFTER_CLOSE})
        quotes = schwab.make_quotes(client)
        self.assertIsNotNone(quotes.get("$SPX"))
        quotes._quotes["$SPX"].updated_at -= tcfg.SCHWAB_QUOTE_TTL_S + 1
        client.answer = RuntimeError("502 Bad Gateway")
        with self.assertLogs("trading", "WARNING"):
            self.assertIsNone(quotes.get("$SPX"))
        self.assertIn("502", quotes.last_error)
        self.assertEqual(client.calls, 2)


class TestStreamTradeTime(unittest.TestCase):
    def test_trade_event_time_becomes_as_of(self):
        svc = QuoteService(client=None)    # no start(): no thread, no network
        svc._feed_data(["Trade", ["/ESZ26:XCME", 7704.0, 1793319334675]])
        q = svc.get("/ESZ26:XCME")
        self.assertEqual(q.last, 7704.0)
        self.assertAlmostEqual(q.as_of, 1793319334.675)


def say(quote, env="live", symbol="SPX"):
    return run("tom", "get_quote", {"symbol": symbol},
               engines={"schwab": FakeEngine("schwab", quote=quote, env=env)})


class TestGetQuoteWords(unittest.TestCase):
    def test_closed_market_says_not_live_with_the_time(self):
        out = say(Quote("$SPX", last=7720.39, as_of=time.time() - 600, closed=True))
        self.assertIn("last 7,720.39", out)
        self.assertIn("Market closed", out)
        self.assertIn("not a live one", out)
        self.assertIn("last update", out)

    def test_old_price_without_status_may_not_be_live(self):
        out = say(Quote("$SPX", last=7720.39,
                        as_of=time.time() - tcfg.QUOTE_STALE_S - 300))
        self.assertIn("minutes ago", out)
        self.assertIn("may not be live", out)

    def test_fresh_price_says_as_of(self):
        out = say(Quote("$SPX", bid=7703.75, ask=7704.25, as_of=time.time()))
        self.assertIn("As of", out)
        self.assertNotIn("not live", out)

    def test_futures_quote_names_its_contract(self):
        out = say(Quote("/ESZ26", bid=7703.75, ask=7704.25, as_of=time.time()),
                  symbol="/ES")
        self.assertTrue(out.startswith("/ES (/ESZ26): bid 7,703.75"), out)
        self.assertTrue(say(Quote("$SPX", last=1.0, as_of=time.time())
                            ).startswith("SPX: "))   # an index keeps its name

    def test_small_prices_keep_their_decimals(self):
        out = say(Quote("/6EZ26", bid=1.12345, ask=1.1235, last=1.12, as_of=time.time()),
                  symbol="/6E")
        self.assertIn("bid 1.12345, ask 1.1235", out)
        self.assertIn("last 1.12. ", out)                 # not 1.12000

    def test_no_time_from_the_broker_is_said(self):
        self.assertIn("no time", say(Quote("$SPX", last=7720.39)))

    def test_no_quote_on_live_schwab_does_not_blame_sandbox(self):
        out = say(Quote("$SPX"))
        self.assertIn("No quote available", out)
        self.assertNotIn("sandbox", out)
        self.assertIn("sandbox", say(Quote("$SPX"), env="sandbox"))


if __name__ == "__main__":
    unittest.main()
