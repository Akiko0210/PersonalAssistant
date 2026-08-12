import unittest

from trading.chains import _normalize_equity, _normalize_futures


def strike_row(strike, call, put):
    return {"strike-price": str(strike),
            "call": call, "call-streamer-symbol": "." + call.replace(" ", ""),
            "put": put, "put-streamer-symbol": "." + put.replace(" ", "")}


class TestEquityNormalize(unittest.TestCase):
    def payload(self):
        return {"items": [{
            "option-chain-type": "Standard",
            "expirations": [
                {"expiration-date": "2026-07-24", "expiration-type": "Weekly",
                 "strikes": [strike_row(6350, "SPXW  260724C06350000",
                                        "SPXW  260724P06350000")]},
                # Same date twice: Regular must win over Weekly.
                {"expiration-date": "2026-08-21", "expiration-type": "Weekly",
                 "strikes": [strike_row(6300, "SPXW  260821C06300000",
                                        "SPXW  260821P06300000")]},
                {"expiration-date": "2026-08-21", "expiration-type": "Regular",
                 "strikes": [strike_row(6400, "SPX   260821C06400000",
                                        "SPX   260821P06400000")]},
            ]}]}

    def test_normalize(self):
        chain = _normalize_equity("SPX", self.payload())
        self.assertEqual(chain.expirations, ["2026-07-24", "2026-08-21"])
        self.assertEqual(chain.strikes["2026-08-21"], [6400.0])
        info = chain.option("2026-08-21", 6400, "C")
        self.assertEqual(info.occ, "SPX   260821C06400000")
        self.assertFalse(chain.is_futures)

    def test_non_standard_chains_skipped(self):
        p = self.payload()
        p["items"].append({"option-chain-type": "Non-standard",
                           "expirations": [
                               {"expiration-date": "2026-09-01",
                                "expiration-type": "Regular",
                                "strikes": []}]})
        chain = _normalize_equity("SPX", p)
        self.assertNotIn("2026-09-01", chain.expirations)


class TestFuturesNormalize(unittest.TestCase):
    def test_normalize(self):
        data = {"option-chains": [{
            "expirations": [{
                "expiration-date": "2026-09-18",
                "expiration-type": "Regular",
                "underlying-symbol": "/ESU6",
                "strikes": [strike_row(6400, "./ESU6 EW3U6 260918C6400",
                                       "./ESU6 EW3U6 260918P6400")]}]}]}
        futures = [{"symbol": "/ESU6", "active-month": True,
                    "streamer-symbol": "/ESU26:XCME"}]
        chain = _normalize_futures("/ES", data, futures)
        self.assertTrue(chain.is_futures)
        self.assertEqual(chain.underlying_streamer, "/ESU26:XCME")
        self.assertEqual(chain.expiration_contract["2026-09-18"], "/ESU6")
        self.assertEqual(chain.option("2026-09-18", 6400, "C").occ,
                         "./ESU6 EW3U6 260918C6400")

    def test_front_month_default_without_instruments(self):
        chain = _normalize_futures("/ES", {"option-chains": []}, [])
        self.assertEqual(chain.underlying_streamer, "/ES:XCME")


if __name__ == "__main__":
    unittest.main()
