import unittest

from trading import symbols as tsym


class TestOcc(unittest.TestCase):
    def test_build_matches_known_format(self):
        # Doc example: SPX May 20 2022 4025 call (weekly root).
        self.assertEqual(tsym.build_occ("SPXW", "2022-05-20", "Call", 4025),
                         "SPXW  220520C04025000")

    def test_build_pads_root_and_fractional_strike(self):
        self.assertEqual(tsym.build_occ("SPY", "2023-07-06", "P", 370.5),
                         "SPY   230706P00370500")

    def test_parse_round_trip(self):
        occ = tsym.build_occ("SPXW", "2026-02-27", "C", 6990)
        p = tsym.parse_occ(occ)
        self.assertEqual(p, {"root": "SPXW", "expiration": "2026-02-27",
                             "cp": "C", "strike": 6990.0})

    def test_parse_future_option(self):
        p = tsym.parse_occ("./ESU6 EW3U6 250919C6000")
        self.assertEqual(p["root"], "/ESU6")
        self.assertEqual(p["expiration"], "2025-09-19")
        self.assertEqual(p["cp"], "C")
        self.assertEqual(p["strike"], 6000.0)

    def test_parse_rejects_non_options(self):
        self.assertIsNone(tsym.parse_occ("SPX"))
        self.assertIsNone(tsym.parse_occ("/ESU6"))


class TestNormalize(unittest.TestCase):
    def test_spoken_forms(self):
        self.assertEqual(tsym.normalize_underlying("spx"), "SPX")
        self.assertEqual(tsym.normalize_underlying("e-mini"), "/ES")
        self.assertEqual(tsym.normalize_underlying("/es"), "/ES")
        self.assertEqual(tsym.normalize_underlying("micros"), "/MES")
        self.assertEqual(tsym.normalize_underlying("aapl"), "AAPL")
        self.assertEqual(tsym.normalize_underlying(""), "")

    def test_futures_helpers(self):
        self.assertTrue(tsym.is_futures("/ES"))
        self.assertFalse(tsym.is_futures("SPX"))
        self.assertEqual(tsym.product_code("/ES"), "ES")
        self.assertEqual(tsym.multiplier("/ES"), 50.0)
        self.assertEqual(tsym.multiplier("/MES"), 5.0)
        self.assertEqual(tsym.multiplier("SPX", "Equity Option"), 100.0)
        self.assertEqual(tsym.multiplier("AAPL", "Equity"), 1.0)


if __name__ == "__main__":
    unittest.main()
