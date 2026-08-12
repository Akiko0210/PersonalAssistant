import unittest

from tests.trading_fixtures import es_chain, spx_chain
from trading import strategies as tstrat


class TestMatch(unittest.TestCase):
    def test_exact_and_case(self):
        self.assertEqual(tstrat.match_strategy("Iron Condor"), "Iron Condor")
        self.assertEqual(tstrat.match_strategy("iron  condor"), "Iron Condor")

    def test_exact_name_beats_containment(self):
        # "condor" is exactly the Condor strategy, even though it is also a
        # substring of Iron Condor.
        self.assertEqual(tstrat.match_strategy("condor"), "Condor")

    def test_ambiguous_containment_rejected(self):
        # "straddle" matches Long and Short Straddle -> refuse to guess.
        self.assertIsNone(tstrat.match_strategy("straddle"))
        self.assertIsNone(tstrat.match_strategy("spread"))
        self.assertIsNone(tstrat.match_strategy("nonsense"))

    def test_unique_containment_allowed(self):
        self.assertEqual(tstrat.match_strategy("double diagonal"),
                         "Double Diagonal")


class TestBuildLegs(unittest.TestCase):
    def setUp(self):
        self.chain = spx_chain()
        self.price = 6351.0  # ATM should be 6350

    def test_every_strategy_builds(self):
        for name in tstrat.STRATEGIES:
            legs = tstrat.build_legs(name, self.chain, self.price)
            self.assertTrue(legs, name)
            for leg in legs:
                self.assertIn(leg.expiration, self.chain.expirations, name)
                self.assertIn(leg.strike,
                              self.chain.strikes[leg.expiration], name)

    def test_iron_condor_shape(self):
        legs = tstrat.build_legs("Iron Condor", self.chain, self.price)
        # 5-point spacing -> wing step 1 -> B=6345 A=6340 C=6355 D=6360.
        self.assertEqual(
            [(l.strike, l.cp, l.side) for l in legs],
            [(6345.0, "Put", "Short"), (6340.0, "Put", "Long"),
             (6355.0, "Call", "Short"), (6360.0, "Call", "Long")])

    def test_butterfly_middle_is_double_size(self):
        legs = tstrat.build_legs("Long Butterfly", self.chain, self.price,
                                 size=2)
        self.assertEqual([l.size for l in legs], [2, 4, 2])

    def test_calendar_uses_two_expirations(self):
        legs = tstrat.build_legs("Calendar Spread", self.chain, self.price)
        self.assertEqual(legs[0].expiration, self.chain.expirations[0])
        self.assertEqual(legs[1].expiration, self.chain.expirations[3])
        self.assertEqual(legs[0].strike, legs[1].strike)

    def test_wing_step_scales_with_spacing(self):
        # /ES fixture has 25-point spacing -> wing step stays 1 strike.
        chain = es_chain()
        legs = tstrat.build_legs("Iron Condor", chain, 6400.0)
        self.assertEqual([l.strike for l in legs],
                         [6375.0, 6350.0, 6425.0, 6450.0])

    def test_explicit_expiration(self):
        legs = tstrat.build_legs("Long Call", self.chain, self.price,
                                 expiration="2026-07-31")
        self.assertEqual(legs[0].expiration, "2026-07-31")

    def test_unknown_strategy(self):
        self.assertEqual(tstrat.build_legs("Nope", self.chain, 1.0), [])


if __name__ == "__main__":
    unittest.main()
