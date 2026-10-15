"""Persona -> broker routing: Tom reaches Schwab only, Bonny tastytrade
only, Linda reads both and can focus on one; the dashboard names its broker
on every call. The trading tools pick the broker from ctx.active_agent, so
there is no argument through which a persona can be talked onto the other
account — asking for it is refused with the right name.
"""

import unittest
from unittest import mock

from brain import agents
from tests.trading_fixtures import FakeEngine
from tools import ToolContext, api_tools, dispatch
from trading import engine as tengine
from trading import main as tmain
from trading import web_api


def run(persona, tool, args=None, engines=None):
    """Dispatch `tool` as `persona` with get_engine faked per broker."""
    engines = engines or {}

    def fake(broker):
        eng = engines.get(broker)
        return (eng, None) if eng else (None, f"{broker} not configured")
    with mock.patch("tools.trading_tools._engine_for", side_effect=fake):
        return dispatch(ToolContext(active_agent=persona), tool, args or {})


class TestRegistry(unittest.TestCase):
    def test_each_trading_persona_has_exactly_one_broker(self):
        self.assertEqual(agents.AGENTS["tom"]["brokers"], ("schwab",))
        self.assertEqual(agents.AGENTS["bonny"]["brokers"], ("tastytrade",))
        self.assertEqual(agents.AGENTS["linda"]["brokers"], ("schwab", "tastytrade"))
        self.assertEqual(agents.AGENTS["alice"]["brokers"], ())
        for key in tmain.keys():
            self.assertIn(key, agents.AGENTS["linda"]["brokers"])

    def test_tom_and_bonny_hold_every_trading_tool_linda_reads_only(self):
        for key in ("tom", "bonny"):
            self.assertTrue(agents.TRADING_TOOLS <= agents.AGENTS[key]["tools"], key)
        linda = {t["name"] for t in api_tools(include=agents.AGENTS["linda"]["tools"])}
        self.assertTrue(agents.TRADING_READ_TOOLS <= linda)
        for name in ("submit_order", "build_strategy", "cancel_order", "get_quote"):
            self.assertNotIn(name, linda)

    def test_bonny_is_addressable(self):
        self.assertEqual(agents.resolve("Bonnie"), "bonny")
        self.assertEqual(agents.match_address("Bonny, what's my P&L?")[0], "bonny")
        self.assertIn("Bonny", agents.roster_block("tom"))

    def test_broker_names_normalize(self):
        self.assertEqual(tmain.normalize("Charles Schwab"), "schwab")
        self.assertEqual(tmain.normalize("thinkorswim"), "schwab")
        self.assertEqual(tmain.normalize("tasty"), "tastytrade")
        self.assertIsNone(tmain.normalize("robinhood"))


class TestRefusals(unittest.TestCase):
    def test_tom_asked_for_tastytrade_sends_the_user_to_bonny(self):
        out = run("tom", "get_positions", {"broker": "tastytrade"},
                  engines={"tastytrade": FakeEngine("tastytrade")})
        self.assertIn("Bonny", out)
        self.assertIn("Charles Schwab only", out)

    def test_bonny_asked_for_schwab_sends_the_user_to_tom(self):
        out = run("bonny", "build_strategy", {"strategy": "Iron Condor", "symbol": "SPX",
                                               "broker": "schwab"},
                  engines={"schwab": FakeEngine("schwab")})
        self.assertIn("Tom", out)
        self.assertIn("tastytrade only", out)

    def test_a_persona_without_a_broker_does_not_trade(self):
        out = run("alice", "get_pnl", engines={"schwab": FakeEngine("schwab")})
        self.assertIn("doesn't trade", out)
        self.assertIn("Bonny", out)

    def test_unknown_broker_name(self):
        self.assertIn("don't know a broker", run("tom", "get_pnl", {"broker": "robinhood"}))

    def test_default_broker_is_the_personas_own(self):
        schwab, tasty = FakeEngine("schwab"), FakeEngine("tastytrade")
        run("tom", "get_positions", engines={"schwab": schwab, "tastytrade": tasty})
        self.assertEqual(schwab.calls, [("positions", "schwab-1", None)])
        self.assertEqual(tasty.calls, [])


class TestLindaReadsBoth(unittest.TestCase):
    def test_pnl_is_per_broker_and_only_summed_in_the_sentence(self):
        engines = {"schwab": FakeEngine("schwab", realized=100.0),
                   "tastytrade": FakeEngine("tastytrade", realized=-30.0)}
        out = run("linda", "get_pnl", {"period": "today"}, engines=engines)
        self.assertIn("Charles Schwab, account ", out)   # broker, then account
        self.assertIn("tastytrade, account ", out)
        self.assertIn("realized 100.00", out)
        self.assertIn("realized -30.00", out)
        self.assertIn("All of them together, added here only: realized 70.00", out)
        for eng in engines.values():
            self.assertEqual(len(eng.calls), 1)   # each account asked once, separately

    def test_focus_on_one_broker(self):
        engines = {"schwab": FakeEngine("schwab", realized=100.0),
                   "tastytrade": FakeEngine("tastytrade", realized=-30.0)}
        out = run("linda", "get_pnl", {"period": "today", "broker": "schwab"}, engines=engines)
        self.assertNotIn("tastytrade", out)
        self.assertNotIn("Across both", out)
        self.assertEqual(engines["tastytrade"].calls, [])

    def test_an_unconfigured_broker_is_said_not_skipped(self):
        out = run("linda", "trading_status", engines={"schwab": FakeEngine("schwab")})
        self.assertIn("tastytrade: tastytrade not configured", out)
        self.assertIn("Charles Schwab: Trading is in live mode", out)


class TestEngineAndRoutes(unittest.TestCase):
    def test_unknown_broker_is_a_sentence_not_a_key_error(self):
        eng, err = tengine.get_engine("nope")
        self.assertIsNone(eng)
        self.assertIn("Unknown broker", err)

    def test_dashboard_lists_brokers_with_their_persona(self):
        rows = {b["key"]: b for b in web_api.brokers()["brokers"]}
        self.assertEqual(rows["schwab"]["persona"], "Tom")
        self.assertEqual(rows["tastytrade"]["persona"], "Bonny")
        self.assertIn("configured", rows["schwab"])

    def test_status_needs_a_real_broker(self):
        import tempfile
        from pathlib import Path
        from trading import config as tcfg
        self.assertIn("error", web_api.status(""))
        # A configured machine builds the engine here; keep its data/ out of
        # the user's.
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(tcfg, "TRADING_DIR", Path(tmp)), \
                mock.patch.dict(tengine._engines, {}, clear=True):
            self.assertEqual(web_api.status("schwab")["broker"], "schwab")


if __name__ == "__main__":
    unittest.main()
