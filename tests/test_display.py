"""The display window (#28): a tool answer becomes the card the window shows
(the same string, never a second fetch), the route serves it honestly with
or without an agent, and the window comes forward only for a new card.
"""

import unittest
from unittest import mock

from display import main as display
from tests.agent_fixtures import make_agent
from tests.trading_fixtures import FakeEngine
from tools import ToolContext, dispatch
from web import server as dashboard


def run(persona, tool, args=None, show=None, engines=None):
    """Dispatch `tool` as `persona`, cards going to `show`, engines faked."""
    engines = engines or {"schwab": FakeEngine("schwab", realized=-62.5)}

    def fake(broker):
        eng = engines.get(broker)
        return (eng, None) if eng else (None, f"{broker} not configured")
    with mock.patch("tools.trading_tools._engine_for", side_effect=fake):
        return dispatch(ToolContext(active_agent=persona, show=show),
                        tool, args or {})


class SeamTests(unittest.TestCase):
    def test_a_trading_answer_goes_on_the_display_word_for_word(self):
        shown = []
        out = run("tom", "get_pnl", show=lambda *card: shown.append(card))
        self.assertEqual(shown, [("Profit and loss — Tom, Charles Schwab", out)])
        # The sign says loss, not just a colour.
        self.assertIn("realized -62.50", out)

    def test_linda_names_the_broker_she_was_asked_about(self):
        shown = []
        run("linda", "get_pnl", {"broker": "tastytrade"},
            show=lambda *card: shown.append(card),
            engines={"tastytrade": FakeEngine("tastytrade", realized=5.0)})
        self.assertEqual(shown[0][0], "Profit and loss — Linda, tastytrade")
        self.assertIn("realized +5.00", shown[0][1])

    def test_a_refusal_replaces_the_card_rather_than_leaving_old_numbers(self):
        shown = []
        out = run("alice", "get_positions", show=lambda *card: shown.append(card))
        self.assertEqual(shown, [("Open positions — Alice", out)])

    def test_tools_without_display_leave_the_card_alone(self):
        shown = []
        out = run("tom", "get_current_time", show=lambda *card: shown.append(card))
        self.assertNotIn("Unknown tool", out)
        self.assertEqual(shown, [])

    def test_no_window_wired_is_fine(self):
        # A delegated task's fresh ToolContext has no show.
        self.assertIn("realized", run("tom", "get_pnl"))

    def test_a_broken_display_never_costs_the_answer(self):
        def boom(*_):
            raise RuntimeError("window gone")
        with self.assertLogs("tools", "ERROR"):
            out = run("tom", "get_pnl", show=boom)
        self.assertIn("realized -62.50", out)

    def test_the_trade_log_is_always_schwab(self):
        from tools import _DISPLAY
        heading = _DISPLAY["query_trade_log"](ToolContext(active_agent="linda"), {})
        self.assertEqual(heading, "Trades from the trade log — Linda, Charles Schwab")


class CardTests(unittest.TestCase):
    def test_one_statement_per_line_and_numbers_stay_whole(self):
        card = display.make_card(
            "Open positions", "IRA: long 1 SPX (avg 1,234.50, unrealized "
            "+45.00). short 2 SPX (avg 3.10, no mark). Total unrealized "
            "+45.00.\nSecond line", 3)
        self.assertEqual(card["lines"], [
            "IRA: long 1 SPX (avg 1,234.50, unrealized +45.00).",
            "short 2 SPX (avg 3.10, no mark).",
            "Total unrealized +45.00.",
            "Second line"])
        self.assertEqual((card["version"], card["title"]), (3, "Open positions"))
        self.assertRegex(card["as_of"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")

    def test_each_card_is_a_new_version_and_reopens_the_window(self):
        agent = make_agent()
        agent.card, agent.display = None, mock.Mock()
        agent.show_card("Orders — Tom", "No orders.")
        agent.show_card("Orders — Tom", "Order 1 filled.")
        self.assertEqual(agent.card["version"], 2)
        self.assertEqual(agent.card["lines"], ["Order 1 filled."])
        self.assertEqual(agent.display.open.call_count, 2)


class RouteTests(unittest.TestCase):
    def test_the_agent_serves_its_card(self):
        agent = make_agent()
        agent.card = display.make_card("Orders", "No orders.", 1)
        self.assertEqual(dashboard.api_display(agent),
                         {"agent_running": True, "card": agent.card})

    def test_standalone_says_the_agent_is_not_here(self):
        self.assertEqual(dashboard.api_display(None),
                         {"agent_running": False, "card": None})

    def test_an_unreachable_agent_reads_as_not_running(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            state = display.fetch(1)
        self.assertFalse(state["agent_running"])
        self.assertIsNone(state["card"])
        self.assertIn("stale_after_s", state)  # the page still words staleness


class WindowTests(unittest.TestCase):
    def test_comes_forward_only_for_a_new_card(self):
        self.assertTrue(display.is_new(2, 1))
        self.assertTrue(display.is_new(1, None))   # first card after none
        self.assertTrue(display.is_new(1, 7))      # agent restarted
        self.assertFalse(display.is_new(3, 3))     # a poll with no change
        self.assertFalse(display.is_new(None, 3))  # the agent went away

    def test_relaunches_only_when_the_window_is_gone(self):
        with mock.patch("importlib.util.find_spec", return_value=object()), \
                mock.patch("subprocess.Popen") as popen:
            w = display.Window(8765).open()
            popen.return_value.poll.return_value = None   # still open
            w.open()
            self.assertEqual(popen.call_count, 1)
            popen.return_value.poll.return_value = 0      # user closed it
            w.open()
            self.assertEqual(popen.call_count, 2)

    def test_off_without_pywebview_or_a_dashboard(self):
        with mock.patch("subprocess.Popen") as popen:
            with mock.patch("importlib.util.find_spec", return_value=None), \
                    self.assertLogs("display", "WARNING"):
                display.Window(8765).open()
            display.Window(None).open()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
