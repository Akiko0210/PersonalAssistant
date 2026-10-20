"""The display window (#28): a tool answer becomes a card built from the
fixed blocks (display/card.py) out of the same figures as the spoken
sentence; only P&L is coloured; the route serves it honestly with or
without an agent; and the window comes forward only for a new card.
"""

import re
import unittest
from unittest import mock

from display import card as blocks
from display import main as display
from tests.agent_fixtures import make_agent
from tests.test_trade_log import LogBase
from tests.trading_fixtures import FakeEngine
from tools import ToolContext, _DISPLAY, dispatch
from tools.trading_tools import _ticket_card
from trading.models import Leg, PlacedOrder, Position
from trading.ticket import Ticket
from web import server as dashboard

FIGURE = re.compile(r"[+-]?\d[\d,]*\.\d\d")


def figures(card):
    """Every money figure on a card: values, totals and detail lines."""
    parts = [card["total"]] + [s.get("total") for s in card["sections"]]
    rows = [r for s in card["sections"] for r in s["rows"]]
    text = [p["value"] for p in parts if p] + [r["value"] or "" for r in rows]
    text += [line for r in rows for line in r["lines"]]
    return [f for t in text for f in FIGURE.findall(t)]


def run(persona, tool, args=None, engines=None):
    """Dispatch `tool` as `persona` with engines faked; (spoken, shown)."""
    engines = engines or {"schwab": FakeEngine("schwab", realized=-62.5)}
    shown = []

    def fake(broker):
        eng = engines.get(broker)
        return (eng, None) if eng else (None, f"{broker} not configured")
    with mock.patch("tools.trading_tools._engine_for", side_effect=fake):
        out = dispatch(ToolContext(active_agent=persona,
                                   show=lambda *c: shown.append(c)),
                       tool, args or {})
    return out, shown


LONG = Position("SPXW  261106P07750000", "SPX", "Equity Option", 1, 90.0, 100)
SHORT = Position("SPXW  261023P07700000", "SPX", "Equity Option", -1, 13.0, 100)
HOLDING = {"positions": [LONG, SHORT],
           "marks": {LONG.symbol: 96.4, SHORT.symbol: 15.1}}


class SeamTests(unittest.TestCase):
    def test_positions_card_figures_are_the_spoken_ones(self):
        out, [(heading, card)] = run("tom", "get_positions", engines={
            "schwab": FakeEngine("schwab", **HOLDING)})
        self.assertEqual(heading, "Open positions — Tom, Charles Schwab")
        self.assertEqual(card["total"], blocks.total("Total unrealized", "+430.00"))
        long_, short = card["sections"][0]["rows"]
        self.assertEqual((long_["badge"], long_["value"]), ("LONG", "+640.00"))
        self.assertEqual((short["badge"], short["value"]), ("SHORT", "-210.00"))
        self.assertTrue(long_["label"].startswith("SPX 7750 Put · Fri Nov 6"))
        for f in figures(card):
            self.assertIn(f, out)

    def test_several_accounts_get_a_section_each_and_a_combined_total(self):
        eng = FakeEngine("schwab", accounts={"00000699": 0, "00000868": 0},
                         **HOLDING)
        out, [(_, card)] = run("tom", "get_positions", {"account": "all"},
                               engines={"schwab": eng})
        self.assertEqual(len(card["sections"]), 2)
        self.assertEqual(card["total"]["value"], "+860.00")
        self.assertIn("+860.00", out)

    def test_orders_say_their_status_in_a_word_and_price_stays_white(self):
        order = PlacedOrder("42", "WORKING", "SPX", price=1.2, price_effect="Credit",
                            legs=[{"symbol": LONG.symbol, "action": "BUY_TO_OPEN",
                                   "quantity": 1}])
        out, [(_, card)] = run("tom", "list_orders", engines={
            "schwab": FakeEngine("schwab", orders=[order])})
        [row] = card["sections"][0]["rows"]
        self.assertEqual((row["badge"], row["value"], row["kind"]),
                         ("WORKING", "1.2 Credit", blocks.MONEY))
        self.assertEqual(row["lines"], ["Buy to open 1 SPX 7750 Put · Fri Nov 6"])

    def test_linda_names_the_broker_she_was_asked_about(self):
        _, [(heading, card)] = run("linda", "get_pnl", {"broker": "tastytrade"},
                                   engines={"tastytrade": FakeEngine("tastytrade",
                                                                     realized=5.0)})
        self.assertEqual(heading, "Profit and loss — Linda, tastytrade")
        self.assertEqual(card["total"]["value"], "+5.00")

    def test_a_refusal_shows_as_plain_notes(self):
        out, [(heading, text)] = run("alice", "get_positions")
        self.assertEqual((heading, text), ("Open positions — Alice", out))

    def test_tools_without_display_leave_the_card_alone(self):
        out, shown = run("tom", "get_current_time")
        self.assertNotIn("Unknown tool", out)
        self.assertEqual(shown, [])

    def test_a_broken_display_never_costs_the_answer(self):
        def boom(*_):
            raise RuntimeError("window gone")
        with mock.patch("tools.trading_tools._engine_for",
                        return_value=(FakeEngine("schwab", realized=-62.5), None)), \
                self.assertLogs("tools", "ERROR"):
            out = dispatch(ToolContext(active_agent="tom", show=boom), "get_pnl", {})
        self.assertIn("realized -62.50", out)

    def test_the_ticket_shows_its_price_first_and_numbers_its_legs(self):
        tk = Ticket(underlying="SPX", strategy="vertical", limit_price=1.2,
                    price_effect="Credit", legs=[
                        Leg(7700, "Put", "2026-11-06", "Short"),
                        Leg(7650, "Put", "2026-11-06", "Long")])
        card = _ticket_card(tk)
        self.assertEqual(card["total"]["value"], "1.20 Credit")
        self.assertEqual([r["badge"] for r in card["sections"][0]["rows"]],
                         ["SHORT", "LONG"])
        self.assertEqual(card["sections"][0]["rows"][1]["lines"],
                         ["Leg 2: Buy to Open, 1 contract"])


class TradeLogCardTests(LogBase):
    def test_trades_behind_a_position_open_legs_first_cash_not_pnl(self):
        for ref, line, hour in (
                ("1", "BOT +1 DIAGONAL SPX 100 (Weeklys) 4 NOV 26/15 OCT 26 7750/7760 PUT @90.00", 10),
                ("2", "BOT +1 VERTICAL SPX 100 (Weeklys) 15 OCT 26 7760/7730 PUT @20.30", 11)):
            self.call("log_trade", alert=f"#{ref} {line} , ACCOUNT *****839SCHW",
                      email_date=f"Wed, 14 Oct 2026 {hour}:00:00 -0400")
        self.call("assign_trades", refs=["1", "2"], new_position="Nov 4 diagonal")
        shown = []
        self.ctx.show = lambda *c: shown.append(c)
        out = self.call("query_trade_log", position=1)
        [(heading, card)] = shown
        self.assertEqual(heading, "Trades from the trade log — Tom, Charles Schwab")
        self.assertEqual(card["total"]["kind"], blocks.MONEY)   # legs remain
        self.assertIn("not P&L", card["total"]["label"])
        legs, trades = card["sections"]
        self.assertEqual(legs["heading"], "Open legs")
        self.assertIn("SHORT", [r["badge"] for r in legs["rows"]])
        self.assertEqual([r["badge"] for r in trades["rows"]], ["BOUGHT", "BOUGHT"])
        self.assertEqual({r["kind"] for r in trades["rows"]}, {blocks.MONEY})
        for f in figures(card):
            self.assertIn(f, out)


class CardTests(unittest.TestCase):
    def test_text_becomes_notes_one_statement_each_numbers_whole(self):
        card = display.make_card("Orders", "Avg 1,234.50 then. Second.\nThird", 3)
        self.assertEqual(card["notes"], ["Avg 1,234.50 then.", "Second.", "Third"])
        self.assertEqual((card["version"], card["title"]), (3, "Orders"))
        self.assertRegex(card["as_of"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")

    def test_one_sections_total_becomes_the_cards(self):
        one = blocks.card([blocks.section(total=blocks.total("T", "+1.00"))])
        self.assertEqual(one["total"]["value"], "+1.00")
        two = blocks.card([blocks.section(total=blocks.total("T", "+1.00")),
                           blocks.section()])
        self.assertIsNone(two["total"])

    def test_contract_names(self):
        self.assertEqual(blocks.symbol_name("AAPL"), "AAPL")
        self.assertTrue(blocks.symbol_name("SPXW  301115C06000000")
                        .endswith("Fri Nov 15 2030"))

    def test_each_card_is_a_new_version_and_reopens_the_window(self):
        agent = make_agent()
        agent.card, agent.display = None, mock.Mock()
        agent.show_card("Orders — Tom", "No orders.")
        agent.show_card("Orders — Tom", blocks.card(notes=["Order 1 filled."]))
        self.assertEqual(agent.card["version"], 2)
        self.assertEqual(agent.card["notes"], ["Order 1 filled."])
        self.assertEqual(agent.display.open.call_count, 2)

    def test_the_trade_log_heading_is_always_schwab(self):
        heading = _DISPLAY["query_trade_log"](ToolContext(active_agent="linda"), {})
        self.assertEqual(heading, "Trades from the trade log — Linda, Charles Schwab")


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
