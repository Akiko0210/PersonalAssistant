"""Several accounts on one broker login (#11): the user names them, picks
one by voice, every answer says which account it read, "all" covers every
account with a combined line, and an order reviewed on one account can't be
placed on another. On 2026-10-15 Tom, facing the user's two Schwab
accounts, could answer nothing and told them to edit a config file.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.trading_fixtures import FakeEngine, spx_chain
from tools import ToolContext, dispatch
from trading import accounts as taccounts
from trading import config as tcfg
from trading import ticket as tticket
from trading.models import Leg
from trading.ticket import Ticket

IRA, RED = "12345699", "87654868"


class AccountsBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(tcfg, "TRADING_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        taccounts.set_name("schwab", "699", "IRA")
        taccounts.set_name("schwab", "868", "red")
        self.eng = FakeEngine("schwab", accounts={IRA: 100.0, RED: -30.0})

    def run_tool(self, tool, args=None, persona="tom"):
        with mock.patch("tools.trading_tools._engine_for",
                        side_effect=lambda b: (self.eng, None) if b == "schwab"
                        else (None, "not configured")):
            return dispatch(ToolContext(active_agent=persona), tool, args or {})


class TestStore(AccountsBase):
    def test_resolve_by_name_digits_and_ambiguity(self):
        nums = [IRA, RED]
        self.assertEqual(taccounts.resolve("schwab", "my IRA account", nums), (IRA, None))
        self.assertEqual(taccounts.resolve("schwab", "Red", nums), (RED, None))
        self.assertEqual(taccounts.resolve("schwab", "868", nums), (RED, None))
        self.assertEqual(taccounts.resolve("schwab", "5699", nums), (IRA, None))
        n, problem = taccounts.resolve("schwab", "blue", nums)
        self.assertIsNone(n)
        self.assertIn("IRA (…699)", problem)
        n, problem = taccounts.resolve("schwab", "8", [IRA, RED, "11111118"])
        self.assertIn("several", problem)

    def test_a_name_moves_rather_than_meaning_two_accounts(self):
        taccounts.set_name("schwab", "868", "IRA")
        self.assertEqual(taccounts.label("schwab", IRA), "account …699")
        self.assertEqual(taccounts.names("schwab", RED), ["red", "IRA"])

    def test_default_is_focus_then_only_account_then_env(self):
        self.assertIsNone(taccounts.default("schwab", "tom", [IRA, RED]))
        self.assertEqual(taccounts.default("schwab", "tom", [IRA]), IRA)
        self.assertEqual(taccounts.default("schwab", "tom", [IRA, RED], configured="868"), RED)
        taccounts.set_focus("schwab", "tom", IRA)
        self.assertEqual(taccounts.default("schwab", "tom", [IRA, RED], configured="868"), IRA)
        self.assertIsNone(taccounts.default("schwab", "linda", [IRA, RED]))  # per persona
        # A focused account that is no longer shared is not used.
        self.assertIsNone(taccounts.default("schwab", "tom", [RED, "1"]))


class TestTools(AccountsBase):
    def test_no_choice_asks_once_and_never_points_at_config(self):
        out = self.run_tool("get_balance")
        self.assertIn("Which Charles Schwab account", out)
        self.assertIn("IRA (…699)", out)
        self.assertNotIn(".env", out)
        self.assertEqual(self.eng.calls, [])

    def test_move_to_the_ira_then_every_answer_is_the_iras_and_says_so(self):
        out = self.run_tool("set_account", {"account": "my IRA"})
        self.assertIn("Now working on IRA (…699)", out)
        out = self.run_tool("get_balance")
        self.assertTrue(out.startswith("IRA (…699): net liquidating value 100.00"), out)
        self.run_tool("get_positions")
        self.assertEqual(self.eng.calls[-1], ("positions", IRA, None))
        # Persisted: a new ToolContext (a new session) still uses it.
        self.assertEqual(taccounts.focus("schwab", "tom"), IRA)

    def test_another_account_answers_without_moving_the_focus(self):
        self.run_tool("set_account", {"account": "IRA"})
        out = self.run_tool("get_pnl", {"period": "today", "account": "red"})
        self.assertIn("red (…868): from", out)
        self.assertEqual(taccounts.focus("schwab", "tom"), IRA)

    def test_all_accounts_per_account_then_a_combined_line(self):
        self.run_tool("set_account", {"account": "IRA"})
        out = self.run_tool("get_pnl", {"period": "today", "account": "all"})
        self.assertIn("IRA (…699): from", out)
        self.assertIn("red (…868): from", out)
        self.assertIn("All of them together, added here only: realized +70.00", out)
        self.assertEqual(taccounts.focus("schwab", "tom"), IRA)

    def test_naming_an_account_does_not_switch_to_it(self):
        self.run_tool("set_account", {"account": "IRA"})
        out = self.run_tool("set_account", {"account": "868", "nickname": "Blue"})
        self.assertIn("is Blue", out)
        self.assertEqual(taccounts.focus("schwab", "tom"), IRA)
        self.assertEqual(taccounts.resolve("schwab", "blue", [IRA, RED]), (RED, None))

    def test_status_lists_accounts_and_the_one_in_use(self):
        self.run_tool("set_account", {"account": "red"})
        out = self.run_tool("trading_status")
        self.assertIn("Accounts: IRA (…699), red (…868).", out)
        self.assertIn("Working on red (…868).", out)


class TestOrdersFollowTheAccount(AccountsBase):
    def setUp(self):
        super().setUp()
        self.chain = spx_chain()
        tk = Ticket(underlying="SPX", legs=[Leg(6350.0, "Put", "2026-07-24", "Short")])
        tk.limit_price, tk.price_effect = 2.0, "Credit"
        tticket.save(tk, "schwab")
        self.reviewed, self.placed = [], []
        test = self

        class Orders:
            def review(self, tk, chain):
                test.reviewed.append(tk.account)
                tk.review = {"fingerprint": tk.fingerprint(), "at": 1e12}
                tticket.save(tk, "schwab")
                return SimpleNamespace(ok=True, errors=[], warnings=[], total_fees=1.0,
                                       buying_power_change=None, new_buying_power=5.0)

            def submit(self, tk, chain, confirmed=False):
                test.placed.append(tk.account)
                return SimpleNamespace(id="9", status="Working"), None
        self.eng.orders = Orders()
        self.eng.chain = lambda u: self.chain
        self.eng.broker = SimpleNamespace(env_label=lambda: "live")

    def test_review_names_the_account_and_submit_goes_there(self):
        self.run_tool("set_account", {"account": "IRA"})
        out = self.run_tool("review_order")
        self.assertIn("Reviewed for IRA (…699)", out)
        out = self.run_tool("submit_order", {"confirmed": True})
        self.assertIn("submitted for IRA (…699)", out)
        self.assertEqual((self.reviewed, self.placed), ([IRA], [IRA]))

    def test_switching_after_review_refuses_the_submit(self):
        self.run_tool("set_account", {"account": "IRA"})
        self.run_tool("review_order")
        out = self.run_tool("set_account", {"account": "red"})
        self.assertIn("reviewed again on this account", out)
        out = self.run_tool("submit_order", {"confirmed": True})
        self.assertIn("Not placed", out)
        self.assertIn("reviewed for IRA (…699)", out)
        self.assertEqual(self.placed, [])

    def test_the_fingerprint_covers_the_account(self):
        tk = tticket.load("schwab")
        tk.account = IRA
        before = tk.fingerprint()
        tk.account = RED
        self.assertNotEqual(before, tk.fingerprint())


class TestTradeLogUsesTheSameNames(AccountsBase):
    def test_nickname_and_focus_become_the_masked_digits(self):
        from tools import trade_log_tools as tl
        known = [("*****699SCHW", 3, "2026-09-01", "2026-10-01"),
                 ("*****868SCHW", 5, "2026-09-01", "2026-10-01")]
        with mock.patch.object(tl.trade_log, "accounts", return_value=known):
            self.assertEqual(tl._account("red"), ("*****868SCHW", None))
            self.assertIn("several accounts", tl._account("", "tom")[1])
            taccounts.set_focus("schwab", "tom", IRA)
            self.assertEqual(tl._account("", "tom"), ("*****699SCHW", None))
            self.assertEqual(tl._account("all", "tom"), (None, None))


if __name__ == "__main__":
    unittest.main()
