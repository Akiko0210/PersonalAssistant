"""The working record: a persona's own chart, written by its tools and shown
in full every turn (stores/working_state.py, tools/working_state_tools.py)."""

import json
import tempfile
import unittest
from pathlib import Path

from brain import agents
import config as cfg
from stores import working_state as ws
from tests.llm_fixtures import make_claude, system_text, text_reply, tool_reply
from tools import ToolContext, dispatch


class RecordBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = (cfg.WORKING_STATE_PATH, cfg.WORKING_RECORD_CHARS)
        cfg.WORKING_STATE_PATH = Path(self.tmp.name) / "working_state.db"

    def tearDown(self):
        cfg.WORKING_STATE_PATH, cfg.WORKING_RECORD_CHARS = self._saved
        self.tmp.cleanup()

    def call(self, name, who="tom", **args):
        return dispatch(ToolContext(active_agent=who), name, args)


class TestStore(RecordBase):
    def test_a_persona_cannot_touch_another_personas_record(self):
        rid = ws.create("tom", "SPX Campaign Diagonal", kind="campaign")
        for write in (lambda: ws.update("linda", rid, body="x"),
                      lambda: ws.add_entry("linda", rid, "x"),
                      lambda: ws.close("linda", rid, "x")):
            with self.assertRaises(ValueError):
                write()
        self.assertIsNone(ws.render("linda", rid))
        self.assertEqual(ws.block("linda"), "")
        self.assertIn("SPX Campaign Diagonal", ws.block("tom"))

    def test_an_over_long_body_is_refused_not_truncated(self):
        rid = ws.create("tom", "Campaign", body="rules")
        with self.assertRaises(ValueError) as e:
            ws.update("tom", rid, body="x" * (cfg.WORKING_BODY_CHARS + 1))
        self.assertIn("Condense it", str(e.exception))
        self.assertIn("rules", ws.render("tom", rid))  # the old body stands
        with self.assertRaises(ValueError):
            ws.add_entry("tom", rid, "y" * (cfg.WORKING_ENTRY_CHARS + 1))

    def test_closing_needs_an_outcome_and_hides_the_record(self):
        rid = ws.create("alice", "Call Elizabeth about the CD", kind="item")
        with self.assertRaises(ValueError):
            ws.close("alice", rid, "  ")
        ws.close("alice", rid, "done — she renewed it")
        self.assertEqual(ws.block("alice"), "")
        self.assertIn("Closed: done — she renewed it", ws.render("alice", rid))
        with self.assertRaises(ValueError):
            ws.add_entry("alice", rid, "late note")
        self.assertIn("(closed)", "\n".join(ws.listing("alice")))

    def test_every_write_logs_the_line_the_audit_counts(self):
        with self.assertLogs("records", "INFO") as logs:
            rid = ws.create("linda", "In-trade optimization urge", kind="pattern")
            ws.add_entry("linda", rid, "Adjusted a working calendar twice before lunch.")
        self.assertIn(f"record linda #{rid} created", logs.output[0])
        self.assertIn(f"record linda #{rid} entry", logs.output[1])


class TestPacking(RecordBase):
    def test_all_entries_shown_when_they_fit(self):
        rid = ws.create("tom", "SPX Campaign Diagonal", body="Long put 20-30 above spot.")
        ws.add_entry("tom", rid, "Entered at 108.")
        ws.add_entry("tom", rid, "Rolled the short to Friday.")
        block = ws.block("tom")
        self.assertTrue(block.startswith("Your working record"))
        self.assertLess(block.index("Long put"), block.index("Entered at 108"))
        self.assertLess(block.index("Entered at 108"), block.index("Rolled"))
        self.assertNotIn("not shown", block)

    def test_over_budget_keeps_headers_and_each_records_newest(self):
        busy = ws.create("tom", "Busy campaign", body="busy rules")
        quiet = ws.create("tom", "Quiet campaign", body="quiet rules")
        ws.add_entry("tom", quiet, "quiet entry one")
        for i in range(40):
            ws.add_entry("tom", busy, f"busy entry {i:02d} " + "x" * 60)
        cfg.WORKING_RECORD_CHARS = 1500
        with self.assertLogs("records", "INFO") as logs:
            block = ws.block("tom")
        self.assertLessEqual(len(block), 1500 + 200)  # the footer may overhang
        for kept in ("busy rules", "quiet rules", "quiet entry one",
                     "busy entry 39", "busy entry 38"):
            self.assertIn(kept, block)
        self.assertNotIn("busy entry 00", block)
        self.assertIn("older entries not shown", block)
        self.assertIn("truncated", logs.output[-1])
        self.assertIn("busy entry 00", ws.render("tom", busy))  # still stored

    def test_records_past_the_budget_still_show_their_titles(self):
        for i in range(4):
            ws.create("tom", f"Campaign {i}", body="r" * 1900)
        cfg.WORKING_RECORD_CHARS = 4500
        block = ws.block("tom")
        for i in range(4):
            self.assertIn(f"Campaign {i}", block)
        self.assertIn("(details: read_record", block)

    def test_the_block_is_stable_between_writes(self):
        # Its bytes are prompt-cached: rendering must not change unless a
        # record does.
        rid = ws.create("tom", "Campaign", body="rules")
        ws.add_entry("tom", rid, "entry")
        self.assertEqual(ws.block("tom"), ws.block("tom"))


class TestTools(RecordBase):
    def test_create_reads_back_and_update_replaces_the_body(self):
        out = self.call("update_record", kind="campaign", title="SPX Campaign Diagonal",
                        body="Long put 20-30 points above spot, 30-40 DTE.")
        self.assertIn("It now reads:", out)
        self.assertIn("20-30 points above spot", out)
        rid = int(out.split("#")[1].split()[0])
        out = self.call("update_record", id=rid, body="Long put 40 above spot.")
        self.assertIn("40 above", out)
        self.assertNotIn("20-30", out)

    def test_refusals_come_back_as_sentences(self):
        self.assertIn("needs a title", self.call("update_record", body="x"))
        self.assertIn("no record #99", self.call("add_entry", id=99, text="x"))
        self.assertIn("Which record", self.call("close_record", outcome="x"))

    def test_entries_carry_the_writers_live_model(self):
        rid = ws.create("tom", "Campaign")
        dispatch(ToolContext(active_agent="tom", convo_model="claude-opus-5-5"),
                 "add_entry", {"id": rid, "text": "decided to roll"})
        with ws._db() as db:
            model = db.execute("SELECT model FROM entries").fetchone()[0]
        self.assertEqual(model, "claude-opus-5-5")

    def test_read_record_lists_open_then_recently_closed(self):
        ws.create("alice", "Open item", kind="item")
        done = ws.create("alice", "Done item", kind="item")
        ws.close("alice", done, "done")
        out = self.call("read_record", who="alice")
        self.assertLess(out.index("Open item"), out.index("Done item (closed)"))
        self.assertEqual(self.call("read_record", who="bob"),
                         "Your working record is empty.")


class TestInThePrompt(RecordBase):
    def test_the_record_sits_between_the_static_prompt_and_the_background(self):
        c = make_claude(active="tom")
        c._record_block = lambda owner: f"RECORD OF {owner}"
        c._pull_context = lambda text, wire_from: type("P", (), {"block": "BACKGROUND"})()
        c.converse("what am I running?")
        system = c.client.messages.calls[0]["system"]
        self.assertEqual([b["text"] for b in system[1:]], ["RECORD OF tom", "BACKGROUND"])
        self.assertIn("cache_control", system[1])
        self.assertNotIn("cache_control", system[2])

    def test_a_delegate_sees_its_own_record(self):
        c = make_claude([text_reply("done")], active="tom")
        c._record_block = lambda owner: f"RECORD OF {owner}"
        c.run_delegated_task("alice", "is 'call Elizabeth' on the list?")
        self.assertIn("RECORD OF alice", system_text(c.client.messages.calls[0]))

    def test_record_tools_stay_out_of_the_folder_dialogue_and_off_bob(self):
        c = make_claude([tool_reply("choose_folder", {"folder": "general"})])
        c.choose_folder_via_dialogue("t", "s", "general", lambda text: "")
        names = {t["name"] for t in c.client.messages.calls[0]["tools"]}
        self.assertFalse(names & agents.RECORD_TOOLS)
        self.assertFalse(agents.RECORD_TOOLS & agents.AGENTS["bob"]["tools"])
        for key in ("tom", "linda", "alice"):
            self.assertLessEqual(agents.RECORD_TOOLS, agents.AGENTS[key]["tools"])

    def test_record_rules_survive_a_dashboard_persona(self):
        overlay = Path(self.tmp.name) / "agents.json"
        overlay.write_text(json.dumps({"linda": {"persona": "Edited on the dashboard."}}))
        saved, cfg.AGENTS_PATH = cfg.AGENTS_PATH, overlay
        try:
            agents.load_agents()
            text = agents.instructions("linda")
            self.assertTrue(text.startswith("Edited on the dashboard."))
            self.assertIn("Coaches Notebook", text)
            self.assertIn("update_record", text)
        finally:
            cfg.AGENTS_PATH = Path(self.tmp.name) / "none.json"
            agents.load_agents()  # back to the coded defaults
            cfg.AGENTS_PATH = saved
        self.assertNotIn("update_record", agents.instructions("bob"))


if __name__ == "__main__":
    unittest.main()
