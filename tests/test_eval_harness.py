"""The test environment's own guarantees (#66): a run can't reach production,
the network or the project's API credits, and a snapshot can't drift.
These are the ISO checks from the test plan, runnable without any model."""

import json
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from evals import gateway, sandbox, snapshot


def _junction(link, target):
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       check=True, capture_output=True)
    else:
        os.symlink(target, link)


class IsolationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run = self.tmp / "run"
        self.prod = self.tmp / "prod" / "data"
        self.run.mkdir()
        self.prod.mkdir(parents=True)

    def test_junction_into_production_is_refused_before_import(self):  # ISO-01
        _junction(self.run / "data", self.prod)
        with self.assertRaisesRegex(sandbox.SandboxError, "outside the run folder"):
            sandbox.check_layout(self.run, [str(self.prod)])

    def test_run_folder_inside_production_is_refused(self):
        with self.assertRaisesRegex(sandbox.SandboxError, "overlaps production"):
            sandbox.check_layout(self.prod / "run", [str(self.prod)])

    def test_configured_path_outside_or_into_production_is_refused(self):
        mod = types.ModuleType("fakecfg")
        mod.WORKING_STATE_PATH = self.run / "data" / "w.db"
        self.assertIn("fakecfg.WORKING_STATE_PATH", sandbox.check_paths(self.run, [mod]))
        mod.TRADE_LOG_PATH = self.tmp / "elsewhere.db"
        with self.assertRaisesRegex(sandbox.SandboxError, "outside the run folder"):
            sandbox.check_paths(self.run, [mod])
        mod.TRADE_LOG_PATH = self.run / "x.db"
        mod.DISCORD_DIR = self.prod / "discord"      # allowed outside, never into production
        with self.assertRaisesRegex(sandbox.SandboxError, "into production"):
            sandbox.check_paths(self.run, [mod], [str(self.prod)])

    def test_every_non_local_tool_is_fixtured_or_fails(self):  # ISO-02, ISO-05
        registry = {"read_record": ({"name": "read_record"}, lambda c, a: "real"),
                    "get_positions": ({"name": "get_positions"}, lambda c, a: "LIVE"),
                    "get_quote": ({"name": "get_quote"}, lambda c, a: "LIVE"),
                    "submit_order": ({"name": "submit_order"}, lambda c, a: "LIVE")}
        calls = []
        sandbox.install_fixtures(registry, {"get_positions": "fixture rows"}, calls)
        self.assertEqual(registry["read_record"][1](None, {}), "real")
        self.assertEqual(registry["get_positions"][1](None, {}), "fixture rows")
        with self.assertRaisesRegex(RuntimeError, "no fixture for get_quote"):
            registry["get_quote"][1](None, {})
        self.assertIn("rejected", registry["submit_order"][1](None, {"qty": 1}))
        self.assertEqual([c["tool"] for c in calls if c["mutating"]], ["submit_order"])

    def test_only_loopback_sockets_open(self):  # ISO-03, in a child so the guard can't leak
        code = ("import socket; from evals import sandbox; sandbox.guard_network()\n"
                "try:\n    socket.create_connection(('93.184.216.34', 443), timeout=2)\n"
                "    print('OPEN')\nexcept PermissionError as e:\n    print('BLOCKED')\n"
                "try:\n    socket.getaddrinfo('api.schwabapi.com', 443)\n    print('DNS')\n"
                "except PermissionError:\n    print('NODNS')")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent.parent).stdout.split()
        self.assertEqual(out, ["BLOCKED", "NODNS"])

    def test_secrets_leave_the_environment(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k", "SCHWAB_APP_SECRET": "s",
                                          "PATH": os.environ.get("PATH", "")}):
            sandbox.scrub_env(self.run)
            self.assertNotIn("ANTHROPIC_API_KEY", os.environ)
            self.assertNotIn("SCHWAB_APP_SECRET", os.environ)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")


class GatewayTest(unittest.TestCase):
    def test_child_never_sees_the_project_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-project"}):
            self.assertNotIn("ANTHROPIC_API_KEY", gateway.child_env())

    def test_refuses_anything_but_the_subscription(self):
        fake = mock.Mock(stdout=json.dumps({"loggedIn": True, "authMethod": "apiKey"}))
        with mock.patch.object(gateway.subprocess, "run", return_value=fake):
            with self.assertRaisesRegex(gateway.GatewayError, "subscription"):
                gateway.require_subscription("claude")

    def test_tail_renders_as_a_labelled_transcript(self):
        text = gateway.render_prompt([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "recall",
                                               "input": {"q": "x"}}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "found"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            {"role": "user", "content": "and now?"}])
        self.assertTrue(text.endswith("and now?"))
        self.assertIn("[tool call recall", text)
        self.assertIn("You (assistant): Done.", text)

    def test_mcp_proxy_round_trip(self):
        seen = []
        bridge = gateway.ToolBridge(lambda n, a: seen.append((n, a)) or f"ran {n}")
        try:
            tools = Path(tempfile.mkdtemp()) / "tools.json"
            tools.write_text(json.dumps([{"name": "read_record", "description": "d",
                                          "input_schema": {"type": "object"}}]))
            msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                    {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                     "params": {"name": "read_record", "arguments": {"id": 26}}}]
            out = subprocess.run(
                [sys.executable, str(Path(gateway.__file__).with_name("mcp_proxy.py")),
                 str(tools), bridge.url],
                input="\n".join(json.dumps(m) for m in msgs) + "\n",
                capture_output=True, text=True, timeout=60).stdout.splitlines()
        finally:
            bridge.close()
        replies = [json.loads(line) for line in out]
        self.assertEqual([r["id"] for r in replies], [1, 2, 3])
        self.assertEqual(replies[1]["result"]["tools"][0]["name"], "read_record")
        self.assertEqual(replies[2]["result"]["content"][0]["text"], "ran read_record")
        self.assertEqual(seen, [("read_record", {"id": 26})])


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        src = self.tmp / "src"
        (src / "data/memory").mkdir(parents=True)
        (src / "data/trading/schwab").mkdir(parents=True)
        db = sqlite3.connect(src / "data/memory/working_state.db")
        db.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, owner, status)")
        db.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY, record_id)")
        db.execute("INSERT INTO records VALUES (1, 'tom', 'open')")
        db.commit()
        db.close()
        (src / "data/trading/schwab/tokens.json").write_text("{}")
        (src / "data/trading/schwab/accounts.json").write_text("{}")
        self.master = self.tmp / "snap"
        self.manifest = snapshot.capture(src, self.master)

    def tearDown(self):
        for p in self.tmp.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)

    def test_capture_excludes_credentials_and_is_read_only(self):
        self.assertIn("data/memory/working_state.db", self.manifest["files"])
        self.assertIn("data/trading/schwab/accounts.json", self.manifest["files"])
        self.assertNotIn("data/trading/schwab/tokens.json", self.manifest["files"])
        self.assertEqual(self.manifest["counts"]["records"], {"tom/open": 1})
        self.assertFalse(os.access(self.master / "data/memory/working_state.db", os.W_OK))

    def test_clone_writes_never_reach_the_master(self):  # ISO-04, ISO-06
        run = self.tmp / "run"
        snapshot.clone(self.master, run)
        db = sqlite3.connect(run / "data/memory/working_state.db")
        db.execute("INSERT INTO records VALUES (2, 'tom', 'open')")
        db.commit()
        db.close()
        self.assertEqual(snapshot.verify(self.master), [])

    def test_verify_catches_a_changed_master(self):
        target = self.master / "data/trading/schwab/accounts.json"
        os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
        target.write_text('{"changed": 1}')
        self.assertEqual(snapshot.verify(self.master),
                         ["changed data/trading/schwab/accounts.json"])


if __name__ == "__main__":
    unittest.main()
