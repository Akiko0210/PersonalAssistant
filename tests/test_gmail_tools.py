"""Gmail tools: registry wiring, auth degradation, and the Gmail-API seams
(thread parsing, MIME building) — all HTTP faked, no network."""

import base64
import json
import logging
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import config as cfg
from brain import agents
from tools import ToolContext, api_tools, dispatch
from tools import gmail_tools

GMAIL_TOOLS = {"search_email_threads", "get_email_thread",
               "create_email_draft", "send_email"}
AUTH_OK = ({"Authorization": "Bearer test-token"}, None)


def b64(text):
    return base64.urlsafe_b64encode(text.encode()).decode()


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeRequests:
    """Scripted stand-in for the requests module: responses keyed by
    (method, url suffix); every call is recorded for assertions."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def _find(self, method, url):
        for (m, suffix), resp in self.responses.items():
            if m == method and url.endswith(suffix):
                return resp
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self._find("GET", url)

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self._find("POST", url)


class TestWiring(unittest.TestCase):
    def test_tools_registered_and_allowlisted_for_alice_and_tom(self):
        for key in ("alice", "tom"):
            self.assertLessEqual(GMAIL_TOOLS, agents.AGENTS[key]["tools"], key)
            names = {t["name"]
                     for t in api_tools(include=agents.AGENTS[key]["tools"])}
            self.assertLessEqual(GMAIL_TOOLS, names, key)

    def test_send_description_demands_explicit_confirmation(self):
        # send_email is gated by instruction, not machinery — losing that
        # wording in the description IS losing the gate.
        schema = next(t for t in api_tools() if t["name"] == "send_email")
        self.assertIn("explicitly said to send", schema["description"])


class TestAuthDegradation(unittest.TestCase):
    def test_auth_problem_comes_back_as_sentence(self):
        with patch.object(gmail_tools, "_auth",
                          return_value=(None, "Gmail authorization failed: no token.")):
            for name in GMAIL_TOOLS:
                out = dispatch(ToolContext(), name,
                               {"query": "x", "thread_id": "t",
                                "to": "a@b.c", "subject": "s", "body": "b"})
                self.assertIn("Gmail authorization failed", out, name)


class TestStartupAuth(unittest.TestCase):
    """Startup runs the interactive auth (browser consent if needed) but a
    failure only costs Gmail: the agent still starts, and the tools name the
    restart-and-authenticate route until a token exists."""

    def test_startup_runs_interactive_auth_and_proceeds(self):
        import voice_agent
        with patch("lib.gmail_auth.get_credentials") as get_creds:
            ok = voice_agent.ensure_gmail_auth(logging.getLogger("test"))
        self.assertTrue(ok)
        get_creds.assert_called_once_with(interactive=True)

    def test_startup_continues_without_gmail_when_auth_fails(self):
        import voice_agent
        log = logging.getLogger("test")
        with patch("lib.gmail_auth.get_credentials",
                   side_effect=RuntimeError("no client secret")), \
             self.assertLogs(log, level="WARNING") as captured:
            ok = voice_agent.ensure_gmail_auth(log)  # no SystemExit
        self.assertFalse(ok)
        self.assertIn("no client secret", captured.output[0])

    def test_startup_stops_waiting_but_the_consent_stays_live(self):
        # The failure that made startup look hung: a human reading a consent
        # screen outlasts any sane wait. The agent must go on without Gmail
        # and still let a late approval land.
        import voice_agent
        log = logging.getLogger("test")
        approved = threading.Event()

        def slow_consent(interactive=False):
            approved.wait(10)  # stands in for the person at the browser

        with patch.object(cfg, "GMAIL_AUTH_STARTUP_WAIT_S", 0.05), \
             patch("lib.gmail_auth.get_credentials", side_effect=slow_consent), \
             self.assertLogs(log, level="WARNING") as captured:
            ok = voice_agent.ensure_gmail_auth(log)
        self.assertFalse(ok)  # the agent starts anyway
        self.assertIn("starting without it", captured.output[0])
        approved.set()  # the consent thread was never cancelled, so it lands

    def test_tools_name_the_restart_route_when_unauthenticated(self):
        # The mid-conversation path never opens a browser: with no token the
        # tool's spoken answer must send the user to restart the app.
        from lib import gmail_auth
        with patch.object(cfg, "GMAIL_TOKEN_PATH", Path("nonexistent-token.json")):
            out = dispatch(ToolContext(), "search_email_threads", {"query": "x"})
        self.assertIn(gmail_auth.NOT_AUTHENTICATED, out)
        self.assertIn("restarting the app", out)


class TestSearchThreads(unittest.TestCase):
    def test_search_parses_thread_metadata_and_skips_failed_details(self):
        fake = FakeRequests({
            ("GET", "/threads"): FakeResponse({"threads": [
                {"id": "t1", "snippet": "lunch?"},
                {"id": "t2", "snippet": "gone"},
            ]}),
            ("GET", "/threads/t1"): FakeResponse({"messages": [
                {"payload": {"headers": [
                    {"name": "Subject", "value": "Lunch"},
                    {"name": "From", "value": "dana@x.com"},
                    {"name": "Date", "value": "Mon"},
                ]}},
                {"payload": {"headers": []}},
            ]}),
            ("GET", "/threads/t2"): FakeResponse({}, status_code=404),
        })
        with patch.object(gmail_tools, "requests", fake), \
             patch.object(gmail_tools, "_auth", return_value=AUTH_OK):
            out = json.loads(dispatch(ToolContext(), "search_email_threads",
                                      {"query": "is:unread"}))
        self.assertEqual(out["count"], 1)
        t = out["threads"][0]
        self.assertEqual((t["thread_id"], t["subject"], t["from"], t["messages"]),
                         ("t1", "Lunch", "dana@x.com", 2))


class TestGetThread(unittest.TestCase):
    def test_bodies_decoded_from_nested_mime_parts(self):
        fake = FakeRequests({
            ("GET", "/threads/t1"): FakeResponse({"messages": [{
                "id": "m1",
                "payload": {
                    "mimeType": "multipart/alternative",
                    "headers": [{"name": "From", "value": "dana@x.com"},
                                {"name": "Subject", "value": "Hi"}],
                    "parts": [{"mimeType": "text/plain",
                               "body": {"data": b64("see you at noon")}}],
                },
            }]}),
        })
        with patch.object(gmail_tools, "requests", fake), \
             patch.object(gmail_tools, "_auth", return_value=AUTH_OK):
            out = json.loads(dispatch(ToolContext(), "get_email_thread",
                                      {"thread_id": "t1"}))
        m = out["messages"][0]
        self.assertEqual(m["body"], "see you at noon")
        self.assertEqual(m["from"], "dana@x.com")


class TestCreateDraft(unittest.TestCase):
    def test_draft_posts_mime_and_attaches_to_thread(self):
        fake = FakeRequests({("POST", "/drafts"): FakeResponse({"id": "d1"})})
        with patch.object(gmail_tools, "requests", fake), \
             patch.object(gmail_tools, "_auth", return_value=AUTH_OK):
            out = json.loads(dispatch(ToolContext(), "create_email_draft",
                                      {"to": "dana@x.com", "subject": "Re: Lunch",
                                       "body": "Noon works.", "thread_id": "t1"}))
        self.assertEqual(out["status"], "draft saved")
        self.assertEqual(out["draft_id"], "d1")
        message = fake.calls[0][2]["json"]["message"]
        self.assertEqual(message["threadId"], "t1")
        raw = base64.urlsafe_b64decode(message["raw"]).decode()
        self.assertIn("To: dana@x.com", raw)
        self.assertIn("Subject: Re: Lunch", raw)
        self.assertIn("Noon works.", raw)


class TestSendEmail(unittest.TestCase):
    def test_send_posts_mime_to_send_endpoint(self):
        fake = FakeRequests({("POST", "/messages/send"):
                             FakeResponse({"id": "m9", "threadId": "t1"})})
        with patch.object(gmail_tools, "requests", fake), \
             patch.object(gmail_tools, "_auth", return_value=AUTH_OK):
            out = json.loads(dispatch(ToolContext(), "send_email",
                                      {"to": "dana@x.com", "subject": "Re: Lunch",
                                       "body": "Noon works.", "thread_id": "t1"}))
        self.assertEqual((out["status"], out["message_id"]), ("sent", "m9"))
        # The send endpoint takes the raw message at the top level (no
        # "message" wrapper, unlike /drafts).
        sent = fake.calls[0][2]["json"]
        self.assertEqual(sent["threadId"], "t1")
        self.assertIn("To: dana@x.com",
                      base64.urlsafe_b64decode(sent["raw"]).decode())


if __name__ == "__main__":
    unittest.main()
