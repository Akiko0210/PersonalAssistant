"""Gmail tools: registry wiring, auth degradation, and the Gmail-API seams
(thread parsing, MIME building) — all HTTP faked, no network."""

import base64
import json
import unittest
from unittest.mock import patch

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
