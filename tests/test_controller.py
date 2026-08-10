"""The dashboard's live controls: service layer, HTTP dispatch, and the
agent-side wiring for mute and typed messages."""

import json
import logging
import queue
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

import config as cfg
import dashboard
from controller import ControlServer
from controller_service import ControllerService
from voice_agent import Agent


def make_service(state=None, sent=None):
    """A ControllerService over plain fakes: a mutable state dict and a list
    collecting sent messages."""
    state = state if state is not None else {"muted": False}
    sent = sent if sent is not None else []
    return ControllerService(
        read_state=lambda: (state["muted"], "conversation_mode"),
        set_mute=lambda m: state.__setitem__("muted", m),
        send_message=sent.append,
    ), state, sent


class ServiceTests(unittest.TestCase):
    def test_mute_applies_and_returns_the_resulting_state(self):
        service, state, _ = make_service()
        out = service.mute({"muted": True})
        self.assertTrue(state["muted"])
        self.assertEqual(out, {"ok": True, "muted": True,
                               "mode": "conversation_mode"})

    def test_mute_rejects_non_boolean_payloads(self):
        service, _, _ = make_service()
        for payload in ({}, {"muted": "yes"}, {"muted": 1}, [], None):
            with self.assertRaises(ValueError):
                service.mute(payload)

    def test_send_message_strips_and_queues(self):
        service, _, sent = make_service()
        out = service.send_message({"text": "  hello there  "})
        self.assertEqual(sent, ["hello there"])
        self.assertEqual(out, {"ok": True, "queued": True})

    def test_send_message_rejects_empty_and_oversized(self):
        service, _, sent = make_service()
        for payload in ({}, {"text": "   "}, {"text": 3}, {"text": "x" * 5000}):
            with self.assertRaises(ValueError):
                service.send_message(payload)
        self.assertEqual(sent, [])

    def test_status_reports_the_live_state(self):
        service, state, _ = make_service()
        state["muted"] = True
        self.assertEqual(service.status(),
                         {"muted": True, "mode": "conversation_mode"})


class FakeLLM:
    active = "alice"

    def __init__(self, reply="the answer"):
        self.reply = reply
        self.asked = []

    def converse(self, text):
        self.asked.append(text)
        return self.reply


def make_agent(reply="the answer"):
    agent = Agent.__new__(Agent)
    agent.log = logging.getLogger("test")
    agent.typed = queue.Queue()
    agent.interject = threading.Event()
    agent.silence = threading.Event()
    agent.llm = FakeLLM(reply)
    agent.spoken = []
    agent.say = lambda text, **kw: (agent.spoken.append(text), False)[1]
    agent._after_reply = lambda interrupted: None
    return agent


class TypedMessageTests(unittest.TestCase):
    def test_queueing_wakes_the_idle_listener(self):
        agent = make_agent()
        agent._queue_typed_message("what's my last note?")
        self.assertEqual(agent.typed.get_nowait(), "what's my last note?")
        self.assertTrue(agent.interject.is_set())

    def test_a_typed_message_is_answered_and_spoken(self):
        agent = make_agent()
        agent._handle_typed_message("what's my last note?")
        self.assertEqual(agent.llm.asked, ["what's my last note?"])
        self.assertEqual(agent.spoken, ["the answer"])

    def test_persona_addressing_works_typed(self):
        agent = make_agent()
        switched = []
        agent._switch_agent = switched.append
        with mock.patch("voice_agent.agents.match_address",
                        return_value=("tom", "do the thing")):
            agent._handle_typed_message("Tom, do the thing")
        self.assertEqual(switched, ["tom"])
        self.assertEqual(agent.llm.asked, ["do the thing"])

    def test_an_empty_reply_is_reported_not_silent(self):
        agent = make_agent(reply="")
        agent._handle_typed_message("hello?")
        self.assertEqual(len(agent.spoken), 1)
        self.assertIn("empty reply", agent.spoken[0])


class HttpTests(unittest.TestCase):
    """Real ControlServer on an ephemeral port."""

    def setUp(self):
        self.service, self.state, self.sent = make_service()
        self.server = ControlServer(self.service, port=0).start()
        self.addCleanup(self.server.stop)
        self.base = f"http://127.0.0.1:{self.server.port}"

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=2) as r:
            return r.status, json.loads(r.read())

    def post(self, path, payload):
        req = urllib.request.Request(self.base + path,
                                     data=json.dumps(payload).encode())
        try:
            with urllib.request.urlopen(req, timeout=2) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_status_round_trip(self):
        self.assertEqual(self.get("/status"),
                         (200, {"muted": False, "mode": "conversation_mode"}))

    def test_mute_round_trip(self):
        status, body = self.post("/mute", {"muted": True})
        self.assertEqual(status, 200)
        self.assertTrue(body["muted"])
        self.assertTrue(self.state["muted"])

    def test_send_message_round_trip(self):
        status, body = self.post("/send_message", {"text": "hi"})
        self.assertEqual((status, body["queued"]), (200, True))
        self.assertEqual(self.sent, ["hi"])

    def test_unknown_action_is_404(self):
        self.assertEqual(self.post("/reboot", {})[0], 404)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/nope", timeout=2)
        self.assertEqual(ctx.exception.code, 404)

    def test_bad_payload_is_400_with_the_reason(self):
        status, body = self.post("/mute", {"muted": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("muted", body["error"])

    def test_a_taken_port_fails_soft(self):
        with self.assertLogs("controller", level="WARNING"):
            second = ControlServer(self.service, port=self.server.port).start()
        self.assertIsNone(second.port)  # not serving, and start() didn't raise
        second.stop()  # and stop() on a never-bound server is safe


class DashboardProxyTests(unittest.TestCase):
    def setUp(self):
        self.service, self.state, self.sent = make_service()
        self.server = ControlServer(self.service, port=0).start()
        self.addCleanup(self.server.stop)
        patch = mock.patch.object(cfg, "CONTROL_PORT", self.server.port)
        patch.start()
        self.addCleanup(patch.stop)

    def test_get_proxies_the_live_state(self):
        self.assertEqual(dashboard.agent_control_get(),
                         {"agent_running": True, "muted": False,
                          "mode": "conversation_mode"})

    def test_post_proxies_and_passes_agent_errors_through(self):
        status, body = dashboard.agent_control_post("mute", {"muted": True})
        self.assertEqual((status, body["muted"]), (200, True))
        status, body = dashboard.agent_control_post("mute", {"muted": "x"})
        self.assertEqual(status, 400)

    def test_no_agent_reads_as_not_running(self):
        self.server.stop()
        self.assertFalse(dashboard.agent_control_get()["agent_running"])
        status, body = dashboard.agent_control_post("mute", {"muted": True})
        self.assertEqual(status, 409)
        self.assertIn("isn't running", body["error"])


if __name__ == "__main__":
    unittest.main()
