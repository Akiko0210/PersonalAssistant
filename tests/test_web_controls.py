"""The dashboard's live controls: direct dispatch onto the Agent, the typed-
message turn wiring, one real HTTP round trip, and the embedded server's
soft-fail promise (a web page must never stop the voice agent from starting).
"""

import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from web import server as dashboard
from tests.agent_fixtures import make_agent


def make_control_agent():
    """The Agent surface the control routes touch, over real primitives."""
    agent = make_agent()

    def set_mute(muted):
        (agent.audio.muted.set if muted else agent.audio.muted.clear)()

    agent.set_mute = set_mute
    agent.sent = []
    agent.queue_typed_message = (
        lambda text, target=None: agent.sent.append((text, target)))
    return agent


class DispatchTests(unittest.TestCase):
    def test_mute_applies_and_returns_the_resulting_state(self):
        agent = make_control_agent()
        status, body = dashboard.api_control_post(agent, "mute", {"muted": True})
        self.assertEqual(status, 200)
        self.assertTrue(agent.audio.muted.is_set())
        self.assertEqual(body, {"ok": True, "muted": True,
                                "mode": "conversation_mode"})

    def test_mute_rejects_non_boolean_payloads(self):
        agent = make_control_agent()
        for payload in ({}, {"muted": "yes"}, {"muted": 1}, [], None):
            status, body = dashboard.api_control_post(agent, "mute", payload)
            self.assertEqual(status, 400, payload)
            self.assertFalse(agent.audio.muted.is_set())

    def test_send_message_strips_validates_and_queues(self):
        agent = make_control_agent()
        status, body = dashboard.api_control_post(
            agent, "send_message", {"text": "  hello  "})
        self.assertEqual((status, body["queued"]), (200, True))
        self.assertEqual(agent.sent, [("hello", None)])
        for payload in ({}, {"text": "  "}, {"text": 3}, {"text": "x" * 5000}):
            status, _ = dashboard.api_control_post(agent, "send_message", payload)
            self.assertEqual(status, 400, payload)
        self.assertEqual(agent.sent, [("hello", None)])

    def test_send_message_carries_the_viewed_persona(self):
        # The regression this guards: a message typed into Bob's thread was
        # answered by the active persona because the target never left the page.
        agent = make_control_agent()
        status, _ = dashboard.api_control_post(
            agent, "send_message", {"text": "hi", "agent": "bob"})
        self.assertEqual(status, 200)
        self.assertEqual(agent.sent, [("hi", "bob")])
        status, _ = dashboard.api_control_post(
            agent, "send_message", {"text": "hi", "agent": "nobody"})
        self.assertEqual(status, 400)
        self.assertEqual(agent.sent, [("hi", "bob")])

    def test_unknown_action_is_404(self):
        status, _ = dashboard.api_control_post(make_control_agent(), "reboot", {})
        self.assertEqual(status, 404)

    def test_standalone_answers_honestly_both_ways(self):
        with mock.patch.object(dashboard, "agent_running", return_value=False):
            status, body = dashboard.api_control_post(None, "mute", {"muted": True})
        self.assertEqual(status, 409)
        self.assertIn("isn't running", body["error"])
        with mock.patch.object(dashboard, "agent_running", return_value=True):
            status, body = dashboard.api_control_post(None, "mute", {"muted": True})
        self.assertEqual(status, 409)
        self.assertIn("use its dashboard", body["error"])

    def test_get_state_shapes(self):
        agent = make_control_agent()
        agent.audio.muted.set()
        self.assertEqual(dashboard.api_control(agent),
                         {"agent_running": True, "muted": True,
                          "mode": "conversation_mode"})
        with mock.patch.object(dashboard, "agent_running", return_value=False):
            self.assertEqual(dashboard.api_control(None),
                             {"agent_running": False, "muted": None,
                              "mode": None})


class TypedMessageTests(unittest.TestCase):
    def test_queueing_wakes_the_idle_listener(self):
        agent = make_agent()
        agent.queue_typed_message("what's my last note?")
        self.assertEqual(agent.typed.get_nowait(),
                         ("what's my last note?", None))
        self.assertTrue(agent.interject.is_set())

    def test_a_typed_message_is_answered_and_spoken(self):
        agent = make_agent()
        agent._after_reply = lambda interrupted: None
        agent._handle_typed_message("what's my last note?")
        self.assertEqual(agent.llm.calls, ["what's my last note?"])
        self.assertEqual(agent.spoken, ["reply::what's my last note?"])

    def test_a_typed_message_is_drained_by_the_turn_loop(self):
        agent = make_agent()  # no utterances: collect returns None (idle wake)
        agent._after_reply = lambda interrupted: None
        agent.queue_typed_message("hello there")
        agent.run_conversation_turn()
        self.assertEqual(agent.llm.calls, ["hello there"])

    def test_persona_addressing_works_typed(self):
        agent = make_agent()
        agent._after_reply = lambda interrupted: None
        switched = []
        agent._switch_agent = switched.append
        with mock.patch("voice_agent.agents.match_address",
                        return_value=("tom", "do the thing")):
            agent._handle_typed_message("Tom, do the thing")
        self.assertEqual(switched, ["tom"])
        self.assertEqual(agent.llm.calls, ["do the thing"])

    def test_a_targeted_message_switches_persona_first(self):
        agent = make_agent()  # FakeLLM.active == "alice"
        agent._after_reply = lambda interrupted: None
        switched = []
        agent._switch_agent = switched.append
        agent._handle_typed_message("what time is it", "bob")
        self.assertEqual(switched, ["bob"])
        self.assertEqual(agent.llm.calls, ["what time is it"])
        agent._handle_typed_message("hi again", "alice")  # already active
        self.assertEqual(switched, ["bob"])

    def test_an_empty_reply_is_reported_not_silent(self):
        agent = make_agent()
        agent.llm.converse = lambda text: ""
        agent._handle_typed_message("hello?")
        self.assertEqual(len(agent.spoken), 1)
        self.assertIn("empty reply", agent.spoken[0])


class HttpRoundTripTests(unittest.TestCase):
    """One pass over the wire, against the real Handler + routing."""

    def setUp(self):
        self.agent = make_control_agent()
        self.server = dashboard.build_server(0, agent=self.agent)
        threading.Thread(target=lambda: self.server.serve_forever(0.05),
                         daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def post(self, path, payload):
        req = urllib.request.Request(self.base + path,
                                     data=json.dumps(payload).encode())
        try:
            with urllib.request.urlopen(req, timeout=2) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_the_full_round_trip(self):
        with urllib.request.urlopen(self.base + "/api/control", timeout=2) as r:
            self.assertEqual(json.loads(r.read())["muted"], False)
        status, body = self.post("/api/control/mute", {"muted": True})
        self.assertEqual((status, body["muted"]), (200, True))
        self.assertTrue(self.agent.audio.muted.is_set())
        status, body = self.post("/api/control/send_message", {"text": "hi"})
        self.assertEqual((status, body["queued"]), (200, True))
        self.assertEqual(self.agent.sent, [("hi", None)])
        self.assertEqual(self.post("/api/control/reboot", {})[0], 404)
        self.assertEqual(self.post("/api/control/mute", {"muted": "x"})[0], 400)


class EmbeddedTests(unittest.TestCase):
    def tearDown(self):
        dashboard._embedded_agent = None

    def test_serve_embedded_marks_this_process_as_the_agent(self):
        handle = dashboard.serve_embedded(make_control_agent(), port=0)
        self.addCleanup(handle.stop)
        self.assertIsNotNone(handle.port)
        self.assertTrue(dashboard.agent_running())  # no lock probe needed

    def test_a_taken_port_fails_soft(self):
        first = dashboard.serve_embedded(make_control_agent(), port=0)
        self.addCleanup(first.stop)
        with self.assertLogs("dashboard", level="WARNING"):
            second = dashboard.serve_embedded(make_control_agent(),
                                              port=first.port)
        self.assertIsNone(second.port)  # dead handle, and start didn't raise
        second.stop()                   # safe on a never-bound server


if __name__ == "__main__":
    unittest.main()
