"""Background delegation, end to end minus hardware and network.

Covers the isolated mini-loop (Claude.run_delegated_task), the deferred pickup
(take_pending_delegations), the spoken interjection delivery
(Agent._deliver_interjections / _speak_interjection), and the deferred-slot
draining in Agent._after_reply — including the two regressions from
session_2026-07-31.log: the forwarded-switch note leak, and the silently
dropped switch.

Fake clients and agent shells throughout, in the house pattern of
test_hats.py and test_switch_announcement.py."""

import functools
import queue
import threading
import unittest
from types import SimpleNamespace

import agents
import config as cfg
from voice_agent import Agent
from tests.llm_fixtures import (FakeBlock, make_claude as _make_claude,
                                text_reply, tool_reply)

# The shell run_delegated_task touches, active as Tom.
make_claude = functools.partial(_make_claude, active="tom",
                                convo_model=cfg.CONVO_MODELS["sonnet"])


class TestRunDelegatedTask(unittest.TestCase):
    def test_note_task_returns_note_and_spoken_text(self):
        c = make_claude([
            tool_reply("save_conversation_note",
                       {"title": "T", "content": "body",
                        "spoken_summary": "s"}),
            text_reply("Saved and ready."),
        ])
        text, note = c.run_delegated_task("bob", "Save this note: body")
        self.assertEqual(text, "Saved and ready.")
        self.assertEqual(note["title"], "T")
        # The main conversation's context stays untouched — the note travels
        # via the return value, never via the shared pending slot.
        self.assertIsNone(c._ctx.pending_note)

    def test_runs_on_registry_default_not_the_active_override(self):
        # The user parking the foreground hat on an expensive model must not
        # silently price up background work.
        c = make_claude([text_reply("done")])
        c._ctx.convo_model = cfg.CONVO_MODELS["opus"]
        c.run_delegated_task("bob", "task")
        self.assertEqual(c.client.messages.calls[0]["model"],
                         cfg.CONVO_MODELS["haiku"])

    def test_background_loop_carries_no_switching_tools(self):
        c = make_claude([text_reply("done")])
        c.run_delegated_task("bob", "task")
        names = {t["name"] for t in c.client.messages.calls[0]["tools"]}
        self.assertIn("save_conversation_note", names)
        self.assertNotIn("switch_agent", names)
        self.assertNotIn("ask_agent", names)
        self.assertNotIn("set_conversation_model", names)

    def test_background_loop_cannot_mutate_orders(self):
        # Review-before-submit means the user HEARS the review; a background
        # worker is out of earshot, so it must never submit or cancel — while
        # read-only trading lookups stay delegable.
        c = make_claude([text_reply("done")])
        c.run_delegated_task("tom", "check my positions")
        names = {t["name"] for t in c.client.messages.calls[0]["tools"]}
        self.assertIn("get_positions", names)
        self.assertIn("get_pnl", names)
        self.assertNotIn("submit_order", names)
        self.assertNotIn("cancel_order", names)

    def test_truncated_delegated_reply_is_reported_honestly(self):
        # The delegated loop used to lack converse()'s max_tokens honesty —
        # a truncated tool call died silently. The shared _tool_loop fixed
        # that; this pins it.
        resp = SimpleNamespace(
            stop_reason="max_tokens",
            content=[FakeBlock(type="tool_use", id="t1",
                               name="save_conversation_note",
                               input={"title": "T"})])
        text, _ = make_claude([resp]).run_delegated_task("bob", "save this")
        self.assertIn("did not complete", text)

    def test_round_cap_reports_honestly(self):
        c = make_claude([tool_reply("get_current_time", {}, f"tu_{i}")
                         for i in range(cfg.DELEGATION_MAX_TOOL_ROUNDS)])
        text, _ = c.run_delegated_task("bob", "task")
        self.assertIn("ran out", text)

    def test_take_pending_delegations_roundtrip(self):
        c = make_claude([])
        c._ctx.pending_delegations = [("bob", "t1"), ("alice", "t2")]
        self.assertEqual(c.take_pending_delegations(),
                         [("bob", "t1"), ("alice", "t2")])
        self.assertEqual(c.take_pending_delegations(), [])


class _Shell:
    """The Agent surface the interjection/_after_reply paths touch."""

    def __init__(self, active="tom"):
        self.spoken = []      # (text, kwargs) from say()
        self.voices = []      # set_voice calls
        self.events = []      # record_tool_event texts
        self.saved = []       # (_save_pending_note args)
        self.delegated = []   # _start_delegation args
        self.switched = []    # _switch_agent args
        self.llm = SimpleNamespace(
            active=active,
            record_tool_event=self.events.append,
            flush_tool_events=lambda persist=False: None,
            take_pending_delegations=lambda: [],
            take_pending_note=lambda: None,
            take_pending_switch=lambda: None,
        )
        self.audio = SimpleNamespace(
            muted=threading.Event(),
            has_buffered_speech=lambda: False,
            flush=lambda: None,
        )
        self.tts = SimpleNamespace(
            set_voice=lambda *a: self.voices.append(a),
            current_voice=lambda: "voice",
        )
        self._speaking_voice = None
        self.log = SimpleNamespace(info=lambda *a: None,
                                   warning=lambda *a: None,
                                   exception=lambda *a: None)
        self.interject = threading.Event()
        self.interjections = queue.Queue()

    def say(self, text, **kw):
        self.spoken.append((text, kw))
        return False

    def _save_pending_note(self, pending, saver=None):
        self.saved.append((pending, saver))

    def _start_delegation(self, key, task):
        self.delegated.append((key, task))

    def _switch_agent(self, key):
        self.switched.append(key)

    def _use_voice(self, hat):
        return Agent._use_voice(self, hat)

    def _speak_interjection(self, item):
        return Agent._speak_interjection(self, item)

    def _deliver_interjections(self):
        return Agent._deliver_interjections(self)


class TestInterjectionDelivery(unittest.TestCase):
    def test_text_result_is_read_by_the_active_persona_interruptibly(self):
        # A lookup result can be a whole note read aloud — the active persona
        # delivers it as a normal reply: its own voice, barge-in working,
        # "continue" available after one.
        shell = _Shell(active="tom")
        shell.interjections.put({"agent": "bob", "text": "found 3 notes.",
                                 "note": None})
        Agent._deliver_interjections(shell)
        text, kw = shell.spoken[0]
        self.assertIn("Bob", text)              # attribution stays audible
        self.assertIn("found 3 notes.", text)
        self.assertIsNot(kw.get("voice"), False)      # voice barge-in works
        self.assertIs(kw.get("save_resume"), True)    # so does "continue"
        self.assertEqual(shell.voices, [])  # no voice switch — Tom reads it
        # The shared conversation learns what happened.
        self.assertTrue(any("Bob" in e for e in shell.events))

    def test_text_result_reaches_history_before_it_is_spoken(self):
        # A barge-in one word in must not strand the result — by the time
        # anything is audible it is already folded into the shared history.
        shell = _Shell()
        order = []
        shell.llm.record_tool_event = lambda t: order.append("recorded")
        shell.say = lambda text, **kw: (order.append("spoke"), False)[1]
        shell.interjections.put({"agent": "bob", "text": "x", "note": None})
        Agent._deliver_interjections(shell)
        self.assertEqual(order, ["recorded", "spoke"])

    def test_interrupted_delivery_leaves_the_rest_queued(self):
        # The user's voice takes the floor: later results wait for the next
        # utterance boundary instead of talking over them.
        shell = _Shell()
        shell.say = lambda text, **kw: True  # user barges in immediately
        shell.interjections.put({"agent": "bob", "text": "one", "note": None})
        shell.interjections.put({"agent": "bob", "text": "two", "note": None})
        Agent._deliver_interjections(shell)
        self.assertEqual(shell.interjections.qsize(), 1)  # "two" still queued

    def test_note_result_still_speaks_as_the_worker_uninterruptibly(self):
        # The folder dialogue is a real exchange with the worker persona —
        # that path keeps its voice switch and its barge-in proofing.
        shell = _Shell(active="tom")
        note = {"title": "T", "content": "b"}
        shell.interjections.put({"agent": "bob", "text": "ready", "note": note})
        Agent._deliver_interjections(shell)
        self.assertIs(shell.spoken[0][1].get("voice"), False)
        self.assertEqual(shell.voices[0][0], agents.AGENTS["bob"]["tts_voice"])
        self.assertEqual(shell.voices[-1][0], agents.AGENTS["tom"]["tts_voice"])

    def test_note_result_announces_then_runs_the_folder_flow(self):
        shell = _Shell()
        note = {"title": "T", "content": "b", "spoken": "", "category": None}
        shell.interjections.put({"agent": "bob", "text": "ready", "note": note})
        Agent._deliver_interjections(shell)
        self.assertIn("ready to file", shell.spoken[0][0])
        self.assertEqual(shell.saved, [(note, "Bob")])

    def test_deferred_while_muted(self):
        shell = _Shell()
        shell.audio.muted.set()
        shell.interjections.put({"agent": "bob", "text": "x", "note": None})
        Agent._deliver_interjections(shell)
        self.assertEqual(shell.spoken, [])
        self.assertFalse(shell.interjections.empty())  # still queued

    def test_deferred_while_user_speech_is_buffered(self):
        shell = _Shell()
        shell.audio.has_buffered_speech = lambda: True
        shell.interjections.put({"agent": "bob", "text": "x", "note": None})
        Agent._deliver_interjections(shell)
        self.assertEqual(shell.spoken, [])
        self.assertFalse(shell.interjections.empty())

    def test_wake_event_cleared_by_delivery(self):
        shell = _Shell()
        shell.interject.set()
        Agent._deliver_interjections(shell)
        self.assertFalse(shell.interject.is_set())

    def test_voice_restored_even_when_the_save_flow_raises(self):
        shell = _Shell(active="alice")
        def boom(pending, saver=None):
            raise RuntimeError("folder dialogue exploded")
        shell._save_pending_note = boom
        shell.interjections.put({"agent": "bob", "text": "r",
                                 "note": {"title": "T", "content": "b"}})
        with self.assertRaises(RuntimeError):
            Agent._deliver_interjections(shell)
        self.assertEqual(shell.voices[-1][0],
                         agents.AGENTS["alice"]["tts_voice"])


class TestAfterReply(unittest.TestCase):
    def test_delegations_start_after_the_reply(self):
        shell = _Shell()
        shell.llm.take_pending_delegations = lambda: [("bob", "save it")]
        Agent._after_reply(shell, interrupted=False)
        self.assertEqual(shell.delegated, [("bob", "save it")])

    def test_forwarded_switch_note_is_drained_in_the_same_turn(self):
        # The 2026-08-01 leak: switch_agent(forward="save...") had Bob prepare
        # the note INSIDE the switch handling, after the pending-note check —
        # so its folder question fired a turn later and ate the user's
        # "switch me back to Tom".
        shell = _Shell()
        note = {"title": "T", "content": "b"}
        notes = [None, note]  # pre-switch check empty; post-forward check full
        shell.llm.take_pending_note = lambda: notes.pop(0)
        switches = [("bob", "save this as a note"), None]
        shell.llm.take_pending_switch = lambda: switches.pop(0)
        shell.llm.converse = lambda text: "Done."
        Agent._after_reply(shell, interrupted=False)
        self.assertEqual(shell.switched, ["bob"])
        self.assertEqual(shell.saved, [(note, "Bob")])  # drained, not leaked

    def test_dropped_switch_is_announced_aloud(self):
        # The other half of the incident: the drop itself was log-only, so the
        # user spent 16 minutes talking to the wrong persona.
        shell = _Shell()
        shell.llm.take_pending_note = lambda: {"title": "T", "content": "b"}
        shell.llm.take_pending_switch = lambda: ("tom", "")
        Agent._after_reply(shell, interrupted=False)
        said = " ".join(t for t, _ in shell.spoken)
        self.assertIn("didn't switch you to Tom", said)

    def test_chained_switch_from_a_forwarded_turn_is_dropped(self):
        shell = _Shell()
        switches = [("bob", "do the thing"), ("tom", "")]
        shell.llm.take_pending_switch = lambda: switches.pop(0)
        shell.llm.converse = lambda text: "Handled."
        Agent._after_reply(shell, interrupted=False)
        self.assertEqual(shell.switched, ["bob"])  # one hand-off per turn


if __name__ == "__main__":
    unittest.main()
