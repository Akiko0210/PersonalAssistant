"""The spoken persona-switch announcement names the model it's running on.

Drives Agent._switch_agent unbound against a shell holding only the attributes
it touches — a real Agent would boot audio, TTS, Whisper and the note stores.
"""

import unittest
from types import SimpleNamespace

import agents
import config as cfg
from llm import Claude
from tools import ToolContext
from voice_agent import Agent


def make_llm():
    """A Claude with only what switch_to / active_model need."""
    c = Claude.__new__(Claude)
    c.active = agents.DEFAULT_AGENT
    c._model_overrides = {}
    c._ctx = ToolContext(active_agent=c.active,
                         convo_model=cfg.CONVO_MODELS["haiku"])
    c.history = []  # switch_to appends its took-over marker here
    c._save_history = lambda: None       # no disk writes from tests
    c._write_agent_state = lambda: None
    return c


class _AgentShell:
    """Just the surface _switch_agent uses."""

    def __init__(self):
        self.llm = make_llm()
        self.tts = SimpleNamespace(set_voice=lambda *a: None,
                                   current_voice=lambda: "voice")
        self._speaking_voice = None
        self.log = SimpleNamespace(info=lambda *a: None)
        self.spoken = []

    def say(self, text, **kw):
        self.spoken.append(text)


class TestSwitchAnnouncement(unittest.TestCase):
    def test_announcement_names_persona_and_model(self):
        shell = _AgentShell()
        Agent._switch_agent(shell, "bob")
        self.assertEqual(shell.spoken, ["Bob here, running on Haiku 4.5."])

    def test_announcement_follows_a_mid_session_model_switch(self):
        shell = _AgentShell()
        Agent._switch_agent(shell, "tom")
        shell.llm._ctx.convo_model = cfg.CONVO_MODELS["deepseek pro"]
        Agent._switch_agent(shell, "bob")
        Agent._switch_agent(shell, "tom")
        self.assertEqual(shell.spoken[-2:], [
            "Bob here, running on Haiku 4.5.",
            "Tom here, running on DeepSeek V4 Pro.",
        ])

    def test_every_persona_announces_a_friendly_model_name(self):
        # A raw api id ("claude-haiku-4-5") leaking into speech would be read
        # out character by character; every registry model must have a label.
        for key in agents.AGENTS:
            shell = _AgentShell()
            shell.llm.active = "__none__"  # force switch_to to do the work
            shell.llm._ctx.active_agent = "__none__"
            Agent._switch_agent(shell, key)
            said = shell.spoken[0]
            self.assertIn(agents.AGENTS[key]["name"], said)
            self.assertNotIn("claude-", said)
            self.assertNotIn("deepseek-v4", said)


if __name__ == "__main__":
    unittest.main()
