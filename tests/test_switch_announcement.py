"""The spoken persona-switch announcement names the model it's running on.

Drives Agent._switch_agent unbound against a shell holding only the attributes
it touches — a real Agent would boot audio, TTS, Whisper and the note stores.
"""

import unittest
from types import SimpleNamespace

from brain import agents
import config as cfg
from voice_agent import Agent
from tests.llm_fixtures import make_claude


class _AgentShell:
    """Just the surface _switch_agent uses."""

    def __init__(self):
        self.llm = make_claude()
        self.tts = SimpleNamespace(set_voice=lambda *a: None,
                                   current_voice=lambda: "voice")
        self._speaking_voice = None
        self.log = SimpleNamespace(info=lambda *a: None)
        self.spoken = []

    def say(self, text, **kw):
        self.spoken.append(text)

    def _use_voice(self, hat):
        return Agent._use_voice(self, hat)


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
