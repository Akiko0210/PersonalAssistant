"""Unit tests for the model tools (read + switch) and provider routing."""

import os
import unittest
from unittest.mock import patch

import agents
import config as cfg
from llm import Claude
from tools import ToolContext, dispatch, api_tools


class TestSetConversationModel(unittest.TestCase):
    def ctx(self):
        return ToolContext(convo_model=cfg.CONVO_MODEL)

    def test_registered(self):
        names = {t["name"] for t in api_tools()}
        self.assertIn("set_conversation_model", names)

    def test_switch_to_opus(self):
        ctx = self.ctx()
        out = dispatch(ctx, "set_conversation_model", {"model": "opus"})
        self.assertEqual(ctx.convo_model, cfg.CONVO_MODELS["opus"])
        self.assertIn("Opus", out)

    def test_switch_each_model(self):
        # DeepSeek switches are gated on the key being present; supply one so
        # this exercises the switch itself, not the gate (tested separately).
        ctx = self.ctx()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            for name, model_id in cfg.CONVO_MODELS.items():
                dispatch(ctx, "set_conversation_model", {"model": name})
                self.assertEqual(ctx.convo_model, model_id)

    def test_enum_matches_registry(self):
        # The schema enum is what the model sees; a CONVO_MODELS entry missing
        # from it would be silently unswitchable by voice.
        schema = next(t for t in api_tools()
                      if t["name"] == "set_conversation_model")
        enum = schema["input_schema"]["properties"]["model"]["enum"]
        self.assertEqual(set(enum), set(cfg.CONVO_MODELS))

    def test_unknown_model_leaves_choice_unchanged(self):
        ctx = self.ctx()
        before = ctx.convo_model
        out = dispatch(ctx, "set_conversation_model", {"model": "gpt"})
        self.assertEqual(ctx.convo_model, before)
        self.assertIn("Unknown model", out)

    def test_already_using_is_idempotent(self):
        ctx = ToolContext(convo_model=cfg.CONVO_MODELS["haiku"])
        out = dispatch(ctx, "set_conversation_model", {"model": "haiku"})
        self.assertIn("Already using", out)
        self.assertEqual(ctx.convo_model, cfg.CONVO_MODELS["haiku"])

    def test_deepseek_refused_without_key(self):
        ctx = self.ctx()
        before = ctx.convo_model
        env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            out = dispatch(ctx, "set_conversation_model", {"model": "deepseek"})
        self.assertEqual(ctx.convo_model, before)  # switch must not happen
        self.assertIn("DEEPSEEK_API_KEY", out)

    def test_deepseek_switches_with_key(self):
        ctx = self.ctx()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            out = dispatch(ctx, "set_conversation_model", {"model": "deepseek pro"})
        self.assertEqual(ctx.convo_model, "deepseek-v4-pro")
        self.assertIn("DeepSeek", out)

    def test_excluded_from_folder_dialogue(self):
        # The folder dialogue passes exclude={save_conversation_note,
        # set_conversation_model}; make sure the tool honors exclusion.
        names = {t["name"] for t in api_tools(
            exclude={"save_conversation_note", "set_conversation_model"})}
        self.assertNotIn("set_conversation_model", names)


class TestGetCurrentModel(unittest.TestCase):
    """The tool exists because the model answered this question wrong from
    shared history three times in one session; these pin the live read."""

    def test_registered_and_argument_free(self):
        schema = next(t for t in api_tools() if t["name"] == "get_current_model")
        self.assertFalse(schema["input_schema"].get("required"))

    def test_every_hat_can_answer_the_question(self):
        for key, agent in agents.AGENTS.items():
            self.assertIn("get_current_model", agent["tools"],
                          f"{key} cannot read its own model")

    def test_reports_the_live_override(self):
        ctx = ToolContext(active_agent="tom",
                          convo_model=cfg.CONVO_MODELS["deepseek pro"])
        out = dispatch(ctx, "get_current_model", {})
        self.assertIn("Tom", out)
        self.assertIn("DeepSeek V4 Pro", out)
        self.assertIn("DeepSeek", out)          # provider named too
        self.assertNotIn("deepseek-v4", out)    # never the raw api id

    def test_falls_back_to_the_persona_registry_default(self):
        ctx = ToolContext(active_agent="tom", convo_model=None)
        out = dispatch(ctx, "get_current_model", {})
        self.assertIn(cfg.convo_model_label(
            cfg.CONVO_MODELS[agents.AGENTS["tom"]["model"]]), out)

    def test_names_anthropic_as_the_provider_for_claude_models(self):
        ctx = ToolContext(active_agent="alice",
                          convo_model=cfg.CONVO_MODELS["opus"])
        out = dispatch(ctx, "get_current_model", {})
        self.assertIn("Opus 5", out)
        self.assertIn("Anthropic", out)


class TestGetCurrentModelAcrossPersonas(unittest.TestCase):
    """One shared history, per-hat model dials: reading the dial must follow
    the active hat, never the conversation."""

    def make_claude(self):
        c = Claude.__new__(Claude)
        c.active = "alice"
        c._model_overrides = {}
        c._ctx = ToolContext(active_agent="alice",
                             convo_model=cfg.CONVO_MODELS["haiku"])
        c.history = []  # switch_to appends its took-over marker here
        c._save_history = lambda: None       # no disk writes from tests
        c._write_agent_state = lambda: None
        return c

    def test_toms_opus_does_not_leak_into_alice(self):
        # Park Tom on Opus, switch to Alice (Haiku), ask: Alice must say Haiku.
        c = self.make_claude()
        c.switch_to("tom")
        c._ctx.convo_model = cfg.CONVO_MODELS["opus"]  # "switch to Opus"
        c.switch_to("alice")
        out = dispatch(c._ctx, "get_current_model", {})
        self.assertIn("Alice", out)
        self.assertIn("Haiku 4.5", out)
        self.assertNotIn("Opus", out)

    def test_and_tom_still_remembers_opus_on_return(self):
        c = self.make_claude()
        c.switch_to("tom")
        c._ctx.convo_model = cfg.CONVO_MODELS["opus"]
        c.switch_to("alice")
        c.switch_to("tom")
        out = dispatch(c._ctx, "get_current_model", {})
        self.assertIn("Tom", out)
        self.assertIn("Opus 5", out)

    def test_never_disagrees_with_the_announced_model(self):
        # active_model_label is what the switch announcement speaks; the tool
        # and the announcement must never tell the user different things.
        c = self.make_claude()
        for key in agents.AGENTS:
            c.switch_to(key)
            out = dispatch(c._ctx, "get_current_model", {})
            self.assertIn(c.active_model_label, out)
            self.assertIn(agents.AGENTS[key]["name"], out)


class TestModelIdentityPrompt(unittest.TestCase):
    def test_states_the_label_and_mandates_the_tool(self):
        block = cfg.model_identity_block("Haiku 4.5")
        self.assertIn("Haiku 4.5", block)
        self.assertIn("get_current_model", block)
        self.assertIn("set_conversation_model", block)

    def test_forbids_answering_from_shared_history(self):
        block = cfg.model_identity_block("Opus 5").lower()
        self.assertIn("shared", block)
        self.assertIn("must call get_current_model", block)
        self.assertIn("guess", block)


class TestModelProvider(unittest.TestCase):
    def test_claude_models_route_to_anthropic(self):
        for mid in ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"):
            self.assertEqual(cfg.model_provider(mid), "anthropic")

    def test_deepseek_models_route_to_deepseek(self):
        for mid in ("deepseek-v4-flash", "deepseek-v4-pro"):
            self.assertEqual(cfg.model_provider(mid), "deepseek")

    def test_every_convo_model_has_a_provider_and_label(self):
        for mid in cfg.CONVO_MODELS.values():
            self.assertIn(cfg.model_provider(mid), ("anthropic", "deepseek"))
            self.assertIn(mid, cfg.CONVO_MODEL_LABELS)
            # get_current_model speaks the provider aloud, so every model must
            # have a real name to speak — not the routing key, not a fallback.
            self.assertIn(cfg.provider_label(mid), ("Anthropic", "DeepSeek"))

    def test_none_and_empty_default_to_anthropic(self):
        self.assertEqual(cfg.model_provider(None), "anthropic")
        self.assertEqual(cfg.model_provider(""), "anthropic")


if __name__ == "__main__":
    unittest.main()
