"""Unit tests for the set_conversation_model tool and provider routing."""

import os
import unittest
from unittest.mock import patch

import config as cfg
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

    def test_none_and_empty_default_to_anthropic(self):
        self.assertEqual(cfg.model_provider(None), "anthropic")
        self.assertEqual(cfg.model_provider(""), "anthropic")


if __name__ == "__main__":
    unittest.main()
