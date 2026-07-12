"""Unit tests for the set_conversation_model tool."""

import unittest

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
        ctx = self.ctx()
        for name, model_id in cfg.CONVO_MODELS.items():
            dispatch(ctx, "set_conversation_model", {"model": name})
            self.assertEqual(ctx.convo_model, model_id)

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

    def test_excluded_from_folder_dialogue(self):
        # The folder dialogue passes exclude={save_conversation_note,
        # set_conversation_model}; make sure the tool honors exclusion.
        names = {t["name"] for t in api_tools(
            exclude={"save_conversation_note", "set_conversation_model"})}
        self.assertNotIn("set_conversation_model", names)


if __name__ == "__main__":
    unittest.main()
