"""Tests for the dashboard's config-override pipeline.

Covers the two halves of a dashboard edit reaching the agent:
- dashboard.validate_payload — the write side: only whitelisted keys, typed,
  bounded, with choices enforced.
- config.apply_overrides — the read side: values land on the module, bad
  values are skipped, everything else falls back to the coded default.
"""

import unittest

import config as cfg
import dashboard


class ApplyOverridesTests(unittest.TestCase):
    def setUp(self):
        self._saved = {name: getattr(cfg, name) for name in cfg.OVERRIDABLE}

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(cfg, name, value)

    def test_applies_whitelisted_values(self):
        applied = cfg.apply_overrides({"CONVO_ENDPOINT_MS": 500,
                                       "CONTINUATION_SETTLE_MS": 250})
        self.assertEqual(sorted(applied),
                         ["CONTINUATION_SETTLE_MS", "CONVO_ENDPOINT_MS"])
        self.assertEqual(cfg.CONVO_ENDPOINT_MS, 500)
        self.assertEqual(cfg.CONTINUATION_SETTLE_MS, 250)

    def test_ignores_unknown_names(self):
        applied = cfg.apply_overrides({"BASE_DIR": "/tmp/evil", "NOPE": 1})
        self.assertEqual(applied, [])
        self.assertEqual(cfg.BASE_DIR, self._saved.get("BASE_DIR", cfg.BASE_DIR))

    def test_skips_uncastable_values(self):
        before = cfg.CONVO_ENDPOINT_MS
        applied = cfg.apply_overrides({"CONVO_ENDPOINT_MS": "not a number"})
        self.assertEqual(applied, [])
        self.assertEqual(cfg.CONVO_ENDPOINT_MS, before)

    def test_backchannel_words_become_frozenset(self):
        cfg.apply_overrides({"BACKCHANNEL_WORDS": ["Yeah", " OK ", "yeah"]})
        self.assertEqual(cfg.BACKCHANNEL_WORDS, frozenset({"yeah", "ok"}))

    def test_non_dict_is_a_noop(self):
        self.assertEqual(cfg.apply_overrides(["CONVO_ENDPOINT_MS"]), [])

    def test_defaults_snapshot_covers_every_overridable(self):
        self.assertEqual(set(cfg.CONFIG_DEFAULTS), set(cfg.OVERRIDABLE))


class ValidatePayloadTests(unittest.TestCase):
    def test_valid_values_pass(self):
        overrides, errors = dashboard.validate_payload({
            "CONVO_ENDPOINT_MS": 600,
            "TRIGGER_RATIO": 0.5,
            "BARGE_IN": False,
            "CONVO_MODEL": "claude-sonnet-5",
            "BACKCHANNEL_WORDS": ["yeah", "OK"],
        })
        self.assertEqual(errors, {})
        self.assertEqual(overrides["CONVO_ENDPOINT_MS"], 600)
        self.assertEqual(overrides["BACKCHANNEL_WORDS"], ["ok", "yeah"])

    def test_unknown_key_rejected(self):
        overrides, errors = dashboard.validate_payload({"BASE_DIR": "x"})
        self.assertEqual(overrides, {})
        self.assertIn("BASE_DIR", errors)

    def test_out_of_bounds_rejected(self):
        _, errors = dashboard.validate_payload({"CONVO_ENDPOINT_MS": 99999})
        self.assertIn("CONVO_ENDPOINT_MS", errors)
        _, errors = dashboard.validate_payload({"VAD_AGGRESSIVENESS": -1})
        self.assertIn("VAD_AGGRESSIVENESS", errors)

    def test_bad_choice_rejected(self):
        _, errors = dashboard.validate_payload({"CONVO_MODEL": "gpt-4"})
        self.assertIn("CONVO_MODEL", errors)
        _, errors = dashboard.validate_payload({"SUMMARY_MODEL": "gpt-4"})
        self.assertIn("SUMMARY_MODEL", errors)

    def test_summary_model_is_a_dropdown_including_the_default(self):
        meta = dashboard.TUNABLES_BY_KEY["SUMMARY_MODEL"]
        self.assertEqual(meta["type"], "choice")
        values = [c["value"] for c in meta["choices"]]
        # the current coded default must be selectable, or the dropdown would
        # render a value the user never chose
        self.assertIn(cfg.SUMMARY_MODEL, values)
        overrides, errors = dashboard.validate_payload({"SUMMARY_MODEL": cfg.SUMMARY_MODEL})
        self.assertEqual(errors, {})
        self.assertEqual(overrides["SUMMARY_MODEL"], cfg.SUMMARY_MODEL)

    def test_bool_must_be_bool(self):
        _, errors = dashboard.validate_payload({"BARGE_IN": "yes"})
        self.assertIn("BARGE_IN", errors)

    def test_null_means_reset(self):
        overrides, errors = dashboard.validate_payload({"CONVO_ENDPOINT_MS": None})
        self.assertEqual(errors, {})
        self.assertNotIn("CONVO_ENDPOINT_MS", overrides)

    def test_nullable_text_accepts_empty(self):
        overrides, errors = dashboard.validate_payload({"TTS_VOICE": ""})
        self.assertEqual(errors, {})
        self.assertIsNone(overrides["TTS_VOICE"])

    def test_empty_word_list_rejected(self):
        _, errors = dashboard.validate_payload({"BACKCHANNEL_WORDS": []})
        self.assertIn("BACKCHANNEL_WORDS", errors)

    def test_every_tunable_has_config_default(self):
        # Each dashboard tunable must be overridable in config, or the form
        # would silently show a field the agent never reads.
        for t in dashboard.TUNABLES:
            self.assertIn(t["key"], cfg.OVERRIDABLE)
            self.assertIn(t["key"], cfg.CONFIG_DEFAULTS)


class PathGuardTests(unittest.TestCase):
    def test_note_id_pattern(self):
        self.assertTrue(dashboard.NOTE_ID_RE.match("note_2026-06-22_141600"))
        self.assertFalse(dashboard.NOTE_ID_RE.match("../secrets"))
        self.assertFalse(dashboard.NOTE_ID_RE.match("note_a/../../b"))

    def test_log_name_pattern(self):
        self.assertTrue(dashboard.LOG_NAME_RE.match("session_2026-07-19.log"))
        self.assertFalse(dashboard.LOG_NAME_RE.match("..\\config.py"))
        self.assertFalse(dashboard.LOG_NAME_RE.match("session_x.log.exe"))


if __name__ == "__main__":
    unittest.main()
